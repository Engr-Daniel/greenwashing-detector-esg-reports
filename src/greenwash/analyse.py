"""End-to-end report analysis: PDF -> sentences -> environmental claims -> scores -> report scorecard.

CLI:  python src/greenwash/analyse.py path/to/report.pdf [--name "Company 2024"]
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenwash.paths import CLAIM_MODEL, RISK_MODEL  # noqa: E402
from greenwash.pdftext import document_sentences, pdf_pages  # noqa: E402
from greenwash.scoring import grade, load_signals, specificity_score  # noqa: E402
from greenwash.targets import extract_target  # noqa: E402

ENV_RX = re.compile(r"\b(?:emission\w*|carbon|co2|co₂|ghg|greenhouse|energy|solar|renewable\w*|water|waste|recycl\w*|plastic|"
                    r"flar\w*|methane|climate|biodiversity|trees?|forest\w*|deforestation|environment\w*|fuel\w*|diesel|"
                    r"electricity|pollut\w*|spill\w*|net[- ]zero|decarboni\w*|footprint|packaging|circular\w*)\b", re.I)
RED = [s.name for s in load_signals() if s.kind == "red"]
GREEN = [s.name for s in load_signals() if s.kind == "green"]


@lru_cache(maxsize=1)
def models():
    """Load the saved models; retrain (≈10 s) if they are missing or were saved by an incompatible scikit-learn."""
    try:
        return joblib.load(CLAIM_MODEL), joblib.load(RISK_MODEL)
    except Exception:  # noqa: BLE001
        from greenwash.models import main as train
        train()
        return joblib.load(CLAIM_MODEL), joblib.load(RISK_MODEL)


def analyse_sentences(rows: list[tuple[int, str]], claim_threshold: float | None = None) -> pd.DataFrame:
    """Score [(page, sentence)] rows. Returns one row per sentence with claim probability, scores and signals."""
    if not rows:
        return pd.DataFrame(columns=["page", "sentence", "p_claim", "is_claim"])
    claim_m, risk_m = models()
    th = claim_threshold if claim_threshold is not None else getattr(claim_m, "threshold_", 0.5)
    df = pd.DataFrame(rows, columns=["page", "sentence"]).drop_duplicates("sentence").reset_index(drop=True)
    df["p_claim"] = claim_m.predict_proba(df.sentence)[:, 1]
    sc = df.sentence.map(specificity_score)
    df["specificity"] = [s for s, _ in sc]
    df["signals"] = [f for _, f in sc]
    df["red_flags"] = df.signals.map(lambda f: [k for k in f if k in RED])
    df["evidence"] = df.signals.map(lambda f: [k for k in f if k in GREEN])
    df["gw_risk"] = risk_m.predict_proba(df[["sentence"]].rename(columns={"sentence": "claim"}))[:, 1]
    df["target"] = df.sentence.map(extract_target)
    df["grade"] = df.specificity.map(grade)
    df["is_claim"] = claim_mask(df, th)
    return df


def claim_mask(df: pd.DataFrame, th: float) -> pd.Series:
    """Environmental statement = detector says claim, OR a measurable target, OR a specific environmental disclosure
    the detector under-weights (quantified + another evidence signal + environmental keyword + some claim probability).
    Without the last two rules the most specific statements would be dropped and reports would look less verifiable
    than they are."""
    specific = ((df.p_claim >= 0.25) & (df.evidence.map(len) >= 2) & df.evidence.map(lambda e: "quantified" in e)
                & df.sentence.str.contains(ENV_RX))
    return (df.p_claim >= th) | df.target.notna() | specific


def analyse_pdf(data: bytes, **kw) -> tuple[pd.DataFrame, int]:
    pages = pdf_pages(data)
    return analyse_sentences(document_sentences(pages), **kw), len(pages)


def scorecard(df: pd.DataFrame, n_pages: int | None = None) -> dict:
    """Report-level summary of environmental claims."""
    c = df[df.is_claim] if len(df) else df
    t = c.target.dropna() if len(c) else pd.Series(dtype=object)
    tt = pd.DataFrame(list(t)) if len(t) else pd.DataFrame(columns=["target_year", "baseline_year", "reduction_pct",
                                                                     "scope", "completeness"])
    n = len(c)
    share = (lambda m: float(m.mean()) if n else np.nan)
    out = {
        "pages": n_pages, "sentences": int(len(df)), "claims": int(n),
        "claim_density": n / max(len(df), 1),
        "median_specificity": float(c.specificity.median()) if n else np.nan,
        "mean_specificity": float(c.specificity.mean()) if n else np.nan,
        "share_vague": share(c.red_flags.map(lambda r: "vague_term" in r)) if n else np.nan,
        "share_any_red": share(c.red_flags.map(bool)) if n else np.nan,
        "share_quantified": share(c.evidence.map(lambda e: "quantified" in e)) if n else np.nan,
        "share_verified": share(c.evidence.map(lambda e: "verification" in e)) if n else np.nan,
        "share_high_risk": share(c.gw_risk >= 0.5) if n else np.nan,
        "targets": int(len(tt)),
        "targets_with_year": int(tt.target_year.notna().sum()) if len(tt) else 0,
        "targets_with_baseline": int(tt.baseline_year.notna().sum()) if len(tt) else 0,
        "targets_complete": int((tt.completeness >= 0.75).sum()) if len(tt) else 0,
    }
    # report grade: specificity of claims, adjusted for target quality
    tq = (tt.completeness.mean() if len(tt) else 0.0)
    out["report_score"] = float(np.clip(0.75 * (out["mean_specificity"] if n else 0) + 25 * tq, 0, 100))
    out["report_grade"] = grade(out["report_score"])
    return out


MIX = [("vague_only", "Red flags, no evidence"), ("mixed", "Red flags + evidence"),
       ("specific", "Evidence, no red flags"), ("neutral", "Neither")]


def claim_mix(df: pd.DataFrame) -> dict:
    """Split claims into four mutually exclusive groups (shares sum to 1)."""
    c = df[df.is_claim]
    red, ev = c.red_flags.map(bool), c.evidence.map(bool)
    n = max(len(c), 1)
    counts = {"vague_only": int((red & ~ev).sum()), "mixed": int((red & ev).sum()),
              "specific": int((~red & ev).sum()), "neutral": int((~red & ~ev).sum())}
    return {k: {"claims": v, "share": v / n} for k, v in counts.items()}


def signal_counts(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df.is_claim]
    sig = {s.name: (s.label, s.kind) for s in load_signals()}
    rows = [{"signal": sig[k][0], "kind": sig[k][1], "claims": int(c.signals.map(lambda f, k=k: k in f).sum())}
            for k in sig]
    out = pd.DataFrame(rows)
    out["share"] = out.claims / max(len(c), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--name")
    a = ap.parse_args()
    df, n = analyse_pdf(Path(a.pdf).read_bytes())
    print(a.name or a.pdf, scorecard(df, n))


if __name__ == "__main__":
    main()
