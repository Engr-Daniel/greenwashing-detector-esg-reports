"""Analyse the Nigerian case-study reports and save compact results for the app and README.

The PDFs themselves are not redistributed (copyright). Put them in data/raw/reports/ and run:
    python src/greenwash/case_studies.py
Only scorecards, signal shares, extracted targets and a handful of short quoted excerpts (with page numbers) are saved.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenwash.analyse import analyse_pdf, scorecard, signal_counts  # noqa: E402
from greenwash.models import load_env_claims  # noqa: E402
from greenwash.paths import PROCESSED, RAW  # noqa: E402
from greenwash.scoring import specificity_score  # noqa: E402

REPORTS = {
    "access_2025": {"company": "Access Holdings Plc", "sector": "Banking", "year": 2025,
                    "title": "Standalone Sustainability Report 2025", "file": "access_2025.pdf"},
    "dangote_2024": {"company": "Dangote Cement Plc", "sector": "Cement", "year": 2024,
                     "title": "Sustainability Report 2024 (Annual Report extract)", "file": "dangote_2024.pdf"},
    "seplat_2023": {"company": "Seplat Energy Plc", "sector": "Oil & gas", "year": 2023,
                    "title": "Sustainability Report 2023", "file": "seplat_2023.pdf"},
}
N_EXCERPTS = 6


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def benchmark() -> dict:
    """Specificity distribution of expert-labelled claims from international listed companies."""
    env = pd.concat(load_env_claims().values())
    sc = env[env.label == 1].text.map(lambda t: specificity_score(t)[0])
    return {"n": int(len(sc)), "median": float(sc.median()), "mean": float(sc.mean()),
            "hist": sc.value_counts(bins=list(range(0, 105, 5)), sort=False).astype(int).tolist()}


def main():
    out = {"benchmark": benchmark(), "reports": {}}
    for key, meta in REPORTS.items():
        pdf = RAW / "reports" / meta["file"]
        if not pdf.exists():
            print(f"skip {key}: {pdf} not found")
            continue
        df, n = analyse_pdf(pdf.read_bytes())
        c = df[df.is_claim].copy()
        sc = {k: _clean(v) for k, v in scorecard(df, n).items()}
        cols = ["page", "sentence", "specificity", "gw_risk", "red_flags", "evidence"]
        top = c[c.p_claim >= 0.6]                      # confident claims only for the "weakest" excerpts
        weakest = top.sort_values(["specificity", "gw_risk"], ascending=[True, False]).head(N_EXCERPTS)[cols]
        strongest = c.sort_values(["specificity", "gw_risk"], ascending=[False, True]).head(N_EXCERPTS)[cols]
        targets = [dict(page=int(r.page), sentence=r.sentence, **r.target) for r in c[c.target.notna()].itertuples()]
        out["reports"][key] = {
            **meta, "scorecard": sc,
            "signals": signal_counts(df).to_dict(orient="records"),
            "spec_hist": c.specificity.value_counts(bins=list(range(0, 105, 5)), sort=False).astype(int).tolist(),
            "by_page": c.groupby("page").agg(claims=("sentence", "size"), specificity=("specificity", "mean"))
                        .reset_index().round(1).to_dict(orient="records"),
            "weakest": weakest.to_dict(orient="records"), "strongest": strongest.to_dict(orient="records"),
            "targets": targets,
        }
        print(key, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sc.items()})
    (PROCESSED / "case_studies.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
