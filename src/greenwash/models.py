"""Train and evaluate (1) the environmental-claim detector and (2) the greenwashing-risk model.

Run:  python src/greenwash/models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FunctionTransformer, Pipeline, make_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenwash.paths import CLAIM_MODEL, METRICS, MODELS, PROCESSED, RAW, RISK_MODEL  # noqa: E402
from greenwash.scoring import features, signal_matrix, specificity_score  # noqa: E402

SEED = 42


# ----------------------------------------------------------------------------- data
def load_env_claims() -> dict[str, pd.DataFrame]:
    """Stammbach et al. (2023) expert-annotated environmental claims: train / dev / test."""
    return {s: pd.read_json(RAW / f"envclaims_{s}.jsonl", lines=True) for s in ("train", "dev", "test")}


def load_greenwashing() -> pd.DataFrame:
    """EmeraldMind claims (620) + green_claims (51) with greenwashing gold labels."""
    a = pd.read_csv(RAW / "emerald_data.csv")[["claim", "gold_label", "company"]].assign(source="emerald_data")
    b = pd.read_csv(RAW / "green_claims.csv")
    b = b.assign(claim=b.rephrased_claim.fillna(b.claim))[["claim", "gold_label", "company"]].assign(source="green_claims")
    df = pd.concat([a, b], ignore_index=True).dropna(subset=["claim"])
    df["y"] = (df.gold_label == "greenwashing").astype(int)
    return df.drop_duplicates("claim").reset_index(drop=True)


# ----------------------------------------------------------------------------- claim detector
def claim_pipeline(C: float = 4.0) -> Pipeline:
    vec = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000),
    )
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=4000,
                                                                random_state=SEED))])


def train_claim_detector(d: dict[str, pd.DataFrame]) -> tuple[Pipeline, dict]:
    best = None
    for C in (0.5, 1, 2, 4, 8, 16):                         # tune C on dev
        m = claim_pipeline(C).fit(d["train"].text, d["train"].label)
        p = m.predict_proba(d["dev"].text)[:, 1]
        for th in np.arange(0.3, 0.71, 0.05):               # and the decision threshold
            f = f1_score(d["dev"].label, p >= th)
            if best is None or f > best[0]:
                best = (f, C, round(float(th), 2))
    _, C, th = best
    m = claim_pipeline(C).fit(pd.concat([d["train"].text, d["dev"].text]), pd.concat([d["train"].label, d["dev"].label]))
    p = m.predict_proba(d["test"].text)[:, 1]
    yhat = p >= th
    y = d["test"].label
    metrics = {"C": C, "threshold": th, "test_n": int(len(y)), "test_pos": int(y.sum()),
               "precision": precision_score(y, yhat), "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
               "roc_auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
               "macro_f1": f1_score(y, yhat, average="macro"), "accuracy_micro_f1": float((yhat == y).mean()),
               "paper_test": {"TF-IDF SVM": {"micro_f1": 0.817, "macro_f1": 0.725},
                              "DistilBERT": {"micro_f1": 0.847, "macro_f1": 0.801},
                              "ClimateBERT": {"micro_f1": 0.867, "macro_f1": 0.819},
                              "RoBERTa-large": {"micro_f1": 0.907, "macro_f1": 0.868}},
               "note": "Paper scores: Stammbach et al. (2023) Table 2, test split (300 sentences in the paper; "
                       "the public release has 265)."}
    m.threshold_ = th
    return m, metrics


# ----------------------------------------------------------------------------- greenwashing risk
def risk_pipeline() -> Pipeline:
    """Signal features (interpretable) + word TF-IDF -> logistic regression."""
    ct = ColumnTransformer([
        ("signals", FunctionTransformer(signal_matrix, validate=False), "claim"),
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "claim"),
    ])
    return Pipeline([("x", ct), ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000,
                                                           random_state=SEED))])


def evaluate_risk(df: pd.DataFrame) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    y = df.y.values
    rule = 100 - df.claim.map(lambda t: specificity_score(t)[0]).values          # higher = riskier
    sig = features(df.claim)
    p_sig = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=2000), sig, y, cv=cv,
                              method="predict_proba")[:, 1]
    p_full = cross_val_predict(risk_pipeline(), df[["claim"]], y, cv=cv, method="predict_proba")[:, 1]

    def pack(score, th=0.5):
        out = {"roc_auc": roc_auc_score(y, score), "avg_precision": average_precision_score(y, score)}
        if th is not None:
            pr, rc, f, _ = precision_recall_fscore_support(y, score >= th, average="binary")
            out.update(precision=pr, recall=rc, f1=f)
        return out
    res = {"n": int(len(y)), "greenwashing_share": float(y.mean()),
           "rule_score_only": pack(rule, None),
           "signals_logreg_cv": pack(p_sig),
           "signals_plus_words_logreg_cv": pack(p_full)}
    # which signals separate the classes (share of claims with the signal, by label)
    lift = sig.groupby(y).mean().T.rename(columns={0: "not_greenwashing", 1: "greenwashing"})
    res["signal_prevalence"] = lift.round(3).to_dict(orient="index")
    return res


def main():
    MODELS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    env = load_env_claims()
    claim_model, claim_metrics = train_claim_detector(env)
    joblib.dump(claim_model, CLAIM_MODEL, compress=3)
    gw = load_greenwashing()
    risk_metrics = evaluate_risk(gw)
    risk_model = risk_pipeline().fit(gw[["claim"]], gw.y)
    joblib.dump(risk_model, RISK_MODEL, compress=3)
    metrics = {"claim_detector": claim_metrics, "greenwashing_risk": risk_metrics,
               "data": {"env_claims": {k: int(len(v)) for k, v in env.items()}, "greenwashing_claims": int(len(gw))}}
    METRICS.write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "signal_prevalence"} if isinstance(v, dict) else v
                      for k, v in metrics.items()}, indent=2, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
