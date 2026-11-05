import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from greenwash.analyse import analyse_pdf, analyse_sentences, scorecard  # noqa: E402
from greenwash.models import load_env_claims, load_greenwashing  # noqa: E402
from greenwash.paths import METRICS, PROCESSED, SAMPLES  # noqa: E402
from greenwash.pdftext import sentences  # noqa: E402


def test_datasets_shape():
    env = load_env_claims()
    assert {k: len(v) for k, v in env.items()} == {"train": 2117, "dev": 265, "test": 265}
    gw = load_greenwashing()
    assert len(gw) == 671 and set(gw.y.unique()) == {0, 1}


def test_pinned_model_metrics():
    m = json.loads(METRICS.read_text())
    cd, gr = m["claim_detector"], m["greenwashing_risk"]
    assert cd["macro_f1"] > 0.84 and cd["roc_auc"] > 0.92          # beats the paper's TF-IDF SVM (0.725)
    assert gr["signals_plus_words_logreg_cv"]["roc_auc"] > 0.75
    assert gr["rule_score_only"]["roc_auc"] > 0.6


def test_detector_separates_obvious_cases():
    df = analyse_sentences([(1, "We will reduce our greenhouse gas emissions by 30% by 2030 against a 2019 baseline."),
                            (1, "Revenue grew by 12% driven by strong demand in our retail segment."),
                            (1, "The board met four times to review the company's audit and remuneration policies.")])
    assert df.iloc[0].is_claim and not df.iloc[1].is_claim and not df.iloc[2].is_claim


def test_fictional_report_end_to_end():
    df = analyse_sentences([(1, s) for s in sentences((SAMPLES / "fictional_example_report.txt").read_text())])
    sc = scorecard(df, 1)
    assert sc["claims"] >= 8 and sc["targets"] >= 1
    worst = df[df.is_claim].sort_values("specificity").iloc[0]
    assert worst.specificity <= 30


def test_pdf_pipeline_roundtrip(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    p = tmp_path / "t.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    y = 800
    for line in ["Our operations are eco-friendly and sustainable for the planet and its people.",
                 "We target net zero greenhouse gas emissions across Scope 1 and 2 by 2040."]:
        c.drawString(40, y, line)
        y -= 20
    c.showPage()
    c.save()
    df, n = analyse_pdf(p.read_bytes())
    assert n == 1 and len(df) == 2 and df.is_claim.all()
    assert reportlab


def test_case_studies_file():
    cs = json.loads((PROCESSED / "case_studies.json").read_text())
    assert set(cs["reports"]) == {"access_2025", "dangote_2024", "seplat_2023"}
    for r in cs["reports"].values():
        assert r["scorecard"]["claims"] > 50 and len(r["weakest"]) <= 6
        assert all(len(w["sentence"].split()) <= 60 for w in r["weakest"] + r["strongest"])  # short excerpts only
