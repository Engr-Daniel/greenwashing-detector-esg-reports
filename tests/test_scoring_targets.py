import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from greenwash.pdftext import clean, sentences  # noqa: E402
from greenwash.scoring import BASE, detect, features, grade, load_signals, specificity_score  # noqa: E402
from greenwash.targets import extract_target  # noqa: E402

VAGUE = "Our products are eco-friendly and we are an industry-leading sustainable company."
SPECIFIC = ("Our Scope 1 and 2 emissions fell by 18% from 41,200 tCO2e in our 2021 baseline year to 33,800 tCO2e in "
            "2025, verified by an independent third party with limited assurance.")


def test_lexicon_loads_and_regexes_compile():
    sig = load_signals()
    assert len(sig) == 13
    assert {s.kind for s in sig} == {"red", "green"}
    assert all(s.weight < 0 for s in sig if s.kind == "red") and all(s.weight > 0 for s in sig if s.kind == "green")


def test_vague_vs_specific_ordering():
    v, fv = specificity_score(VAGUE)
    s, fs = specificity_score(SPECIFIC)
    assert v < BASE < s
    assert {"vague_term", "superlative"} <= set(fv)
    assert {"quantified", "baseline", "scope_defined", "verification"} <= set(fs)
    assert grade(s) == "A" and grade(v) in {"D", "E"}


def test_neutral_sentence_scores_base():
    assert specificity_score("The board met four times during the year to review strategy.")[0] == BASE


def test_green_exceptions_not_flagged_as_vague():
    assert "vague_term" not in detect("We issued a green bond and reduced greenhouse gas emissions.")


def test_offsets_and_absolute_claims():
    f = detect("Our cornflakes are a carbon neutral product thanks to carbon credits.")
    assert {"absolute_claim", "offset_reliance"} <= set(f)


def test_scores_are_clipped():
    worst = "Our entire eco-friendly, carbon neutral, industry-leading products dramatically offset everything; we aim to."
    assert 0 <= specificity_score(worst)[0] <= 100


def test_features_shape():
    X = features([VAGUE, SPECIFIC])
    assert X.shape == (2, 13) and set(X.values.ravel()) <= {0, 1}


def test_target_extraction_complete():
    t = extract_target("Seplat aims to reduce Scope 1 and 2 emissions intensity by 65% from 2021 levels by 2030.")
    assert t["target_year"] == 2030 and t["baseline_year"] == 2021 and t["reduction_pct"] == 65
    assert t["scope"].startswith("Scope 1") and t["completeness"] == 1.0


def test_target_rejects_past_results_and_vague_commitments():
    assert extract_target("Emissions reduced by 49%, from 3,293 tCO2e in 2024 to 1,668 tCO2e in 2025.") is None
    assert extract_target("We remain committed to environmental sustainability and reducing our footprint.") is None
    assert extract_target("Dangote Cement is committed to reducing carbon emissions by 20% by the year 2030.")["target_year"] == 2030


def test_sentence_splitting():
    text = "We cut emissions by 10% vs. 2020 levels at our plant.\nThe board met.\n\nWe installed 2 MW of solar at the Kano site in 2025."
    out = sentences(text, min_words=5)
    assert len(out) == 2 and out[0].startswith("We cut emissions by 10% vs. 2020")
    assert "  " not in clean("a  b\n c")
