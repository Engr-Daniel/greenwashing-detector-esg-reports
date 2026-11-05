from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
SAMPLES = ROOT / "sample_reports"

CLAIM_MODEL = MODELS / "claim_detector.joblib"
RISK_MODEL = MODELS / "greenwashing_risk.joblib"
METRICS = PROCESSED / "model_metrics.json"
LEXICON = REFERENCE / "red_flags.csv"
