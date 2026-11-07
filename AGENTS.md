# AGENTS.md

Instructions for AI coding agents and human collaborators working in this repository.

## Project at a glance
An NLP tool and Streamlit app that finds environmental claims in ESG / sustainability reports (PDF) and scores how
specific and verifiable they are. Personal portfolio project by Daniel Oyewale.

**Read `MEMORY.md` (context & decisions) and `TASK.md` (priorities) before starting work.**

## Commands
```bash
pip install -r requirements-dev.txt          # app only: requirements.txt
pytest -q                                    # must pass before committing
streamlit run app/streamlit_app.py
python src/greenwash/models.py               # retrain both models, write data/processed/model_metrics.json
python src/greenwash/case_studies.py         # needs PDFs in data/raw/reports/ (git-ignored)
python src/greenwash/figures.py              # README figures
```

## Code map
| Path | Purpose |
|---|---|
| `src/greenwash/pdftext.py` | PDF text extraction, cleaning, sentence splitting |
| `src/greenwash/scoring.py` | Signal detection from `data/reference/red_flags.csv`, specificity score, grades |
| `src/greenwash/targets.py` | Measurable-target extraction (year, baseline, %, scope, completeness) |
| `src/greenwash/models.py` | Claim detector + language-risk model: training & evaluation |
| `src/greenwash/analyse.py` | End-to-end analysis, `claim_mask`, scorecard |
| `app/streamlit_app.py` | UI only. No scoring logic |

## Rules
1. **Never present scores as proof of greenwashing.** The wording is always "specificity / verifiability / disclosure
   quality". Keep the disclaimer in the app, README and audit note.
2. **Copyright:** don't commit company PDFs or full extracted text. Case studies store scorecards and ≤6 short
   excerpts per list, with page numbers.
3. **Signal weights are normative:** every row in `red_flags.csv` needs a rationale and a regulatory source. Don't
   tune weights to the EmeraldMind labels (it would leak into the reported AUC).
4. **Pin scikit-learn** in `requirements.txt` to the version that saved `models/*.joblib`. Retrain after upgrading.
5. Feature functions used inside saved pipelines must live in `src/greenwash` modules (not `__main__`) so they unpickle.
6. Keep the app mobile-friendly: KPI rows as one HTML grid, legends below charts on phones.
7. After each session update `TASK.md` and record decisions in `MEMORY.md`.

## Git attribution
Never add `Co-Authored-By`, `Claude-Session` or other AI attribution lines to commits or PRs (see CLAUDE.md).

## Commit conventions
`feat:`, `fix:`, `docs:`, `data:`, `test:`, `refactor:`, `chore:`.
