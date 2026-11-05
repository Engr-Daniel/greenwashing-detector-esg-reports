# MEMORY.md

Long-term project memory: context, decisions and lessons. Newest first.

## Context
- **Owner:** Daniel Oyewale (Petgrave.io). Portfolio project 5 (bonus AI project). Others: 1 carbon-footprint-esg-dashboard
  (done), 2 Petgrave Data Strategy Blueprint case study (paused), 3 solar-minigrid-site-suitability-nigeria (done),
  4 climate risk & crop yields (next).
- **Build constraints:** Hugging Face, Zenodo and company websites were blocked from the build environment, so no
  transformer downloads. Data came from GitHub. The owner supplied the three Nigerian report PDFs.

## Results snapshot
- Claim detector: macro-F1 0.864, accuracy 0.894, AUC 0.941 (C=1, threshold 0.55)
- Language risk: CV AUC 0.78 (signals + words); rules alone 0.64
- Case studies: Access 2025 C (score 49, 108 claims), Dangote 2024 C (46, 287), Seplat 2023 D (42, 258)

## Decision log
| Date | Decision | Reason |
|---|---|---|
| 2026-11-05 | TF-IDF + LR claim detector instead of ClimateBERT | No HF access. Matches RoBERTa-large macro-F1, CPU-fast, free to host |
| 2026-11-05 | Specificity score = transparent rules with normative weights; ML risk as secondary signal | Explainable to regulators and investors; avoids overfitting the small, semi-synthetic labelled set |
| 2026-11-05 | Strict target definition (year, or target word + %) | Generic "commitment to sustainability" inflated target counts |
| 2026-11-05 | `claim_mask` adds measurable targets and specific environmental disclosures | Detector under-weights numeric disclosures. Without this, good reporting would be penalised |
| 2026-11-05 | Case-study PDFs git-ignored; only scorecards and short excerpts committed | Copyright |
| 2026-11-05 | scikit-learn pinned to 1.9.1; app retrains if models fail to load | Joblib pickles are version-specific |
| 2026-11-05 | Optional LLM second opinion with the user's own Anthropic key | Satisfies the "LLM API" brief without making the app depend on a paid key |

## Lessons learned
- Python `re` lookbehind must be fixed-width, so the sentence splitter merges abbreviations after splitting.
- Functions inside sklearn pipelines must be importable (not `__main__`) to unpickle in the app.
- `str.contains` with capturing groups warns; use `(?:...)`.
