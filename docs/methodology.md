# Methodology

## 1. Question
*How specific and verifiable are the environmental claims in a corporate sustainability report?*

The tool screens **disclosure quality**. It does **not** decide whether a company is greenwashing. That needs outside
evidence (actual emissions, capex, regulatory findings, news), and a precise claim can still be false.

## 2. Pipeline
| Step | Module | What happens |
|---|---|---|
| Read | `pdftext.py` | `pypdf` extracts text per page. De-hyphenation, bullet and paragraph breaks, abbreviation-aware sentence splitting. Drops fragments (<6 words), run-ons (>60 words), number-heavy table rows and title-case runs (headings, infographics, SDG lists) |
| Detect claims | `models.py` | Word (1–2) + character (3–5) TF-IDF, then logistic regression (class-balanced). C and threshold tuned on dev, refit on train+dev |
| Score | `scoring.py` | 13 transparent signals (below), base 50, clipped to 0–100, grades A–E |
| Language risk | `models.py` | Logistic regression on the 13 signals + word TF-IDF, trained on labelled greenwashing claims |
| Targets | `targets.py` | Target year, baseline, % change, scope, completeness |
| Report scorecard | `analyse.py` | Shares of claims with each signal; report score = 0.75 × mean claim specificity + 25 × mean target completeness |

**Which sentences count as environmental claims.** A sentence counts if the detector's probability is at least the
threshold (default 0.55), **or** it states a measurable target, **or** it is a specific environmental disclosure:
claim probability ≥ 0.25, quantified, at least one more evidence signal, and an environmental keyword. The last two
rules exist because the training data under-represents numeric performance statements. Without them the most
specific sentences would be dropped and every report would look *less* verifiable than it is.

## 3. Data
| Dataset | Use | Size | Licence |
|---|---|---|---|
| Environmental claims, Stammbach, Webersinke, Bingler, Kraus & Leippold (ACL 2023) ([GitHub](https://github.com/dominiksinsaarland/environmental_claims)) | Claim detector | 2,647 sentences (train 2,117 / dev 265 / test 265), from annual reports, sustainability reports and earnings calls of listed firms, 4 expert annotators each | Research use; commercial use needs the authors' permission |
| EmeraldMind (Kaoukis et al., WWW 2026) ([GitHub](https://github.com/ai4greenwashing/EmeraldMind)) | Language-risk model, signal validation | 671 claims (620 semi-synthetic + 51 real) labelled *greenwashing / not greenwashing*, with justifications | Apache 2.0 |
| Access Holdings 2025, Dangote Cement 2024, Seplat Energy 2023 sustainability reports | Case studies | 63 / 72 / 122 pages | Company copyright: PDFs not redistributed; only short excerpts with page numbers |

## 4. Specificity signals (`data/reference/red_flags.csv`)
| Signal | Points | Basis |
|---|---|---|
| Vague or generic term ("eco-friendly", "sustainable", "green") | −12 | EU Directive 2024/825 Annex I; CMA Green Claims Code principle 2 |
| Absolute or neutrality claim ("carbon neutral", "zero impact", "100% green") | −15 | EU 2024/825 Annex I 4c (offset-based product neutrality claims banned) |
| Superlative or leadership ("industry-leading", "pioneer") | −10 | CMA principle 5 (fair comparisons) |
| Overgeneralisation ("entire supply chain", "all our products") | −10 | CMA principle 3; EmeraldMind Type 3 |
| Exaggerated impact ("significantly", "transform the industry") | −10 | CMA principle 1; EmeraldMind Type 4 |
| Relies on offsets ("carbon credits", "tree planting") | −12 | EU 2024/825; SBTi Net-Zero Standard |
| Aspirational without plan ("aim to", "journey", "strive") | −8 | EU 2024/825 Art. 2 (future performance needs a plan) |
| Quantified (number + unit or %) | +10 | CMA principle 1 |
| Dated (target or reporting year) | +8 | SBTi; IFRS S2 para 33 |
| Baseline stated | +8 | GHG Protocol ch. 5; IFRS S2 para 33 |
| Scope or boundary defined | +8 | GHG Protocol; IFRS S2 para 29 |
| Third-party verification (assurance, ISO 14064, SBTi, certified) | +12 | CMA principle 6; EU Green Claims proposal Art. 10 |
| Concrete action (installed, replaced, recycled) | +6 | CMA principle 1 |

Weights are **normative**: set from the regulatory guidance, not fitted to data, so the score is explainable and
can't overfit. Each signal counts once per sentence. Grades: A ≥ 75, B ≥ 60, C ≥ 45, D ≥ 30, E < 30.

## 5. Results
**Claim detector** (public test split, 265 sentences, 67 claims): macro-F1 **0.864**, accuracy 0.894, F1 (claim
class) 0.800, ROC-AUC 0.941. The paper reports macro-F1 of TF-IDF SVM 0.725, DistilBERT 0.801, ClimateBERT 0.819 and
RoBERTa-large 0.868 on its 300-sentence test split. The public release has 265, so the comparison is indicative.

**Greenwashing language** (671 claims, 5-fold stratified CV): rule score alone ROC-AUC 0.64; signals + words
logistic regression ROC-AUC **0.78**, average precision 0.69. Signal over-representation in greenwashing claims:
exaggerated impact 7.6×, superlatives 3.0×, vague terms 2.6×, overgeneralisation 2.6×.

**Benchmark**: the 665 expert-labelled environmental claims from international listed firms have median
specificity 50 and mean 49.3. This is used as the reference distribution in the app.

## 6. Limitations
- Measures language, not truth. No external fact-checking.
- The EmeraldMind claims are short, news-style and partly synthetic, about US/EU brands. The language-risk score is a
  secondary signal only.
- PDF extraction loses tables and charts. Scanned PDFs (images) produce no text. Multi-column layouts can merge sentences.
- English only. Nigerian English and sector jargon (e.g. "flare-out", "gas commercialisation") aren't in the training data.
- Report types differ (standalone report vs annual-report section) and years differ (2023–2025), so cross-company
  comparisons are indicative.
- Target extraction is rule-based. Targets spread over several sentences or held in tables are missed.
