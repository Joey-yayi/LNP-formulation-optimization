# Prospective R5 Multi-Model LNP Candidate Selection

## Overview

This module performs **prospective candidate selection for R5 validation** in the DC2.4 mRNA-LNP formulation project.

The goal is not to perform another retrospective model comparison. Instead, the script freezes the target-specific R1-R4 learning stage, predicts a common pool of previously untested candidate formulations with multiple model families, and selects a small R5 panel designed to test two questions:

1. **Can the models enrich for genuinely high-transfection LNP formulations?**
2. **Can the models prospectively recover the relative ranking of unseen high, intermediate, and low formulations?**

The default R5 panel contains **12 new formulations**:

- **H1-H5:** five high-confidence multi-model consensus candidates
- **H6-H7:** two additional high-predicted but compositionally more diverse candidates
- **M1-M2:** two genuinely intermediate candidates
- **L1-L2:** two low-predicted controls
- **D1:** one controlled model-disagreement candidate

Thus, the design intentionally places most experimental resources in the high-response region while retaining enough response range to evaluate prospective ranking.

---

## Scientific rationale

The R5 selection strategy is based on a **target-specific, data-efficient, model-informed sequential experimental design**.

Rather than attempting to uniformly cover the entire high-dimensional LNP formulation space, the workflow uses the accumulated R1-R4 DC2.4 data to identify candidate regions that are:

- predicted to have high transfection performance,
- consistently ranked highly by different model families,
- sufficiently supported by the existing applicability domain,
- compositionally non-redundant where possible, and
- informative for validating model ranking.

This allows R5 to serve as a prospective validation round rather than simply another round of retrospective model fitting.

---

## Models used

The same prospective candidate pool is evaluated by multiple model families.

### Tree-based models

- Random Forest
- XGBoost
- Gradient Boosting
- HistGradientBoosting
- LightGBM

### Tabular / foundation models

- TabPFN
- TabFM

The final candidate ranking does **not** directly average raw prediction values alone. Because different model families may have different calibration and prediction ranges, each model first ranks the same candidate pool. The script then combines the **percentile ranks** across models.

For candidate \(i\), the multi-model consensus score is:

\[
\mathrm{ConsensusRank}_i
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathrm{PercentileRank}_{im}
\]

where \(M\) is the number of successfully available models.

This makes the consensus primarily reflect whether different model families agree that a formulation belongs near the top, middle, or bottom of the prospective candidate space.

---

## R5 v1.1 selection logic

Version 1.1 was introduced to make the prospective validation panel more balanced and scientifically interpretable.

### High-confidence candidates

High candidates are selected using a combination of:

- high multi-model consensus rank,
- agreement among models,
- applicability-domain support,
- and formulation diversity.

The highest-confidence candidates are required to be near the top of the common prediction pool and to be classified as top-performing by a majority of available models.

### Intermediate candidates

A percentile rank alone can be misleading if the candidate pool itself is enriched around high-performing regions.

Therefore, v1.1 defines intermediate candidates using **both**:

- percentile-based consensus ranking, and
- the absolute mean prediction across available models.

The default intermediate target is approximately:

```text
all-model mean prediction ≈ 0.65
```

with a default search range of:

```text
0.58–0.72
```

This is intended to create a clearer separation between high-, intermediate-, and low-predicted candidates.

### Low controls

Low controls are selected to be clearly lower than the high and intermediate candidates while remaining experimentally meaningful and within the model applicability domain.

The default low target is approximately:

```text
all-model mean prediction ≈ 0.42
```

with a default search range of:

```text
0.32–0.50
```

### Duplicate ionizable-lipid identities

By default, v1.1 excludes nominal dual-ionizable formulations in which:

```text
IL1 == IL2
```

For example, a formulation represented as `SM102 + SM102` is chemically redundant because it is effectively a single-ionizable-lipid formulation at the combined molar fraction. This restriction improves interpretability of the prospective panel.

### Model-disagreement candidate

One candidate is intentionally selected from a region where otherwise strong models disagree.

This point is not intended to be the highest-performing formulation. Instead, it acts as an **information-rich probe** of a part of the response landscape that is less consistently resolved across model families.

---

## Applicability domain

The script estimates a simple applicability-domain metric using the standardized selected feature space.

For each prospective candidate, the mean distance to its five nearest R1-R4 training points is calculated and compared with the distribution of within-training nearest-neighbor distances.

Important output fields include:

- `AD_mean5NN_distance`
- `AD_ratio_to_train95`
- `AD_in95`
- `AD_in125pct95`

High-confidence prospective candidates are preferentially selected from regions that remain reasonably close to the experimentally observed R1-R4 feature space.

The applicability-domain metric should be interpreted as a practical safeguard against extreme extrapolation, not as a formal proof of prediction reliability.

---

## Candidate-pool generation

When `--candidate-source virtual` is used, the script generates a large feasible prospective library using two complementary strategies.

### Local refinement

Approximately 65% of candidates are generated around experimentally successful R1-R4 formulations.

This increases sampling density around candidate high-response regions.

### Broad feasible exploration

The remaining candidates are generated more broadly within the experimentally observed formulation ranges while using lipid identities present in the existing formulation palette.

Candidate formulations must satisfy composition and feasibility constraints, including:

- total ionizable-lipid fraction,
- helper-lipid fraction,
- cholesterol fraction,
- PEG-lipid fraction,
- and approximate closure of the molar composition to 100%.

Exact duplicates of R1-R4 formulations are removed before prospective prediction.

---

## Feature representation

The R5 deployment workflow uses the same feature-engineering functions as the final R1-R4 publication pipeline.

The feature representation includes:

- formulation-composition variables,
- ionizable-lipid structural descriptors,
- and Morgan fingerprint features.

Auxiliary structural features are selected using the same mRMR-style feature-selection logic used by the imported R1-R4 model pipeline.

The exact target is locked to:

```text
Normalized for DC2.4
```

No alternative cell-line target is used as a fallback.

---

## Recommended repository structure

```text
src/
└── prospective_validation/
    ├── README.md
    └── DC24_R5_Prospective_MultiModel_Selector_v1_1.py

results/
└── prospective_validation/
    ├── R5_selected_12_FROZEN_v1_1.csv
    └── README.md
```

The full automatically generated workbooks and run manifests may contain local machine paths. Before publishing those files, remove or replace machine-specific paths.

---

## Requirements

Recommended Python version:

```text
Python >= 3.10
```

Core dependencies:

```text
numpy
pandas
scikit-learn
scipy
openpyxl
```

Optional model dependencies:

```text
xgboost
lightgbm
tabpfn / tabpfn-client
tabfm
rdkit
```

The script also imports feature-engineering and model-construction functions from the final R1-R4 tree-model pipeline. Therefore, the publication version of that pipeline must be available locally or supplied explicitly through `--tree-script`.

Exact package versions should ideally be recorded in the repository environment file or requirements lock file used for the final analysis.

---

## Required inputs

### 1. R1-R4 training workbook

The workbook must contain the final QC-passed R1-R4 formulation dataset and the exact target column:

```text
Normalized for DC2.4
```

### 2. Final R1-R4 tree-model pipeline

The script imports the final feature-engineering, grouping, mRMR-selection, and model-specification functions from the publication tree-model code.

Use the explicit command-line argument when necessary:

```bash
--tree-script "<path-to-final-R1-R4-tree-model-script>"
```

### 3. Optional external candidate workbook

Instead of virtual candidate generation, a manually designed candidate library can be supplied with:

```bash
--candidate-source excel
--candidate-excel "<candidate-workbook.xlsx>"
```

---

## Usage

### A. Full prospective run

This mode:

1. loads the frozen R1-R4 dataset,
2. rebuilds the final feature representation,
3. fits the requested deployment models,
4. generates or loads candidate formulations,
5. predicts the common candidate pool,
6. calculates consensus ranks and applicability-domain metrics,
7. selects the final R5 panel,
8. and freezes the prospective prediction outputs.

Example:

```bash
python DC24_R5_Prospective_MultiModel_Selector_v1_1.py \
  --train-excel "<path-to-R1-R4-workbook.xlsx>" \
  --tree-script "<path-to-final-tree-model-script.py>"
```

---

### B. Re-select the validation panel from an existing frozen prediction workbook

This is the recommended mode when model predictions have already been generated and only the R5 validation-panel composition needs to be improved **before observing any R5 wet-lab outcome**.

```bash
python DC24_R5_Prospective_MultiModel_Selector_v1_1.py \
  --reselect-workbook "<path-to-R5_prospective_selection_results.xlsx>"
```

By default, this mode preserves the previously selected:

- high-predicted candidates, and
- model-disagreement candidate,

while re-selecting the intermediate and low controls using the v1.1 criteria.

**No model is retrained in this mode.**

This distinction is important for prospective-validation provenance.

---

### C. Full re-selection from an existing scored candidate pool

To apply the v1.1 selection rules to all 12 candidates while still reusing the already generated model predictions:

```bash
python DC24_R5_Prospective_MultiModel_Selector_v1_1.py \
  --reselect-workbook "<path-to-R5_prospective_selection_results.xlsx>" \
  --reselect-all
```

---

### D. Adjust intermediate or low prediction targets

Example:

```bash
python DC24_R5_Prospective_MultiModel_Selector_v1_1.py \
  --reselect-workbook "<path-to-R5_prospective_selection_results.xlsx>" \
  --medium-raw-target 0.65 \
  --medium-raw-low 0.58 \
  --medium-raw-high 0.72 \
  --low-raw-target 0.42 \
  --low-raw-low 0.32 \
  --low-raw-high 0.50
```

These thresholds should be finalized **before** examining the R5 experimental outcomes.

---

## Main outputs

### `R5_selected_12_FROZEN_v1_1.csv`

Unblinded prospective prediction file containing:

- formulation composition,
- selection role,
- per-model predictions,
- per-model percentile ranks,
- consensus rank,
- model disagreement,
- applicability-domain metrics,
- and candidate provenance.

This file is the key prospective prediction record.

### `R5_prospective_selection_results_v1_1.xlsx`

Detailed workbook containing:

- final prediction key,
- blinded wet-lab sheet,
- full common model-scored candidate pool,
- and available provenance information.

### `R5_BLINDED_WETLAB_SHEET_v1_1.xlsx`

A blinded experimental sheet containing formulation identities and compositions without the prediction labels used for candidate selection.

This is intended for wet-lab execution when experimental blinding is desired.

### `R5_FREEZE_HASHES_v1_1.json`

SHA-256 hashes used to document that the frozen prediction outputs were generated before the prospective experiment.

**Note:** automatically generated JSON/workbook metadata can include local filesystem paths. Redact or convert those paths to repository-relative paths before public release.

---

## Prospective experimental design

The default panel is:

| Role | n | Purpose |
|---|---:|---|
| High consensus | 5 | Test whether multi-model agreement enriches for high-performing LNPs |
| High diverse | 2 | Probe whether the high-response region extends across nearby but more diverse compositions |
| Intermediate | 2 | Provide a middle response range for prospective ranking |
| Low control | 2 | Test whether the models correctly identify lower-response formulations |
| Model disagreement | 1 | Probe a less-resolved region where strong models disagree |
| **Total** | **12** | Prospective ranking + high-hit validation |

The main objective is not necessarily to predict the exact numerical value of every R5 formulation. The more important prospective questions are whether the frozen model can:

- enrich the high-performing region,
- distinguish high from intermediate and low candidates,
- and recover the relative ranking of unseen formulations.

---

## Recommended post-experiment evaluation

After R5 wet-lab measurements are obtained, evaluate the frozen predictions without changing the candidate-selection key.

Recommended primary metrics:

- Spearman rank correlation
- Top-3 overlap
- Top-5 overlap
- high-predicted hit rate
- high / intermediate / low group separation

Secondary quantitative metrics may include:

- RMSE
- MAE
- R²

R², RMSE, and MAE should only be interpreted directly when the R5 experimental target is normalized on the same scale as the frozen R1-R4 target.

The script contains an optional evaluation mode:

```bash
python DC24_R5_Prospective_MultiModel_Selector_v1_1.py \
  --evaluate-only \
  --frozen-dir "<path-to-frozen-R5-directory>" \
  --actual-excel "<path-to-measured-R5-workbook.xlsx>" \
  --actual-column "Normalized for DC2.4"
```

---

## Prospective-validation guardrails

For publication-quality prospective validation:

1. **Freeze the R5 prediction key before measuring R5 outcomes.**
2. Do not regenerate candidate rankings after observing R5 transfection results.
3. Keep R5 formulation preparation and biological assay conditions as close as possible to the R1-R4 conditions.
4. Include appropriate same-plate historical or reference controls when possible.
5. Preserve the frozen prediction file and its hash.
6. Clearly distinguish retrospective R1-R4 model development from prospective R5 validation in the manuscript.
7. If the validation panel was revised after inspecting model predictions but before wet-lab outcomes, document that fact transparently.

---

## TabPFN privacy note

When a hosted TabPFN client backend is used, numerical training and candidate feature matrices may be transmitted to the external inference service.

No API token is hard-coded in this script.

Credentials are read only from environment variables such as:

```text
PRIORLABS_API_KEY
TABPFN_TOKEN
```

Use a local backend or `--skip-tabpfn` when remote transfer is inappropriate.

Never commit API keys, tokens, or private credentials to the repository.

---

## Reproducibility

The workflow uses a fixed random seed by default:

```text
20260817
```

The prospective output should be frozen before experimentation, and SHA-256 hashes should be retained as provenance.

For the final publication repository, also provide:

- the final R1-R4 preprocessing/modeling script,
- an environment file or exact package versions,
- the processed modeling dataset or a clearly documented data-access statement,
- and the final frozen prospective R5 prediction table.

---

## Interpretation

The R5 experiment is designed as a test of **prospective prediction and ranking**, not as another round of retrospective model optimization.

A successful validation would be supported by findings such as:

```text
High > Intermediate > Low
```

together with positive prospective rank correlation and enrichment of experimentally high-performing formulations among the candidates assigned high frozen prediction ranks.

If a candidate predicted to be high exceeds the historical performance range experimentally, this should be interpreted as successful prioritization of a high-value region rather than evidence that tree models accurately extrapolated the exact response magnitude beyond the observed training range.

---

## Citation

If this code is used in a publication, cite the associated manuscript and repository release.

A versioned GitHub release is recommended once the R5 prediction panel and the corresponding prospective experimental outcomes are finalized.
