# DC2.4 R1–R4 LNP Model Selection and Cumulative Learning

## Overview

This directory contains the final R1–R4 modeling pipeline used to evaluate mRNA-LNP transfection efficiency in DC2.4 dendritic cells.

The analysis was designed for a small, target-specific dataset and emphasizes leakage-safe model evaluation, formulation-grouped cross-validation, molecular-structure features, and cumulative performance across experimental rounds.

The primary target is:

**Normalized for DC2.4**

The current frozen R1–R4 dataset contains **104 QC-passed experimental formulations** distributed as:

- R1: 30
- R2: 26
- R3: 28
- R4: 20

These 104 observations correspond to **97 unique formulation groups** after exact duplicate formulations were grouped.

---

## Main Analysis Workflow

```text
R1–R4 experimental formulations
        ↓
Data cleaning and QC
        ↓
Corrected round-label recovery
        ↓
Exact-formulation grouping
        ↓
Core formulation features
+ ionizable-lipid structural descriptors
+ 128-bit Morgan fingerprints
        ↓
Fold-safe mutual-information / mRMR feature selection
        ↓
Grouped nested cross-validation
        ↓
Model comparison
        ↓
Primary tree ensemble
        ↓
Cumulative R1 → R4 performance analysis
        ↓
Final model fit
```

---

## Dataset Processing

The current run used the worksheet:

`Round1&2&3&4`

and the exact target column:

`Normalized for DC2.4`

The processing pipeline produced:

| Processing stage | Number of rows |
|---|---:|
| Rows with non-missing target | 182 |
| After invalid/template removal | 140 |
| After QC filtering | 104 |

Round labels were corrected for historical replicate identifiers containing more than one round label. The final retained round counts were:

| Round | n |
|---|---:|
| R1 | 30 |
| R2 | 26 |
| R3 | 28 |
| R4 | 20 |
| **Total** | **104** |

Exact duplicate formulations were kept in the same cross-validation group to avoid leakage.

---

## Feature Representation

The final feature representation combines formulation composition with ionizable-lipid structural information.

### Core formulation features

The core feature block contains **35 variables**, including:

- IL1 and IL2 identities;
- IL1 and IL2 molar percentages;
- total ionizable-lipid fraction;
- IL1 fraction within the ionizable-lipid mixture;
- helper-lipid identity and fraction;
- cholesterol fraction;
- PEG-lipid identity and fraction;
- selected formulation interaction terms;
- fixed one-hot encoded categorical variables.

### Structural features

Ionizable-lipid structural information was represented using:

- manually defined structural descriptors;
- **128-bit Morgan molecular fingerprints** generated from SMILES using RDKit.

For dual-ionizable-lipid formulations, structural descriptors and Morgan fingerprints were weighted according to the IL1 and IL2 molar fractions.

The complete feature blocks in the current run were:

- Core features: **35**
- Auxiliary structural/fingerprint features: **144**
- Total pre-selection features: **179**

To reduce redundancy in the small-sample setting, auxiliary features were selected **inside each outer training fold only** using mutual information and an mRMR-style relevance–redundancy procedure.

The default number of selected auxiliary features was **8**, giving **43 features** in the final fitted ensemble models.

---

## Models Compared

The following regression models were evaluated under the same grouped nested cross-validation framework:

- Random Forest
- Extra Trees
- Gradient Boosting
- Ridge regression
- HistGradientBoosting
- XGBoost
- LightGBM

The pre-specified primary ensemble was:

**Random Forest + Extra Trees + Gradient Boosting**

---

## Cross-Validation Design

The primary evaluation used **grouped nested cross-validation**.

- Outer CV: 5 folds × 3 repeats
- Total outer evaluations per model: 15
- Inner CV: 4 folds
- Hyperparameter search iterations: 12
- Grouping unit: exact LNP formulation
- Auxiliary feature selection: performed inside each outer training fold only

This design prevents exact replicate formulations from being split across training and test folds and reduces information leakage from feature selection or hyperparameter optimization.

---

## Final R1–R4 Model Performance

The final 104-formulation dataset produced the following out-of-fold performance:

| Model | R² | RMSE | MAE | Spearman | Top-20% recall |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0.704** | **0.120** | 0.098 | 0.835 | **0.714** |
| Random Forest | **0.704** | 0.120 | 0.099 | **0.837** | 0.667 |
| Primary Tree Ensemble | **0.699** | 0.121 | 0.100 | **0.825** | 0.619 |
| Gradient Boosting | 0.698 | 0.121 | 0.100 | 0.833 | 0.667 |
| HistGradientBoosting | 0.695 | 0.122 | 0.099 | 0.831 | 0.619 |
| LightGBM | 0.692 | 0.122 | 0.101 | 0.826 | **0.714** |
| Extra Trees | 0.656 | 0.130 | 0.106 | 0.799 | 0.667 |
| Ridge | 0.518 | 0.153 | 0.128 | 0.721 | 0.667 |

The current primary ensemble therefore achieved:

- **R² = 0.699**
- **RMSE = 0.121**
- **MAE = 0.100**
- **Spearman = 0.825**
- **Top-20% recall = 0.619**

The best individual R² in the current run was obtained by XGBoost (**R² = 0.704**), while Random Forest showed the highest Spearman correlation among the individual models (**0.837**).

---

## Cumulative R1–R4 Learning

To evaluate how predictive performance changed as target-specific experimental data accumulated, the same modeling framework was applied to cumulative round subsets.

| Cumulative dataset | n | Formulation groups | R² | RMSE | Spearman | Top-20% recall |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 30 | 30 | 0.455 | 0.091 | 0.644 | 0.667 |
| R1 + R2 | 56 | 56 | 0.442 | 0.149 | 0.645 | 0.417 |
| R1 + R2 + R3 | 84 | 78 | 0.562 | 0.128 | 0.727 | 0.471 |
| R1 + R2 + R3 + R4 | 104 | 97 | **0.699** | **0.121** | **0.825** | **0.619** |

The cumulative analysis shows a clear improvement after R2, with both R² and ranking performance increasing substantially as R3 and R4 data were incorporated.

The small decrease in R² from R1 to R1+R2 should not be interpreted as model deterioration in isolation because the target range and formulation space broadened in R2. The more informative overall trend is the improvement from the early-stage dataset to the final R1–R4 model.

---

## Leave-One-Round-Out Stress Test

A leave-one-round-out analysis was also generated as a distribution-shift stress test.

The held-out-round R² values were:

- R1: -4.139
- R2: 0.327
- R3: -0.152
- R4: -2.272

These results indicate substantial cross-round distribution shift and should **not** be interpreted as the main predictive-performance estimate.

The grouped nested-CV results are used as the primary retrospective model evaluation, while genuinely prospective prediction of previously untested formulations is planned as the final external validation.

---

## Prospective Validation

The next stage of the project is prospective validation using approximately **10–15 previously untested formulations**.

The final preprocessing, feature representation, model configuration, and candidate-ranking strategy will be frozen before wet-lab testing.

Candidate selection will include a mixture of:

- predicted high-performing formulations;
- intermediate predicted formulations;
- predicted low-performing controls;
- a limited number of exploratory or boundary candidates.

The primary objective is to test whether the model can **predict and rank genuinely unseen formulations**, rather than only reproduce patterns within the retrospective R1–R4 dataset.

---

## Running the Script

### Required packages

```text
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
joblib
rdkit
xgboost
lightgbm
```

### Recommended command

For reproducible use outside the original development computer, explicitly provide the workbook path:

```bash
python DC24_R1R4_ModelSelection.py --data-path "path/to/R1-4_all_LNP_normalized_1.35.xlsx"
```

Optional faster test run:

```bash
python DC24_R1R4_ModelSelection.py \
    --data-path "path/to/R1-4_all_LNP_normalized_1.35.xlsx" \
    --outer-repeats 1 \
    --tune-iter 5
```

The current development version also contains a local fallback workbook path for direct execution on the original workstation. For public reuse, `--data-path` is recommended.

---

## Main Output Files

The script generates several machine-readable outputs.

Recommended files to retain in the public repository are:

### `publication_safe_model_results.xlsx`

The complete analysis workbook. It contains:

- model summary;
- nested-CV out-of-fold predictions;
- fold-level metrics;
- fold-selected auxiliary features;
- auxiliary-feature selection frequency;
- cumulative-round results;
- leave-one-round-out results;
- training data used after QC;
- removed rows and QC records;
- final selected auxiliary features;
- run information.

### `model_summary.csv`

Compact comparison of all evaluated models.

### `cumulative_round_primary_ensemble_summary.csv`

Primary ensemble performance across cumulative R1–R4 datasets.

### `nested_cv_primary_ensemble_oof.csv`

Out-of-fold predictions for the primary tree ensemble.

These CSV files are useful because GitHub can display them directly in the browser.

---

## Repository Organization

Recommended directory structure:

```text
src/
└── model_selection/
    └── r1_r4/
        ├── README.md
        └── DC24_R1R4_ModelSelection.py

results/
└── r1_r4/
    ├── publication_safe_model_results.xlsx
    ├── model_summary.csv
    ├── cumulative_round_primary_ensemble_summary.csv
    └── nested_cv_primary_ensemble_oof.csv
```

Automatically generated diagnostic PNG files do not need to be retained in the main repository at this stage. Publication-ready figures can be added later in a dedicated `figures/` directory.

---

## Reproducibility Notes

- Random seed: 42
- Exact duplicate formulations are grouped during cross-validation.
- Feature selection is performed inside outer training folds.
- Hyperparameter tuning is performed inside outer training folds.
- Post-synthesis size/PDI/zeta measurements are not used as primary pre-experimental predictive features.
- Structural descriptors and Morgan fingerprints are used only as formulation/structure inputs.
- The current analysis uses the 1.35-normalized DC2.4 target.
- The final retrospective dataset contains 104 QC-passed observations and 97 exact formulation groups.

---

## Status

**Current status:** final retrospective R1–R4 modeling pipeline completed.

**Next planned step:** frozen-model prospective validation on previously untested LNP formulations.
