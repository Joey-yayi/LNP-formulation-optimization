# TabPFN / TabFM Benchmark for DC2.4 LNP Transfection Prediction

## Overview

This directory contains the R1–R4 small-sample benchmark comparing **TabPFN** and **TabFM** for prediction of mRNA-LNP transfection efficiency in DC2.4 dendritic cells.

The benchmark uses the same final R1–R4 experimental dataset and the same formulation-grouped evaluation principle used in the main tree-model analysis.

**Primary target:** `Normalized for DC2.4`

**Final dataset:** 104 QC-passed experimental formulations representing 97 exact formulation groups.

The goal of this analysis is to test whether pretrained tabular foundation models recover the same predictive signal observed with conventional tree-based models in the small, target-specific LNP dataset.

---

## Dataset

The benchmark uses the final R1–R4 workbook:

`R1-4 all LNP normalized 1.35 (new).xlsx`

Only the exact column:

`Normalized for DC2.4`

is used as the DC2.4 transfection target.

The preprocessing pipeline retained:

- 104 QC-passed experimental rows
- 97 exact formulation groups
- 7 duplicate formulation groups
- 14 rows belonging to duplicate groups

Exact duplicate formulations are assigned to the same cross-validation group to prevent replicate leakage.

---

## Feature Representation

The benchmark uses the same formulation and molecular-structure representation as the main R1–R4 modeling pipeline.

### Core formulation features

**35 core variables**, including formulation composition, ionizable-lipid identities and ratios, helper lipid, cholesterol, PEG-lipid variables, and selected interaction terms.

### Auxiliary molecular features

**144 auxiliary features**, including:

- ionizable-lipid structural descriptors
- 128-bit Morgan molecular fingerprints generated from SMILES using RDKit

For dual-ionizable-lipid formulations, molecular descriptors and Morgan fingerprints are weighted according to the IL1/IL2 molar fractions.

### Fold-safe auxiliary feature selection

Within each outer training fold, auxiliary features are screened using mutual information and an mRMR-style relevance–redundancy procedure.

The default number of selected auxiliary features is:

`aux_top_k = 8`

Core features are always retained.

---

## Cross-Validation Design

The benchmark uses repeated exact-formulation-grouped cross-validation:

- Outer folds: 5
- Outer repeats: 3
- Total outer evaluations per model: 15
- Grouping unit: exact LNP formulation
- Random seed: 42

The same outer splits are used for TabPFN and TabFM.

The benchmark intentionally does not perform task-specific inner-loop hyperparameter tuning for the foundation models.

---

## Models

### TabPFN

TabPFN is evaluated through the TabPFN client backend.

In the current implementation, training features and target values are sent to the hosted Prior Labs service during fitting/prediction.

### TabFM

TabFM v1.0.0 is evaluated using the PyTorch backend.

Pretrained weights are loaded from Hugging Face on first use. Users should review the applicable TabFM license before reuse or redistribution.

---

## Final R1–R4 Results

| Model | R² | RMSE | MAE | Spearman | Pearson | Top-20% recall | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TabPFN** | **0.710** | **0.119** | **0.101** | **0.833** | **0.844** | 0.619 | 0.676 |
| TabFM | 0.681 | 0.125 | 0.104 | 0.822 | 0.826 | 0.619 | 0.692 |

TabPFN showed numerically higher R² and slightly lower prediction error than TabFM.

However, paired exact-formulation cluster bootstrap comparisons did **not** show a statistically distinguishable difference between the two models.

For R²:

- TabFM − TabPFN = −0.028
- 95% CI = −0.075 to 0.015

For Spearman:

- TabFM − TabPFN = −0.011
- 95% CI = −0.044 to 0.020

Because the confidence intervals include zero, the current benchmark should not be interpreted as evidence that TabPFN significantly outperforms TabFM.

---

## Relation to the Tree-Model Benchmark

The same R1–R4 dataset was also evaluated with conventional regression models.

Representative results from the main tree-model analysis were:

| Model | R² | RMSE | Spearman |
|---|---:|---:|---:|
| XGBoost | 0.704 | 0.120 | 0.835 |
| Random Forest | 0.704 | 0.120 | 0.837 |
| Primary tree ensemble | 0.699 | 0.121 | 0.825 |
| TabPFN | 0.710 | 0.119 | 0.833 |
| TabFM | 0.681 | 0.125 | 0.822 |

The similar performance across conventional tree-based models, TabPFN, and TabFM indicates that the predictive signal is not dependent on a single model architecture.

The current TabPFN/TabFM script can merge tree-model summary statistics for descriptive comparison. However, tree out-of-fold predictions were not successfully matched in the current benchmark run, so paired bootstrap testing was performed only between TabPFN and TabFM.

---

## Output Files

Recommended public result files:

### `tabpfn_tabfm_publication_results.xlsx`

Main results workbook containing:

- combined model summary
- TabPFN/TabFM summary
- repeat-level metrics
- fold-level metrics
- OOF predictions
- paired bootstrap comparison
- selected auxiliary features
- auxiliary-feature selection frequency
- split assignments
- foundation-model status
- training data used after QC
- QC audit information
- run metadata

### `tabpfn_tabfm_summary.csv`

Compact TabPFN/TabFM performance summary.

### `tabpfn_tabfm_oof_predictions.csv`

Out-of-fold predictions for both models. This file can be used to reproduce parity plots and recalculate performance metrics.

### Optional: `split_assignments.csv`

Exact repeated grouped-CV split assignments. This is useful for strict reproducibility but is redundant with the corresponding worksheet in the main Excel results file.

---

## Recommended Repository Structure

```text
src/
└── tabpfn_tabfm/
    ├── README.md
    └── DC24_LNP_TabPFN_TabFM_R1R4.py

results/
└── tabpfn_tabfm/
    ├── tabpfn_tabfm_publication_results.xlsx
    ├── tabpfn_tabfm_summary.csv
    └── tabpfn_tabfm_oof_predictions.csv
```

The automatically generated parity and residual plots are useful diagnostic outputs, but they do not need to be stored in the main repository before the publication figures are finalized.

---

## Running the Benchmark

Example:

```bash
python DC24_LNP_TabPFN_TabFM_R1R4.py \
    --data-path "path/to/R1-4 all LNP normalized 1.35 (new).xlsx" \
    --sheet-name "Round1&2&3&4 "
```

The current direct-run development script can also resolve the local R1–R4 workbook and small worksheet-name variations automatically.

---

## Reproducibility Notes

- Target is locked to `Normalized for DC2.4`.
- No HaCaT or alternative normalized response column is substituted.
- Exact formulation duplicates are grouped during CV.
- Auxiliary feature selection is performed inside the outer training folds.
- Morgan fingerprints contain 128 bits.
- TabPFN and TabFM use the same repeated grouped-CV splits.
- Paired model comparisons use exact-formulation cluster bootstrap with 2,000 iterations.
- The current benchmark run used 104 QC-passed rows and 97 formulation groups.
- Round labels are not used as model inputs or CV grouping variables in this benchmark.

---

## Current Status

**Completed:** R1–R4 TabPFN/TabFM retrospective grouped-CV benchmark.

**Next step:** compare the frozen final model(s) on genuinely unseen prospective LNP formulations.
