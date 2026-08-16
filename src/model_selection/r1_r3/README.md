# DC2.4 mRNA-LNP Transfection Model Selection and Validation

This directory contains a publication-oriented machine-learning pipeline for modeling normalized DC2.4 mRNA-LNP transfection efficiency from lipid formulation variables and ionizable-lipid structural features.

## Main script

```text
DC24_LNP_Transfection_ModelSelection_NestedCV.py
```

The script is designed for small-sample LNP datasets and emphasizes leakage-resistant model evaluation, reproducibility, formulation-level grouping, cumulative-round analysis, and prospective external validation.

## Scientific objective

The workflow evaluates whether LNP composition can be used to:

1. predict normalized DC2.4 transfection efficiency;
2. rank candidate formulations by expected performance;
3. compare multiple regression models under the same validation framework;
4. quantify how model performance changes as data accumulate across experimental rounds;
5. assess generalization across rounds and in prospective experimental validation.

This script performs **model selection and validation**. It does not generate new best-formulation candidates. Candidate generation should be maintained as a separate workflow.

## Recommended repository structure

```text
LNP-formulation-optimization/
├── data/
│   └── processed/
│       └── lnp_dc24_hacat_modeling_dataset.xlsx
├── src/
│   └── model_selection/
│       ├── DC24_LNP_Transfection_ModelSelection_NestedCV.py
│       └── README.md
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

The script also accepts closely related workbook spellings, including:

```text
Inp_dc24_hacat_modeling_dataset.xlsx
lnp_dc24_hacat_modeling_dataset.xlsx
Lnp_dc24_hacat_modeling_dataset.xlsx
LNP_dc24_hacat_modeling_dataset.xlsx
```

## Primary modeling target

The default response variable is the plate-normalized DC2.4 transfection-efficiency column:

```text
normolized for DC2.4
```

Common spelling variants are detected automatically.

HaCaT measurements may remain in the workbook for reference, but they are **not used as the response variable or as predictor features** in the DC2.4 model.

A second target mode is available for direct modeling of log-transformed raw luminescence:

```text
log10(DC_Cell_Transfection_Efficiency)
```

The target definition must remain identical between training and external validation.

## Quality-control criteria

The default primary-model QC rules are:

- PDI ≤ 0.50;
- particle size between 30 and 300 nm;
- samples with missing required size or PDI measurements are excluded;
- template rows, repeated header rows, unknown lipid identities, and invalid molar totals are removed;
- post-synthesis particle size, PDI, and zeta potential are not used as predictors in the pre-experimental model.

QC thresholds can be changed from the command line.

## Input formulation variables

Core formulation variables are retained in every outer training fold and are not removed by mRMR:

- ionizable lipid 1 identity and molar percentage;
- ionizable lipid 2 identity and molar percentage;
- total ionizable-lipid percentage;
- ionizable-lipid mixing ratio;
- phospholipid identity and molar percentage;
- cholesterol molar percentage;
- PEG-lipid identity and molar percentage;
- predefined formulation interaction terms.

Optional auxiliary variables include:

- weighted ionizable-lipid structural descriptors;
- Morgan fingerprint bits derived from SMILES;
- precomputed fingerprint columns stored in the workbook.

mRMR is applied only to auxiliary structural and fingerprint variables, and selection is performed independently inside each outer training fold.

## Models evaluated

The primary workflow evaluates:

- Random Forest;
- Extra Trees;
- Gradient Boosting;
- Ridge regression;
- HistGradientBoosting, when available;
- XGBoost, when installed;
- LightGBM, when installed.

The prespecified primary ensemble averages predictions from:

```text
Random Forest + Extra Trees + Gradient Boosting
```

The ensemble membership is defined before prospective validation to reduce post-hoc model-selection bias.

## Validation design

### Formulation-grouped repeated nested cross-validation

The default validation configuration is:

```text
Outer validation: 5 folds × 3 repeats
Inner tuning:      4 folds
Parameter trials:  12 randomized combinations per model and outer fold
Split mode:         formulation-grouped
Auxiliary features: top 8 selected within each outer training fold
```

Exact duplicate formulations are assigned to the same training or test fold. They cannot appear on opposite sides of a split.

Within each outer fold:

1. auxiliary feature selection is performed using only the outer training data;
2. hyperparameter tuning is performed using only the outer training data;
3. the fitted model predicts the untouched outer test fold;
4. all out-of-fold predictions are aggregated for final performance estimation.

### Cumulative-round analysis

The script independently evaluates cumulative datasets:

```text
R1
R1 + R2
R1 + R2 + R3
...
```

This analysis shows how predictive performance changes as experimental data accumulate.

### Leave-one-round-out validation

Each experimental round can be held out as a complete test set. This analysis measures sensitivity to round-to-round distribution shifts and experimental batch effects.

### Prospective external validation

A separate workbook, such as S15 or a later experimental round, can be supplied after the model and analysis workflow have been locked.

External R², RMSE, MAE, Spearman correlation, Pearson correlation, residuals, and prediction tables are exported automatically.

External RMSE is scientifically meaningful only when the experimental values use exactly the same target scale as the training data.

## Evaluation metrics

### R²

R² measures agreement between experimental and predicted numerical values:

```text
R² = 1 - Σ(y - ŷ)² / Σ(y - mean(y))²
```

Higher values indicate better numerical prediction. Negative R² means that predictions are worse than using the test-set mean as a constant predictor.

### RMSE

RMSE is the square root of the mean squared prediction error:

```text
RMSE = sqrt(mean((y - ŷ)²))
```

It is expressed in the same units as the modeling target and gives greater weight to large prediction errors.

### MAE

MAE is the mean absolute prediction error:

```text
MAE = mean(abs(y - ŷ))
```

### Spearman correlation

Spearman correlation evaluates whether predicted and experimental rankings are consistent. It is especially relevant when the objective is to prioritize high-performing LNP candidates rather than reproduce every target value exactly.

### Top-20% recall

Top-20% recall reports the fraction of experimentally top-performing samples that are also included in the predicted top 20%.

## Installation

A Python environment with Python 3.10 or newer is recommended.

Required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl joblib
```

Optional packages:

```bash
pip install rdkit xgboost lightgbm
```

If RDKit is unavailable, the script attempts to load precomputed Morgan fingerprint columns from the Excel workbook.

## Running the analysis

From the repository root:

```bash
python src/model_selection/DC24_LNP_Transfection_ModelSelection_NestedCV.py \
  --data-path "data/processed/lnp_dc24_hacat_modeling_dataset.xlsx"
```

On Windows PowerShell:

```powershell
python src/model_selection/DC24_LNP_Transfection_ModelSelection_NestedCV.py `
  --data-path "data/processed/lnp_dc24_hacat_modeling_dataset.xlsx"
```

A faster test run can be performed with:

```bash
python src/model_selection/DC24_LNP_Transfection_ModelSelection_NestedCV.py \
  --data-path "data/processed/lnp_dc24_hacat_modeling_dataset.xlsx" \
  --outer-repeats 1 \
  --tune-iter 5
```

## Prospective external validation

When the external workbook already contains the same normalized target scale:

```bash
python src/model_selection/DC24_LNP_Transfection_ModelSelection_NestedCV.py \
  --data-path "data/processed/lnp_dc24_hacat_modeling_dataset.xlsx" \
  --external-path "data/processed/S15_validation.xlsx" \
  --external-actual-column "actual_normolized_for_DC2.4"
```

For a model trained directly on log10 raw luminescence:

```bash
python src/model_selection/DC24_LNP_Transfection_ModelSelection_NestedCV.py \
  --data-path "data/processed/lnp_dc24_hacat_modeling_dataset.xlsx" \
  --target-mode log10_raw \
  --raw-target-column "DC_Cell_Transfection_Efficiency" \
  --external-path "data/processed/S15_validation.xlsx" \
  --external-actual-column "DC_Cell_Transfection_Efficiency" \
  --external-actual-scale log10_raw
```

## Main outputs

The script creates a timestamped output directory containing:

```text
publication_safe_model_results.xlsx
model_summary.csv
nested_cv_primary_ensemble_oof.csv
nested_cv_primary_ensemble_parity.png
nested_cv_primary_ensemble_residuals.png
cumulative_round_model_summary.csv
cumulative_round_primary_ensemble_summary.csv
cumulative_round_learning_curve.png
leave_one_round_out_parity.png
final_model_bundle.joblib
run_manifest.json
```

The Excel workbook contains:

- overall model summaries;
- sample-level out-of-fold predictions;
- fold-level metrics;
- fold-specific selected auxiliary features;
- cumulative-round analyses;
- leave-one-round-out predictions and metrics;
- external-validation results;
- included and excluded samples;
- final selected auxiliary features;
- run configuration and provenance.

## Representative results from the current dataset

Using the current processed DC2.4 dataset and the default formulation-grouped repeated nested-CV configuration, the primary tree ensemble produced approximately:

```text
Aggregated OOF R²:       0.713
RMSE:                    0.136
MAE:                     0.111
Spearman correlation:    0.827
```

Cumulative-round performance was approximately:

| Training data | R² | RMSE | Spearman |
|---|---:|---:|---:|
| R1 | 0.461 | 0.097 | 0.644 |
| R1 + R2 | 0.623 | 0.145 | 0.772 |
| R1 + R2 + R3 | 0.713 | 0.136 | 0.827 |

These values are dataset- and software-version-dependent and should be regenerated from the archived code, environment, and workbook used for the final manuscript.

## Interpretation and limitations

The reported grouped nested-CV performance primarily reflects interpolation within the explored lipid and formulation space.

The model should not be interpreted as having the same accuracy for:

- completely new ionizable-lipid chemistries;
- substantially different formulation ranges;
- new cell lines;
- uncontrolled experimental batches;
- external experiments that use a different normalization procedure.

Across-round validation may be substantially weaker than grouped nested CV because experimental rounds can differ in formulation distribution, cell state, LNP preparation, and assay calibration.

The model is therefore best interpreted as a tool for formulation prioritization and ranking within the explored design space, together with prospective experimental validation.

## TabPFN status

TabPFN is not included in the current primary pipeline.

It may be evaluated later as an **optional standalone benchmark** under the same formulation-grouped outer folds. It should not be added to the primary ensemble after inspecting the current results unless the complete analysis plan is redefined and rerun.

For a fair comparison, a future TabPFN analysis should:

1. use the same outer formulation groups;
2. use feature selection based only on each outer training fold;
3. use a fixed and documented TabPFN checkpoint and package version;
4. report the same R², RMSE, MAE, Spearman, and Top-20% recall metrics;
5. remain separate from the prespecified primary tree ensemble unless inclusion is declared before prospective validation.

## Reproducibility safeguards

The workflow includes the following safeguards:

- no feature selection on outer test labels;
- no hyperparameter tuning on outer test samples;
- no tiny random holdout described as an independent test set;
- exact duplicate formulations remain in the same fold;
- core formulation variables are always retained;
- optional structural features are selected within each fold;
- post-synthesis QC variables are excluded from pre-experimental prediction;
- random seeds, model parameters, selected features, metrics, and output paths are recorded;
- the fitted final ensemble is archived as a joblib bundle.

## License

This repository is distributed under the license specified in the repository-level `LICENSE` file.

## Citation

When the associated manuscript or preprint becomes available, add its citation here. Until then, cite the repository version or archived release used for analysis.
