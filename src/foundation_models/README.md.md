# TabPFN and TabFM Benchmark for DC2.4 LNP Formulation Modeling

This directory contains a publication-oriented, leakage-safe benchmark of **TabPFN** and **TabFM** for predicting normalized DC2.4 mRNA–lipid nanoparticle (LNP) transfection efficiency.

The workflow is intended for small tabular formulation datasets and prevents exact duplicate formulations from being split across training and test partitions.

## Repository location

```text
src/foundation_models/
├── dc24_tabpfn_tabfm_grouped_cv.py
└── README.md
```

Recommended repository layout:

```text
LNP-formulation-optimization/
├── data/
│   └── processed/
│       └── lnp_dc24_hacat_modeling_dataset.xlsx
├── src/
│   ├── model_selection/
│   └── foundation_models/
│       ├── dc24_tabpfn_tabfm_grouped_cv.py
│       └── README.md
├── results/
│   └── foundation_models/
├── requirements.txt
├── LICENSE
└── README.md
```

## Scientific objective

The script evaluates whether pretrained tabular foundation models can predict normalized DC2.4 transfection efficiency from LNP formulation and ionizable-lipid structural features.

The analysis uses:

- LNP component identities and molar percentages;
- engineered formulation-interaction features;
- weighted ionizable-lipid molecular descriptors;
- 128-bit Morgan fingerprints derived from ionizable-lipid SMILES;
- repeated exact-formulation-grouped cross-validation;
- repeated out-of-fold prediction aggregation;
- exact-formulation cluster-bootstrap confidence intervals;
- paired bootstrap comparisons between models.

HaCaT measurements may remain in the workbook for reference, but they are **not used as the target or as predictor variables** in the DC2.4 analysis.

## Leakage-control strategy

The workflow applies the following safeguards:

1. **Exact-formulation grouping**  
   Formulations with identical lipid identities, molar percentages, and available N/P ratio are assigned to the same group.

2. **Grouped outer cross-validation**  
   Repeated measurements from the same formulation remain entirely in either the training partition or the test partition for a given fold.

3. **Fold-local feature selection**  
   Auxiliary molecular descriptors and Morgan fingerprint bits are selected by mRMR using only the current outer-training fold.

4. **Strict response selection**  
   The normalized DC2.4 target is required by default. The script does not silently substitute another response column.

5. **Complete audit trail**  
   Split assignments, OOF predictions, fold metrics, repeat metrics, software versions, hashes, bootstrap results, and run configuration are exported.

## Validation design

The formal benchmark uses:

```text
5 grouped folds × 3 repeated partitions = 15 outer test fits per model
```

Each sample receives one out-of-fold prediction in each repeat. The three valid OOF predictions are averaged to obtain the final repeated OOF prediction used for the primary pooled metrics.

Default settings:

| Parameter | Default |
|---|---:|
| Outer folds | 5 |
| Outer repeats | 3 |
| Auxiliary features selected per fold | 8 |
| Morgan fingerprint size | 128 bits |
| Morgan radius | 2 |
| PDI cutoff | ≤ 0.5 |
| Particle-size range | 30–300 nm |
| Cluster-bootstrap iterations | 2,000 |
| Global random state | 42 |
| Bootstrap seed | 20260803 |

## Reference dataset

For the manuscript-oriented run:

- 106 original rows;
- 2 invalid or template rows removed;
- 20 QC-failing rows removed;
- 84 rows retained;
- 78 exact formulation groups;
- 6 duplicate formulation groups;
- R1, R2, and R3 represented.

The normalized DC2.4 response ranged from approximately 0.05 to 1.00.

## Reference performance

Performance calculated from mean repeated OOF predictions:

| Model | R² | RMSE | MAE | Spearman | Top-20% recall |
|---|---:|---:|---:|---:|---:|
| Primary tree ensemble | 0.713 | 0.136 | 0.111 | 0.827 | 0.588 |
| TabFM | 0.709 | 0.137 | 0.110 | 0.836 | 0.588 |
| TabPFN | 0.693 | 0.141 | 0.114 | 0.809 | 0.647 |

Approximate 95% exact-formulation cluster-bootstrap intervals for R²:

| Model | R² 95% CI |
|---|---|
| Primary tree ensemble | 0.598–0.796 |
| TabFM | 0.587–0.796 |
| TabPFN | 0.566–0.782 |

Paired formulation-cluster bootstrap comparisons did not identify a supported difference between the three models because all pairwise confidence intervals included zero.

The results should therefore be interpreted as **comparable overall predictive performance**, with TabFM showing the numerically highest rank correlation and TabPFN showing the numerically highest Top-20% recall.

## Environment setup

Python 3.12 was used for the reference run.

### Windows

```bash
python -m venv .venv_tabfm
.venv_tabfm\Scripts\activate
python -m pip install --upgrade pip
```

### macOS or Linux

```bash
python3 -m venv .venv_tabfm
source .venv_tabfm/bin/activate
python -m pip install --upgrade pip
```

Install the main scientific dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl joblib rdkit torch tabpfn-client
```

TabFM must also be installed in the same environment according to its upstream installation instructions.

For a formal release, pin exact versions in `requirements.txt` or an environment lock file.

## Data requirements

Default workbook name:

```text
lnp_dc24_hacat_modeling_dataset.xlsx
```

Recommended location:

```text
data/processed/lnp_dc24_hacat_modeling_dataset.xlsx
```

The workbook should contain:

- candidate or formulation identifier;
- ionizable lipid 1 and optional ionizable lipid 2;
- ionizable-lipid molar percentages;
- helper phospholipid identity and molar percentage;
- cholesterol molar percentage;
- PEG-lipid identity and molar percentage;
- normalized DC2.4 transfection-efficiency target;
- particle size and/or PDI for QC;
- optional N/P ratio;
- round label or candidate IDs from which round labels can be inferred;
- an ionizable-lipid SMILES table, preferably in a sheet named `SMILES NAME`.

Recognized normalized-target aliases include:

```text
normolized for DC2.4
normalized for DC2.4
normalized DC2.4
normalized
```

The historical misspelling `normolized` is supported for compatibility.

## Formal benchmark command

Run from the repository root:

```bash
python src/foundation_models/dc24_tabpfn_tabfm_grouped_cv.py \
  --data-path data/processed/lnp_dc24_hacat_modeling_dataset.xlsx \
  --outer-folds 5 \
  --outer-repeats 3 \
  --aux-top-k 8 \
  --bootstrap-iterations 2000
```

Windows PowerShell one-line command:

```powershell
python src/foundation_models/dc24_tabpfn_tabfm_grouped_cv.py --data-path data/processed/lnp_dc24_hacat_modeling_dataset.xlsx --outer-folds 5 --outer-repeats 3 --aux-top-k 8 --bootstrap-iterations 2000
```

## Validation-only run

Validate workbook loading, QC, features, fingerprints, groups, and split assignments without fitting foundation models:

```bash
python src/foundation_models/dc24_tabpfn_tabfm_grouped_cv.py \
  --validate-only \
  --data-path data/processed/lnp_dc24_hacat_modeling_dataset.xlsx
```

## Important command-line options

| Option | Purpose |
|---|---|
| `--no-tabpfn` | Skip TabPFN |
| `--no-tabfm` | Skip TabFM |
| `--tabpfn-backend auto` | Prefer hosted client and fall back to local backend |
| `--tabpfn-backend client` | Require Prior Labs hosted inference |
| `--tabpfn-backend local` | Require local TabPFN inference |
| `--tabfm-backend pytorch` | Use the PyTorch TabFM backend |
| `--require-morgan` | Stop if Morgan fingerprints cannot be generated |
| `--no-auto-repair-rdkit` | Disable the one-time RDKit repair attempt |
| `--allow-missing-qc` | Retain rows with missing size or PDI |
| `--allow-model-failure` | Preserve a partial exploratory run when one requested model fails |
| `--output-dir PATH` | Specify the output directory |
| `--tree-summary-path PATH` | Add a prior tree-model summary |
| `--tree-oof-path PATH` | Add prior tree-model OOF predictions |

`--allow-model-failure` should not be used for the final formal benchmark unless a partial analysis is explicitly intended.

## Output files

### Main tables

| File | Description |
|---|---|
| `paper_combined_model_summary.csv` | Combined tree, TabFM, and TabPFN summary |
| `tabpfn_tabfm_summary.csv` | Foundation-model metrics and confidence intervals |
| `tabpfn_tabfm_oof_predictions.csv` | Mean repeated OOF prediction for each sample and model |
| `foundation_predictions_by_repeat.csv` | Individual OOF predictions from every repeat |
| `foundation_fold_metrics.csv` | Metrics from all 15 outer test fits |
| `foundation_repeat_metrics.csv` | Metrics from each complete repeated partition |
| `paired_cluster_bootstrap_comparisons.csv` | Paired model-difference intervals |
| `cluster_bootstrap_metric_distributions.csv` | Raw bootstrap distributions |

### Feature and split audit

| File | Description |
|---|---|
| `split_assignments.csv` | Train/test role of every sample in every fold |
| `fold_selected_auxiliary_features.csv` | Auxiliary features selected within each training fold |
| `auxiliary_feature_selection_frequency.csv` | Selection stability across folds |
| `column_audit.csv` | Workbook-column inclusion and exclusion audit |

### Figures

| File | Description |
|---|---|
| `paper_model_accuracy_comparison.png/.pdf` | R², Spearman, and Top-20% recall comparison |
| `paper_model_error_comparison.png/.pdf` | RMSE and MAE comparison |
| `tabfm_formulation_grouped_cv_parity.png/.pdf` | TabFM parity plot |
| `tabpfn_formulation_grouped_cv_parity.png/.pdf` | TabPFN parity plot |
| `tabfm_formulation_grouped_cv_residuals.png/.pdf` | TabFM residual diagnostic |
| `tabpfn_formulation_grouped_cv_residuals.png/.pdf` | TabPFN residual diagnostic |

### Reproducibility records

| File | Description |
|---|---|
| `configuration.json` | Resolved command-line arguments |
| `environment_versions.csv` | Software versions and SHA-256 hashes |
| `run_manifest.json` | Complete run summary and output metadata |
| `run.log` | Console and diagnostic log |
| `tabpfn_tabfm_publication_results.xlsx` | Consolidated workbook of major outputs |

Before public release, remove or replace local absolute paths, hostnames, tokens, and user-specific executable paths from public copies of configuration or environment files.

## Metric definitions

### R²

R² measures the proportion of response variability explained by the predictions. Higher values indicate better numerical agreement, but R² alone does not measure ranking quality.

### RMSE and MAE

RMSE penalizes large errors more strongly. MAE reports the average absolute prediction error in normalized DC2.4 units.

### Spearman rank correlation

Spearman correlation measures agreement between experimental and predicted rankings. It is especially relevant when the practical goal is to prioritize candidate formulations rather than reproduce every absolute value exactly.

### Top-20% recall

Top-20% recall is calculated as:

```text
number of true top-20% formulations also present in the predicted top-20%
divided by
number of true top-20% formulations
```

For 84 samples, the top 20% contains 17 samples.

### Calibration slope

A calibration slope below 1 indicates that predictions are compressed toward the dataset mean. High-performing formulations may be underestimated and low-performing formulations may be overestimated even when mean bias is close to zero.

## Interpretation guidance

The primary intended use is **candidate prioritization**.

The current results support the following conclusions:

- the formulation data contain reproducible predictive structure;
- TabFM and TabPFN perform comparably to the conventional tree ensemble;
- rank-based performance is useful for formulation screening;
- extreme response values remain harder to predict;
- prospective wet-lab validation is still required.

The models should not be presented as replacements for experimental validation.

## Privacy and remote inference

The default `auto` TabPFN backend prefers `tabpfn-client`.

When the hosted client is used:

- training features and target values are sent to Prior Labs;
- test features are submitted for remote inference;
- access tokens must be supplied through environment variables;
- credentials must never be written directly into source code or committed to Git.

For data that cannot be transferred remotely, use:

```bash
--tabpfn-backend local
```

or:

```bash
--no-tabpfn
```

## Licensing

The repository license applies only to code and materials owned by this project.

It does not override third-party terms for:

- TabPFN software, hosted services, or model weights;
- TabFM source code;
- TabFM pretrained weights;
- RDKit;
- PyTorch;
- other scientific Python dependencies.

TabFM pretrained weights used by this workflow are subject to a separate non-commercial license. Model weights and access tokens must not be committed to this repository.

## Reproducibility checklist

Archive the following with a formal release:

- exact code commit;
- processed dataset and SHA-256 hash;
- split assignments;
- OOF predictions;
- fold- and repeat-level metrics;
- cluster-bootstrap distributions;
- paired bootstrap comparisons;
- package versions;
- public run manifest;
- figure outputs;
- data dictionary and QC rules.

A successful formal run should end with:

```text
[Done] Publication-oriented TabPFN / TabFM grouped-CV benchmark completed.
```

and an operating-system exit code of `0`.

## Troubleshooting

### TabPFN is reported as unavailable

Verify that `tabpfn-client` or local `tabpfn` is installed in the active interpreter. For hosted inference, confirm that a valid access token is available in the environment.

### TabFM is not found

Install TabFM in `.venv_tabfm` or run the script from the interpreter containing TabFM.

### RDKit cannot load on Windows

The workflow attempts one conservative wheel-only repair. If Windows application-control policy blocks RDKit DLL files, install RDKit in an approved environment or add precomputed fingerprint columns such as `fp_0` through `fp_127` to the workbook.

### Multiple workbooks are found

Byte-identical copies are detected and one canonical copy is selected. If multiple candidate workbooks have different SHA-256 hashes, the script stops rather than guessing which dataset is correct.

### A prior tree-model OOF file cannot be merged

Tree-model integration is optional. The completed TabPFN and TabFM outputs are saved before the merge is attempted, so an incompatible historical tree file does not invalidate the completed benchmark.

## Citation

Until a manuscript DOI or software release DOI is available, cite the repository and exact Git commit used for the analysis.

Suggested temporary software citation:

```text
Campbell, C. DC2.4 LNP Formulation Optimization: Leakage-Safe TabPFN and
TabFM Grouped Cross-Validation Benchmark. GitHub repository, version 2.0.5.
```

Replace the author name, repository URL, release version, year, and DOI with the final publication metadata.

## Contact

For questions about the dataset, formulation definitions, or reproduction of the benchmark, open an issue in the project repository.
