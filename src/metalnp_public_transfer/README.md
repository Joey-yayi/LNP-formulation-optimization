# MetaLNP-Style Public-to-In-House Benchmark

## Overview

This directory contains a **MetaLNP-style public-to-in-house transfer benchmark** for the final R1–R4 DC2.4 mRNA-LNP dataset.

The purpose of this analysis is **not** to retrain or replace the in-house tree-model / TabPFN / TabFM pipelines. Instead, it asks a separate question:

> Can knowledge learned from heterogeneous public LNP tasks transfer reliably to the target-specific DC2.4 mRNA-LNP formulation space?

The script trains public-task models using the processed MetaLNP dataset and then evaluates their transfer to the in-house DC2.4 dataset under zero-shot and few-shot settings.

This implementation reproduces the **MetaLNP-style task-based meta-learning concept** using pure PyTorch MAML, FoMAML, MetaSGD, and a supervised ANN baseline. It should not be described as an exact reproduction of the original MetaLNP repository.

---

## Final In-House Population

The benchmark is intentionally aligned to the same final R1–R4 publication population used by the main tree-model and TabPFN/TabFM analyses.

**Workbook**

`R1-4 all LNP normalized 1.35 (new).xlsx`

**Worksheet**

`Round1&2&3&4`

Minor worksheet-name whitespace differences are resolved automatically.

**Target**

`Normalized for DC2.4`

No alternative normalized, HaCaT, or raw response column is substituted.

### Publication-population preprocessing

The script reproduces the final in-house filtering sequence:

```text
182 rows with non-missing DC2.4 target
        ↓
invalid / template formulation removal
        ↓
140 formulation-valid rows
        ↓
physicochemical QC
        ↓
104 QC-passed R1–R4 rows
```

The script stops with an error if the final population does not contain **104 rows**, unless this safety check is explicitly disabled.

### Invalid/template formulation rules

The in-house formulation is retained only when:

- IL1 is one of:
  - MC3
  - ALC-0315
  - SM102
  - C12-200
  - CKK-E12
  - DOTAP
  - DODAP
- IL2 is empty or one of the same supported ionizable lipids;
- helper phospholipid is DOPE or DSPC;
- PEG lipid is C14-PEG, DMG-PEG2000, or PEG-Mannose;
- total lipid molar composition is within 80–120 mol%.

### QC rules

The publication QC defaults are:

- PDI ≤ 0.5
- particle size between 30 and 300 nm
- missing PDI excluded
- missing particle size excluded

Size and PDI are used for QC only; they are not used as public-to-in-house predictive features.

---

## Round Recovery

Historical replicate labels may contain more than one round identifier, for example:

```text
R1-17 (R3-28)
R4-25 (R3-11)
```

For audit purposes, the script assigns the **highest round number appearing in the label** as the current experimental round.

The expected final bookkeeping is approximately:

- R1: 30
- R2: 26
- R3: 28
- R4: 20

Round labels are **not used as predictive features**.

---

## Public MetaLNP Data

The default public training source is the processed MetaLNP training split:

```text
MetaLNPs/data/Processed/siRNAho/train_df_task_nosirna_clean.csv
```

The script detects:

- public target column;
- task ID;
- SMILES.

Public samples are organized into support/query tasks.

Default public task settings:

- support size = 10
- query size = 10
- task batch size = 8
- meta-training iterations = 800

The public training target is transformed to a **within-task percentile scale** to reduce direct scale mismatch across heterogeneous public tasks.

---

## Feature Representation

### Public features

The public feature representation contains:

- numeric formulation-composition variables when available;
- Morgan fingerprints generated from SMILES.

Default Morgan settings:

- radius = 4
- bits = 2048

### In-house features

The in-house feature representation is mapped into the same shared feature space using:

- total ionizable-lipid mol%;
- helper-lipid mol%;
- cholesterol mol%;
- PEG mol%;
- weighted IL1/IL2 Morgan fingerprints.

For dual-ionizable-lipid formulations:

```text
weighted fingerprint
=
IL1 fraction × Morgan(IL1)
+
IL2 fraction × Morgan(IL2)
```

Public and in-house feature matrices are aligned before transfer evaluation.

---

## Models

The benchmark evaluates:

- MAML
- FoMAML
- MetaSGD
- supervised ANN baseline

The meta-learning implementation is written in pure PyTorch.

The public models are trained only on public tasks before target-domain evaluation.

---

## Evaluation Modes

### A. Zero-Shot Transfer

```text
public training
        ↓
public-trained model
        ↓
direct prediction of all 104 in-house formulations
```

Zero-shot performance is primarily evaluated on the in-house percentile target because public and in-house raw response scales are not directly equivalent.

### B. Few-Shot Target Adaptation

Default publication benchmark:

```text
public training
        ↓
public-trained model
        ↓
80% in-house support set for adaptation
        ↓
20% in-house query/test set
```

This is repeated **20 times** using random support/query splits.

The repeated-random benchmark is kept as the primary MetaLNP transfer comparison because it measures whether public initialization provides useful target-domain adaptation across multiple target splits.

Round-based evaluation is available as an optional stress test:

```bash
--split_mode round
```

For example:

```text
R1 + R2 + R3 support/adaptation
        ↓
R4 query/test
```

This should be interpreted as a cross-round distribution-shift test rather than as the primary retrospective comparison.

---

## Metrics

The benchmark reports:

- R²
- RMSE
- MAE
- Pearson correlation
- Spearman correlation
- Top-20% precision

For transfer models, a negative R² indicates that calibrated numerical prediction is worse than predicting the query-set mean.

A positive Spearman correlation together with negative R² can indicate that some ranking information is retained even when numerical calibration across domains is poor.

---

## Important Interpretation

This benchmark should **not** be used to claim that MetaLNP is generally inferior to the in-house model.

The two approaches answer different questions.

### MetaLNP-style benchmark

```text
heterogeneous public tasks
        ↓
learn transferable prior knowledge
        ↓
transfer / few-shot adaptation to DC2.4
```

### In-house iterative strategy

```text
target-specific DC2.4 experiments
        ↓
model learning
        ↓
new target-specific experiments
        ↓
model updating
```

Therefore, the comparison is intended to test **cross-domain transferability**, not to establish a universal ranking of algorithms.

The central interpretation is:

> Public LNP knowledge may contain useful prior or ranking information, but reliable quantitative prediction in a new biological system depends strongly on alignment between the training data and the target experimental domain.

---

## Running the Script

### Direct run

The script automatically searches the current user's home directory for the development workbook and cloned MetaLNP repository.

```bash
python run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py
```

### Portable / GitHub use

Explicit paths are recommended for reproducibility:

```bash
python run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py \
    --inhouse_xlsx "path/to/R1-4 all LNP normalized 1.35 (new).xlsx" \
    --public_train_csv "path/to/MetaLNPs/data/Processed/siRNAho/train_df_task_nosirna_clean.csv"
```

### Quick test

```bash
python run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py \
    --quick \
    --models maml,metasgd
```

### Optional round-based stress test

```bash
python run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py \
    --split_mode round \
    --support_rounds "R1,R2,R3" \
    --query_rounds "R4"
```

---

## Required Packages

```text
numpy
pandas
scipy
scikit-learn
openpyxl
torch
rdkit
```

---

## Output Files

The main output directory contains:

### Core result files

`MetaLNP_ONLY_public_to_inhouse_results.xlsx`

Complete benchmark workbook containing:

- model summary
- repeat-level metrics
- predictions
- population audit
- final 104-row in-house population
- invalid/template removed rows
- QC removed rows
- aligned feature names
- configuration

`metalnp_only_model_summary.csv`

Compact summary of zero-shot and few-shot model performance.

`metalnp_only_repeat_metrics.csv`

Per-repeat few-shot results.

`metalnp_only_predictions.csv`

Row-level predictions for reproducibility.

### Population-audit files

`inhouse_population_audit.csv`

Expected to show:

```text
target_nonmissing                  182
after_invalid_template_removal     140
after_QC                           104
```

`inhouse_removed_rows.csv`

Combined audit of invalid/template and QC exclusions.

### Supporting output

`metalnp_feature_mapping_report.json`

Feature mapping and public/in-house alignment information.

Model `.pt` files are also generated but are not required for the main public results repository.

---

## Recommended GitHub Structure

```text
src/
└── metalnp_public_transfer/
    ├── README.md
    └── run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py

results/
└── metalnp_public_transfer/
    ├── MetaLNP_ONLY_public_to_inhouse_results.xlsx
    ├── metalnp_only_model_summary.csv
    ├── metalnp_only_repeat_metrics.csv
    ├── metalnp_only_predictions.csv
    └── inhouse_population_audit.csv
```

The `.pt` model weights, local run logs, and files containing machine-specific paths do not need to be uploaded to the main repository.

---

## Result Reporting

Do **not** copy results from the earlier 153-row run into the manuscript or final repository.

Only report the new benchmark after confirming that the console shows:

```text
[PopulationAudit] target_nonmissing=182
                  -> after_invalid_template_removal=140
                  -> after_QC=104
```

and that the benchmark completes successfully.

After the final 104-row run is completed, the result table in this README can be updated with:

- best few-shot raw-target R²
- RMSE
- Pearson
- Spearman
- Top-20% precision
- zero-shot performance
- numerical-stability status of MAML/FoMAML/MetaSGD

---

## Status

**Current code status:** publication-population alignment implemented.

**Required next step:** rerun the benchmark and confirm the final in-house sample count is 104 before public release or manuscript reporting.
