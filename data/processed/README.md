# Processed data

This directory contains the curated LNP formulation dataset used for machine-learning model development and validation.

## Dataset

`lnp_dc24_hacat_modeling_dataset.xlsx`

The primary modeling target is plate-normalized DC2.4 transfection efficiency. HaCaT measurements are retained in the workbook for reference but are not used as the target or predictor variables in the DC2.4 model.

## Quality-control criteria

- PDI ≤ 0.50
- Particle size: 30–300 nm
- Samples with missing required size or PDI measurements are excluded from the primary model
