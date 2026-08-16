# Iterative LNP Formulation Design

## Purpose

This document describes **how formulations were selected across R1–R4** in the small-sample iterative LNP optimization study.

The focus of this file is the **experimental-design logic**:

- how the initial formulation space was explored;
- how experimental feedback was used to refine later rounds;
- how high-response regions were studied in greater detail without abandoning broader exploration;
- how R4 combined exploitation, exploration, and calibration/validation;
- how the completed R1–R4 dataset was separated from the planned prospective validation stage.

Detailed model training, feature engineering, grouped nested cross-validation, Morgan fingerprints, mRMR feature selection, and model benchmarking are documented separately in the publication modeling README.

---

## 1. Study Design Overview

The primary optimization task was **mRNA-LNP transfection in DC2.4 dendritic cells**.

The overall workflow followed one continuous iterative cycle:

**formulation design → DC2.4 transfection testing → model updating → candidate selection → next-round design**

R1 was used mainly for broad exploration of the formulation space. In subsequent rounds, experimental results and interim model predictions were used to increase sampling density in promising regions while retaining a degree of broader exploration.

The goal was not simply to increase sample number, but to make each new round more informative for the target task.

### Primary readout

- DC2.4 transfection efficiency
- normalized across experimental plates before cross-round modeling

### Secondary biological-context readout

- HaCaT keratinocyte transfection efficiency
- analyzed as a secondary biological context rather than as the optimization target of the primary DC2.4 model

### Final modeling dataset

After data cleaning and QC, **104 experimentally tested formulations from R1–R4** were retained for final systematic modeling.

The number of formulations originally designed or experimentally prepared in individual rounds can be larger than the final retained count because replicate, calibration, template, and QC-excluded rows are not all included in the final modeling dataset.

---

## 2. Formulation Space

Each LNP formulation was defined by a combination of categorical and continuous variables.

| Variable | Type | Range / options |
|---|---|---|
| Ionizable lipid 1 (IL1) | Categorical | MC3, ALC-0315, SM102, C12-200, CKK-E12, DOTAP, DODAP |
| IL1 mol% | Continuous | formulation-dependent |
| Ionizable lipid 2 (IL2) | Categorical | same lipid palette |
| IL2 mol% | Continuous | formulation-dependent |
| Total ionizable lipid | Continuous | IL1 + IL2 |
| Helper phospholipid | Categorical | DOPE, DSPC |
| Helper-lipid mol% | Continuous | formulation-dependent |
| Cholesterol mol% | Continuous | formulation-dependent |
| PEG lipid | Categorical | DMG-PEG2000, C14-PEG, PEG-Mannose |
| PEG mol% | Continuous | formulation-dependent |
| Mixture constraint | Constraint | total lipid components approximately sum to 100 mol% |

Publicly available LNP studies were used as **references for lipid selection and approximate formulation ranges**, rather than as direct training data for the target-specific DC2.4 model.

Specific formulation combinations and ratios were generated within the in-house experimental design and were experimentally tested in the DC2.4 system.

---

## 3. Round 1 — Broad Exploration

### Role

**Establish broad initial coverage of the formulation space and generate the first target-specific training data.**

R1 was designed under a **design-of-experiments (DoE) framework with Latin hypercube sampling (LHS)-style broad coverage**.

The purpose was not to immediately identify a single optimum. Instead, R1 was used to sample different regions of the formulation space and establish an initial relationship between formulation composition and DC2.4 transfection efficiency.

### Design priorities

- represent multiple ionizable-lipid identities and combinations;
- cover a broad range of component ratios;
- include different helper-lipid and PEG-lipid conditions;
- avoid concentrating all experiments in a narrow region before any target-specific response data were available;
- generate an initial dataset suitable for subsequent model fitting.

### Role in the iterative strategy

R1 can be viewed as the **initial set of sparse survey points** used to begin learning the DC2.4 formulation-response landscape.

---

## 4. Round 2 — Model-Informed Space Expansion

### Role

**Expand the formulation space using information learned from R1 while introducing additional formulation diversity.**

R2 was not simply a local replication of the best R1 formulations. It expanded the design by introducing additional lipid combinations and formulation conditions while increasing attention to regions that showed promising DC2.4 responses.

New formulation elements included, where experimentally applicable:

- CKK-E12-containing formulations;
- additional PEG-lipid variants such as PEG-Mannose and C14-PEG;
- systematic variation of ionizable-lipid ratios;
- helper-lipid, cholesterol, and PEG composition changes.

### Design logic

R2 combined two objectives:

1. **refinement** of regions that appeared promising from R1;
2. **continued exploration** of formulation combinations that remained insufficiently sampled.

The new experimental results were then returned to the modeling workflow for the next round.

---

## 5. Round 3 — Focused Refinement of High-Response Regions

### Role

**Increase local information density around promising formulation families while preserving selected exploratory and reproducibility samples.**

By R3, the accumulated R1–R2 dataset allowed the optimization process to move from primarily broad exploration toward more focused candidate selection.

Interim machine-learning models were used as decision-support tools together with the experimental results to identify formulation families and parameter ranges requiring further study.

### Main design questions

R3 was designed to ask:

- Which ionizable-lipid combinations consistently occupy high-response regions?
- How sensitive is transfection efficiency to IL1/IL2 ratio changes?
- Are promising responses robust to helper-lipid and cholesterol changes?
- Do high-performing formulations reproduce across experimental rounds?
- Which local formulation changes cause large decreases or increases in DC2.4 transfection?

### Candidate roles

R3 therefore included a mixture of:

- high-response-region refinement;
- local ratio and composition scans;
- selected broader probes;
- historical/technical replicates for reproducibility.

The purpose was to learn the **shape of the local high-response region**, rather than merely to reproduce one previously successful point.

---

## 6. Round 4 — Information-Rich Adaptive Refinement

### Role

**Use the knowledge accumulated from R1–R3 to perform targeted local refinement, broader boundary exploration, and calibration/validation within the same experimental round.**

R4 was not designed as a completely independent external test set. Instead, it was an **adaptive information-gathering round** whose results were later incorporated into the final R1–R4 model.

The R4 candidates were organized conceptually into three complementary roles.

### 6.1 Exploit — Refine High-Response Regions

The Exploit subset focused on formulation families that had already shown relatively high DC2.4 transfection efficiency.

Rather than repeating a single high-performing formulation, nearby formulation variables were varied to learn the local response landscape in more detail.

Variables examined included:

- IL1/IL2 identity and relative ratio;
- total ionizable-lipid content;
- helper-lipid fraction;
- cholesterol fraction;
- PEG-lipid composition and fraction.

Particular attention was given to promising dual-ionizable-lipid regions, including **C12-200/SM102** and **CKK-E12/SM102** families.

#### Scientific purpose

The goal was to determine whether a high-response formulation represented:

- a narrow isolated peak;
- a broader high-performance plateau;
- or a region highly sensitive to small changes in composition.

This subset therefore provided detailed local information around promising regions rather than simply maximizing the number of high predicted values.

### 6.2 Explore — Sample Underrepresented and Boundary Regions

The Explore subset retained broader coverage of the formulation space.

These formulations were chosen from regions that were less represented in the existing R1–R3 dataset, including composition boundaries and combinations outside the densest high-response region.

Explore candidates were **not required to have the highest predicted transfection efficiency**.

Their purpose was to provide informative examples that help define:

- low-response regions;
- formulation-space boundaries;
- feasible versus unstable formulation regions;
- composition regions with limited existing experimental support;
- areas in which model behavior required additional experimental information.

Low-performing results were therefore not considered failed experiments; they could still provide valuable information about the boundaries and structure of the formulation-response landscape.

### 6.3 Validate / Calibration — Historical Anchors and Transfer-Oriented Probes

A third subset was included for calibration and targeted validation.

This subset contained:

- replicate formulations derived from previously high-performing R3 formulations;
- selected CKK-E12-containing validation formulations;
- formulations inspired by high-performing regimes reported in external/public LNP studies;
- at least one boundary or model-disagreement probe.

Historical replicate formulations were used to monitor reproducibility and cross-round experimental shifts.

External/public LNP information was **not directly merged into the target-specific DC2.4 training dataset** for R4. Instead, public studies were used as prior references for candidate design and to test whether selected formulation ideas could transfer to the local experimental system.

---

## 7. Interpretation of R4

R4 should therefore be interpreted as an **adaptive refinement round**, not as a single homogeneous test set.

Its subsets answered different questions:

| R4 role | Main question |
|---|---|
| Exploit | What does the high-response region look like in greater local detail? |
| Explore | What can be learned from under-sampled, boundary, or lower-response regions? |
| Validate / Calibration | Are historical responses reproducible, are there cross-round shifts, and can selected external formulation ideas transfer? |

After QC, **20 R4 observations were retained** and incorporated into the final **104-formulation R1–R4 dataset**.

Because R4 contributes to final model refinement and training, it is **not treated as the final prospective external validation set**.

---

## 8. Final Systematic Modeling After R1–R4

After all R1–R4 experimental data had been collected, the accumulated 104-formulation dataset was used for final systematic model development and evaluation.

At this stage, the publication modeling pipeline incorporated:

- core formulation variables;
- ionizable-lipid structural descriptors;
- 128-bit Morgan molecular fingerprints generated from SMILES;
- IL1/IL2-weighted structural features for dual-ionizable-lipid formulations;
- fold-safe auxiliary feature selection using mutual information and mRMR;
- grouped nested cross-validation;
- comparison of multiple regression models.

These final retrospective modeling procedures were performed **after completion of the R1–R4 experimental acquisition process** and should not be interpreted as a separate experimental-design loop.

Detailed implementation is documented in the modeling README.

---

## 9. Public-Data / Meta-Learning Benchmark

Public LNP datasets were evaluated separately as a transfer-learning/meta-learning benchmark.

This analysis was designed to answer a different question:

> Can models trained on heterogeneous public LNP data transfer reliably to the in-house DC2.4 mRNA-LNP task?

The public-data benchmark is therefore a **comparison analysis**, not part of the target-specific R1–R4 training dataset.

Detailed MetaLNP-style benchmark methods and results are documented separately in the supporting analysis files.

---

## 10. Prospective Validation — Planned

The final prospective validation will be performed **after the R1–R4 model and preprocessing pipeline are frozen**.

Approximately **10–15 previously untested formulations** will be selected to span different predicted-response regions, including:

- predicted high-performing candidates;
- intermediate predicted candidates;
- predicted low-performing controls;
- a limited number of exploratory, boundary, or model-disagreement candidates.

Predictions and candidate rankings will be saved before wet-lab testing.

The prospective experiment is intended to evaluate whether the final model can **predict and prioritize genuinely unseen LNP formulations**, rather than only reproduce patterns already present in the retrospective R1–R4 dataset.

Candidate performance will be evaluated using complementary metrics such as:

- predicted-versus-measured agreement;
- Spearman rank correlation;
- RMSE/MAE when directly comparable;
- high-performer recall;
- Top-k hit rate or enrichment.

The success criterion is **reliable prediction and ranking of unseen formulations**, not necessarily discovery of a formulation exceeding every historical maximum.

---

## 11. Wet-Lab Feasibility Constraints

Candidate formulations must remain experimentally feasible.

Typical constraints include:

- lipid components approximately sum to 100 mol%;
- PEG-lipid fraction remains within the experimentally supported range;
- total ionizable-lipid fraction remains within the explored range;
- helper-lipid and cholesterol fractions remain compatible with formulation preparation;
- candidate lipids are restricted to reagents available in-house;
- particle size and PDI are evaluated as QC criteria before inclusion in modeling.

Exact QC thresholds and preprocessing rules are defined in the publication modeling pipeline.

---

## 12. Conceptual Summary

The experimental strategy can be summarized as:

```text
Prior formulation knowledge
        ↓
R1 — Broad exploration
        ↓
Experimental feedback + model learning
        ↓
R2 — Space expansion + early refinement
        ↓
Experimental feedback + model updating
        ↓
R3 — Focused refinement of high-response regions
        ↓
Experimental feedback + model updating
        ↓
R4 — Exploit + Explore + Validate/Calibration
        ↓
Final R1–R4 model (104 QC-passed formulations)
        ↓
Freeze model and preprocessing
        ↓
Prospective prediction of 10–15 unseen formulations
        ↓
Experimental validation
```

Conceptually, the successive rounds are analogous to placing sparse experimental **survey points** in an initially uncertain formulation landscape. Each round adds information that makes the response landscape clearer, allowing later experiments to be placed more deliberately while still preserving exploration of poorly characterized regions.

---

## 13. Relationship to Other Repository Documentation

This file documents **experimental formulation selection and round-by-round design logic**.

It should not duplicate the full modeling README.

Recommended documentation structure:

```text
README.md
└── project overview and links

docs/
├── experimental_design/
│   └── iterative_formulation_design.md   ← this file
│
└── modeling/
    └── publication_modeling_README.md    ← model training, features, CV, metrics

results/
└── model outputs, figures, and prospective-validation results
```

This separation keeps the repository reproducible while making it clear which decisions belong to **experimental design** and which belong to **final statistical/model evaluation**.
