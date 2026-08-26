# Iterative LNP Formulation Design
## Reconstructed historical workflow, mathematical selection rules, and reproducibility notes

**Project scope:** small-sample, target-specific optimization of mRNA-LNP formulations for DC2.4 transfection.

**Important terminology note:** In this repository, **AI-assisted** refers to the combination of machine-learning prediction, AI-supported experimental planning, and sequential model-informed candidate selection. **D-optimal design itself is a classical design-of-experiments (DoE) method, not an AI model.**

---

# 1. Why this README exists

The formulation campaign was developed iteratively with AI-assisted planning, Python scripts, intermediate spreadsheets, and experimental feedback. Not every historical decision survived as one clean, publication-ready script. This README therefore reconstructs the workflow from the strongest available evidence:

1. surviving executable code;
2. contemporaneous planning notes;
3. the actual formulation tables that were experimentally prepared;
4. later round-specific design workbooks.

The reconstruction is intended to describe **what was actually done as closely as possible**, rather than retroactively forcing every round into one modern Bayesian-optimization framework.

Where an exact historical numerical acquisition score cannot be recovered, the README says so and reports the **best-supported reconstruction**.

---

# 2. Overall experimental-design concept

The campaign can be summarized as a sequential transition from global space coverage to target-specific local refinement:

```text
Prior lipid/formulation knowledge
        ↓
R1 — global structured exploration
        ↓
Experimental DC2.4 response
        ↓
R2 — response-restricted D-optimal / mixture-space expansion
        ↓
Experimental feedback + interim model analysis
        ↓
R3 — TabPFN-assisted active refinement of promising regions
        ↓
Experimental feedback + model updating
        ↓
R4 — hybrid exploit + explore + calibration round
        ↓
Final cumulative R1–R4 modeling
        ↓
Freeze model / preprocessing
        ↓
Prospective testing of unseen formulations
```

The central design principle was therefore not simply “add more samples,” but:

> **allocate each new experimental batch either to cover an under-characterized region, refine a promising region, or verify the stability of a previously identified high-response formulation.**

---

# 3. Formulation variables

The experimentally controllable formulation space included:

- ionizable/cationic lipid 1 identity;
- ionizable/cationic lipid 2 identity;
- IL1 mol%;
- IL2 mol%;
- IL1:IL2 relative ratio;
- total ionizable-lipid fraction;
- helper phospholipid identity (primarily DOPE or DSPC);
- helper-phospholipid mol%;
- cholesterol mol%;
- PEG-lipid identity;
- PEG-lipid mol%.

The mixture was subject to a composition constraint of approximately

\[
\sum_j x_j \approx 100\;\mathrm{mol\%}
\]

together with wet-lab feasibility constraints.

Later publication modeling additionally incorporated ionizable-lipid molecular information derived from SMILES. An early utility script (`DatabaseImport.py`) queried PubChem for canonical SMILES / molecular metadata. That script is **molecular-data acquisition infrastructure**, not a formulation-selection algorithm.

---

# 4. R1 — global structured exploration

## 4.1 Historical intent

The early planning documents repeatedly used the language of:

- broad space filling;
- Latin-hypercube-style / maximum-space exploration;
- D-optimal design.

The surviving executable R1 code resolves the ambiguity in the **implemented** method: the final saved Python procedure generated a finite candidate universe and selected 36 formulations by a greedy log-determinant D-criterion.

Therefore, for reproducibility, this repository describes the executed R1 method as:

> **knowledge-enhanced greedy D-criterion (D-optimal-inspired) experimental design**

rather than claiming that the final 36 formulations were produced by a continuous LHS implementation.

If an earlier LHS script is recovered in the future, it can be documented as an upstream candidate-generation step. At present, the surviving executable evidence supports the D-criterion procedure below.

---

## 4.2 R1 candidate universe

The saved script generated:

\[
6\times6\times2\times3=216
\]

discrete candidates using:

### Ionizable/cationic lipid palette

- ALC-0315
- SM102
- MC3
- C12-200
- DOTAP
- DODAP

### Helper phospholipids

- DOPE
- DSPC

### Three formulation templates

| Template | Total ionizable lipid | PL | Cholesterol | PEG |
|---|---:|---:|---:|---:|
| Type 1 | 50.0 | 10.0 | 38.5 | 1.5 |
| Type 2 | 58.0 | 20.0 | 19.25 | 2.75 |
| Type 3 | 66.0 | 30.0 | 0.0 | 4.0 |

Within the R1 candidate generator, IL1 and IL2 each received half of the total ionizable-lipid fraction.

---

## 4.3 R1 feature encoding

Categorical variables were one-hot encoded:

\[
x_{\mathrm{cat}}=
[
\mathrm{onehot}(IL1),
\mathrm{onehot}(IL2),
\mathrm{onehot}(PL),
\mathrm{onehot}(Template)
].
\]

The numerical design block contained:

\[
x_{\mathrm{num}}=
[
IL1\%,IL2\%,PL\%,Chol\%,PEG\%,TotalIL\%,
I_{\mathrm{same}},R_{IL/PL}
].
\]

The identical-lipid indicator was:

\[
I_{\mathrm{same}}=
\begin{cases}
1,& IL1=IL2\\
0,& IL1\neq IL2
\end{cases}
\]

and the ionizable-lipid / helper-lipid ratio feature was:

\[
R_{IL/PL}
=
\frac{TotalIL\%}{PL\%+0.1}.
\]

The complete feature matrix was standardized using `StandardScaler`.

---

## 4.4 R1 selection function

For a selected subset \(S\), with standardized design matrix \(X_S\), the historical code calculated

\[
\boxed{
D(S)=\log\det(X_S^\mathrm{T}X_S)
}
\]

and greedily added the candidate producing the largest log-determinant:

\[
\boxed{
x^*
=
\arg\max_{x\in\mathcal C\setminus S}
\log\det
\left[
X_{S\cup\{x\}}^\mathrm{T}
X_{S\cup\{x\}}
\right]
}
\]

where \(\mathcal C\) is the 216-formulation candidate set.

The historical implementation:

- initialized from a seeded random candidate;
- evaluated up to 200 remaining candidates in each greedy iteration;
- used `random_seed = 42`;
- selected 36 formulations.

This is a **greedy approximation to a D-optimal design objective** rather than an exact global combinatorial solution.

---

## 4.5 Knowledge-guided tie breaking

The historical R1 code also contained weak prior bonuses:

- C12-200: +0.05
- DOTAP: +0.03
- CKK-E12: +0.02
- remaining lipids: 0

When the leading two D-criterion values differed by <5%, the candidate with the larger knowledge bonus could be preferred.

Because CKK-E12 was not included in the six-lipid R1 candidate universe in the surviving executable script, the CKK-E12 bonus did not affect that run.

Thus R1 is best described as:

> **global space coverage dominated by the D-criterion, with weak prior knowledge used only as a secondary tie-breaking rule.**

---

# 5. R2 — response-restricted D-optimal / mixture-space expansion

## 5.1 What the original notes show

The contemporaneous planning notes for R2 explicitly proposed:

- identify the best-performing lipid families from R1;
- refine IL1:IL2 ratios;
- refine helper-lipid fraction;
- refine cholesterol concentration;
- test additional PEG conditions;
- use D-optimal / mixture-design logic to avoid the full factorial burden.

The early R1-analysis code also explicitly generated:

- ratio scans such as 70:30, 60:40, 50:50, 40:60, 30:70;
- cholesterol gradients;
- cross-combinations of promising lipid pairs.

The final R2 workbook is broader than that early prototype, indicating that the design was subsequently expanded.

---

## 5.2 What the final R2 formulation table shows

The final 40-formulation R2 table contains a deliberately broadened but **R1-informed** candidate space.

### Lipid-pair families represented

The final table contains eight principal pair families, including:

- C12-200 + DOTAP;
- C12-200 + SM102;
- C12-200 + CKK-E12/CCK12;
- CKK-E12/CCK12 + SM102;
- CKK-E12/CCK12 + CKK-E12/CCK12;
- SM102 + DOTAP;
- SM102 + ALC-0315;
- SM102 + SM102.

### Discrete ratio levels represented

\[
20:80,\;35:65,\;50:50,\;65:35,\;80:20
\]

### Helper-lipid fractions represented

\[
5,\;15,\;25,\;35,\;50\;\mathrm{mol\%}
\]

### Cholesterol levels represented

\[
19.25,\;25.5,\;32.0,\;38.5\;\mathrm{mol\%}
\]

### PEG identities represented

- DMG-PEG2000
- C14-PEG
- PEG-Mannose

with multiple PEG mol% levels.

This pattern is highly consistent with the contemporaneous plan to create a **restricted mixture-design candidate space around R1-informed lipid families and then select a limited informative subset**.

---

## 5.3 Best-supported reconstruction of the R2 algorithm

The exact final R2 Python down-selection script has not yet been recovered. However, three independent pieces of evidence support the same reconstruction:

1. the R2 workbook was historically named as a `D_optimal_40_formulations` design;
2. the contemporaneous notes explicitly specify **D-optimal mixture design** for the second-stage mapping;
3. the final 40 formulations cover multiple discrete levels of all major variables rather than simply taking the 40 highest-response local variants.

Therefore, the best-supported reconstruction is:

### Step 1 — response-restricted candidate-space construction

Use R1 experimental response to reduce the original lipid universe to promising and scientifically informative lipid families:

\[
\mathcal F_2
=
\text{R1-informed lipid-pair families}.
\]

Construct a factorial / mixture candidate set:

\[
\mathcal C_2
=
\mathcal F_2
\times
\mathcal R
\times
\mathcal P
\times
\mathcal H
\times
\mathcal G
\]

where:

- \(\mathcal R\) = IL1:IL2 ratio grid;
- \(\mathcal P\) = helper-lipid type/fraction grid;
- \(\mathcal H\) = cholesterol grid;
- \(\mathcal G\) = PEG type/fraction grid.

Candidates violating mixture feasibility were excluded.

### Step 2 — information-spreading down-selection

The historical R2 selection is reconstructed as a D-criterion down-selection analogous to R1:

\[
\boxed{
S_2^*
\approx
\arg\max_{|S|=40}
\log\det(X_S^\mathrm{T}X_S)
}
\]

within the **R1-informed reduced candidate space**.

This is not equivalent to simply taking the top 40 predicted formulations.

### Interpretation

R2 therefore combined:

- **exploitation at the family level** — promising lipid families from R1 were emphasized;
- **exploration at the parameter level** — ratio, PL, cholesterol and PEG conditions remained broad.

A concise description is:

> **response-informed candidate-space restriction followed by D-optimal-style mixture-space coverage.**

---

# 6. R3 — TabPFN-assisted active local refinement

R3 is the first round for which the surviving design workbook explicitly uses the language:

> **“TabPFN active learning strategy”**

and describes the sampling principle as prioritizing regions with:

> **high expected response + high uncertainty**

The R3 workbook is titled as a TabPFN strategy and its final 26-formulation sheet states that the design is based on the TabPFN active-learning iteration strategy.

This provides substantially stronger evidence for **model-informed active selection** than was available from the earlier summary alone.

---

## 6.1 R3 training information

The R3 planning logic used the accumulated R1 + R2 data to identify high-response formulation families and parameter interactions.

The contemporaneous strategy document emphasizes:

- small-sample learning;
- predicted high-response regions;
- predictive uncertainty;
- nonlinear interactions between IL ratio, helper lipid and cholesterol;
- focused experimental allocation rather than another broad global screen.

---

## 6.2 R3 modular batch design

The actual R3 formulation table contains 26 formulations divided into five explicit modules.

### Module A — CKK-E12 + SM102

Purpose:

- systematic IL1:IL2 ratio refinement;
- DOPE / cholesterol comparison;
- PEG-Mannose comparison.

Representative ratio scan:

\[
70:30,\;60:40,\;50:50,\;40:60
\]

around a common local composition background.

### Module B — C12-200 + SM102

Purpose:

- dense sampling around high-response R2 anchors;
- cholesterol interpolation;
- DOPE-fraction perturbation;
- PEG-type comparison.

### Module C — SM102 + ALC-0315

Purpose:

- explore an additional high-potential family identified from R2;
- test directional ratio changes and low-helper-lipid conditions.

### Module D — high-SM102 formulations

Purpose:

- test whether high SM102 content remains effective under alternative helper-lipid / PEG conditions;
- evaluate robustness of a previously strong but sparsely sampled region.

### Module E — validation repeats

Purpose:

- reproduce selected R2 high performers;
- separate genuine formulation response from single-batch experimental noise.

The final R3 table explicitly contains historical-anchor formulations based on selected R2 candidates.

---

## 6.3 Reconstructed active-learning rule for R3

The historical documents contain two related Bayesian/active-learning concepts:

1. **Expected Improvement (EI)** appears in early planning as a Bayesian-optimization acquisition option;
2. **Upper Confidence Bound (UCB)** appears in the surviving TabPFN active-learning prototype as

\[
\boxed{
UCB(x)=\mu(x)+\kappa\sigma(x)
}
\]

where:

- \(\mu(x)\) is the predicted response;
- \(\sigma(x)\) is predictive uncertainty;
- \(\kappa\) controls exploration strength.

The surviving code written for the later R4 planning prototype used \(\kappa=0.5\).

For R3, the exact saved candidate-by-candidate acquisition-score table is not available. Therefore the most defensible reconstruction is **UCB-like constrained batch active learning**, rather than claiming that every R3 point was the numerical top-ranked EI or UCB candidate.

Operationally, the R3 batch can be represented as:

\[
\boxed{
S_3
=
S_{\mathrm{exploit}}
\cup
S_{\mathrm{explore}}
\cup
S_{\mathrm{validate}}
}
\]

where

\[
S_{\mathrm{exploit}}
=
\text{high predicted-response candidates near R1–R2 high-response regions},
\]

\[
S_{\mathrm{explore}}
=
\text{uncertain / boundary / compositionally informative candidates},
\]

and

\[
S_{\mathrm{validate}}
=
\text{historical high-response anchors}.
\]

The model-generated ranking was therefore combined with explicit scientific and wet-lab constraints.

---

## 6.4 Local refinement equation

The observed R3 design can also be written as constrained perturbation around a promising anchor:

\[
\boxed{
x_{\mathrm{new}}
=
x_{\mathrm{anchor}}+\Delta x
}
\]

subject to

\[
x_{\mathrm{new}}\in\mathcal X,
\]

\[
l_j\leq x_j\leq u_j,
\]

and

\[
\sum_j x_j\approx100.
\]

This equation describes the actual structure of the R3 table particularly well: nearby points differ along selected axes such as IL ratio, DOPE fraction, cholesterol fraction or PEG identity.

---

# 7. Bayesian-optimization language: what can and cannot be claimed

The historical notes clearly show that **Bayesian-style active-learning concepts were part of the project planning**.

The early notes explicitly mention a `Bayesian_Optimizer` with `expected_improvement`, and later notes discuss:

- exploitation by predicted response;
- exploration by uncertainty;
- UCB;
- EI;
- iterative model updating.

However, the experimental record does not support describing the entire R1–R4 campaign as one strict Gaussian-process Bayesian-optimization loop.

The recommended terminology is therefore:

> **sequential hybrid DoE and model-informed active-learning design**

or

> **data-efficient, target-specific, AI-assisted sequential experimental design**

For R3–R4 specifically, it is reasonable to use:

> **Bayesian-style exploration–exploitation logic**

provided that the Methods distinguish the conceptual acquisition logic from a fully preserved numerical GP-BO implementation.

---

# 8. Expected Improvement as the optimization concept

The early planning notes explicitly refer to Expected Improvement.

For maximization, EI can be written as:

\[
z(x)=
\frac{\mu(x)-y_{\mathrm{best}}-\xi}{\sigma(x)}
\]

and

\[
\boxed{
EI(x)
=
[\mu(x)-y_{\mathrm{best}}-\xi]\Phi(z)
+
\sigma(x)\phi(z)
}
\]

where:

- \(y_{\mathrm{best}}\) is the best observed response;
- \(\mu(x)\) is the predicted mean;
- \(\sigma(x)\) is predictive uncertainty;
- \(\xi\) is an exploration offset;
- \(\Phi\) and \(\phi\) are the standard-normal CDF and PDF.

EI explains the logic of selecting candidates that have both:

- a high predicted response;
- sufficient uncertainty to plausibly exceed the current best.

In the historical project, EI should be described as part of the **Bayesian optimization / active-learning design concept**, not as a universally preserved numerical ranking function for every R2–R4 point.

---

# 9. R4 — hybrid TabPFN/UCB-informed exploit–explore–calibration design

The surviving R3 workbook contains a Python prototype explicitly written for **Round 4 generation after R3**.

That prototype:

1. trains `TabPFNRegressor`;
2. generates a focused candidate space;
3. estimates a predicted mean and uncertainty;
4. computes

\[
\boxed{
UCB(x)=\mu(x)+0.5\,\sigma(x)
}
\]

and proposes candidates with high UCB.

This is the clearest surviving explicit acquisition-function code in the historical files.

However, the final experimental R4 batch is broader than a pure “top-UCB” list. The final formulations clearly contain distinct functional blocks.

---

## 9.1 R4 Exploit / local response-surface refinement

Examples in the final table include a systematic C12-200/SM102 ratio scan:

\[
70:30,\;60:40,\;50:50,\;40:60,\;30:70
\]

under a common background composition.

Another local sequence holds the C12-200/SM102 ratio approximately constant while varying the DOPE/cholesterol balance.

These points were designed to resolve the local shape of a promising response region rather than to expand the global space.

---

## 9.2 R4 Explore / boundary sampling

A second R4 block contains substantially more diverse lipid pairs and more extreme compositions.

The purpose of these points was not necessarily to maximize predicted transfection. They sampled:

- underrepresented lipid combinations;
- composition boundaries;
- high-uncertainty regions;
- conditions that could identify low-performance or formulation-failure boundaries.

This is consistent with the historical active-learning objective of preventing the model from collapsing into one local optimum.

---

## 9.3 R4 Calibration / historical anchors

The final R4 table contains direct or near-direct repeats of R3 formulations, including entries explicitly labeled as R3 historical formulations.

These samples provided:

- cross-round reproducibility information;
- batch-shift calibration;
- a way to distinguish formulation effects from assay-scale changes.

---

## 9.4 Best-supported reconstruction of R4 selection

The most plausible as-executed interpretation is:

\[
\boxed{
S_4
=
S_{\mathrm{UCB/exploit}}
\cup
S_{\mathrm{boundary/explore}}
\cup
S_{\mathrm{calibration}}
}
\]

with the TabPFN/UCB prototype acting as one decision-support component rather than the sole rule for the complete 30-formulation batch.

Accordingly, R4 should be described as:

> **a hybrid batch active-learning round combining model-informed exploitation, boundary exploration and historical calibration.**

---

# 10. Why this is still a legitimate active-learning story

The campaign did not use the same acquisition equation in every round, but that is not a methodological weakness.

The information need changed over time:

### R1

Question:

> Where is the global formulation space?

Tool:

> D-criterion / space coverage.

### R2

Question:

> Which R1-informed lipid families deserve denser but still broad parameter coverage?

Tool:

> response-restricted D-optimal / mixture-space design.

### R3

Question:

> What is the local geometry of the high-response regions, and which uncertain regions remain worth testing?

Tool:

> TabPFN-assisted prediction + uncertainty-aware active refinement + explicit validation modules.

### R4

Question:

> Can the model further refine the best local regions without losing information about boundaries and experimental reproducibility?

Tool:

> UCB-informed exploitation + exploration + calibration.

Thus the overall campaign is better described as an **adaptive experimental-design policy** than as a single fixed acquisition function.

---

# 11. Retrospective information-value audit

To quantitatively test the claim that later-round points were informative, a separate retrospective analysis is performed.

This audit is deliberately separated from the historical selection algorithm.

For each round:

### R2 audit

Train only on R1:

\[
D_{\mathrm{train}}=R1
\]

then score the actual R2 formulations.

### R3 audit

Train only on R1+R2:

\[
D_{\mathrm{train}}=R1+R2
\]

then score the actual R3 formulations.

### R4 audit

Train only on R1+R2+R3:

\[
D_{\mathrm{train}}=R1+R2+R3
\]

then score the actual R4 formulations.

This avoids temporal leakage.

---

## 11.1 Predictive uncertainty

With a Gaussian-process audit surrogate:

\[
f(x)\mid D_t
\sim
\mathcal N
\left[
\mu_t(x),\sigma_t^2(x)
\right].
\]

Uncertainty score:

\[
\boxed{
U_t(x)=\sigma_t(x)
}
\]

---

## 11.2 Expected Improvement

\[
\boxed{
EI_t(x)
=
[\mu_t(x)-y_{\mathrm{best}}-\xi]\Phi(z)
+
\sigma_t(x)\phi(z)
}
\]

with

\[
z=
\frac{\mu_t(x)-y_{\mathrm{best}}-\xi}
{\sigma_t(x)}.
\]

This is primarily an **optimization-value** measure.

---

## 11.3 Expected information gain

A GP-based single-point expected information-gain score can be written as:

\[
\boxed{
IG_t(x)
=
\frac{1}{2}
\log
\left(
1+
\frac{\sigma_{f,t}^2(x)}
{\sigma_n^2}
\right)
}
\]

where:

- \(\sigma_{f,t}^2(x)\) is latent predictive variance;
- \(\sigma_n^2\) is estimated observation-noise variance.

This quantifies expected model-learning value under the GP assumptions.

---

## 11.4 Formulation-space novelty

A response-independent novelty score is:

\[
\boxed{
N_t(x)
=
\min_{x_i\in D_t}
\|
\tilde{x}-\tilde{x}_i
\|_2
}
\]

where \(\tilde{x}\) is the encoded/scaled formulation vector.

This is particularly useful for distinguishing:

- local refinement points;
- genuinely space-expanding points.

---

## 11.5 Why the metrics remain separate

The audit does **not** collapse EI, uncertainty, information gain and novelty into one arbitrary weighted score.

They answer different questions:

| Metric | Main interpretation |
|---|---|
| Predicted mean | expected performance |
| EI | potential to improve over the current best |
| uncertainty | how little the model knows locally |
| information gain | expected learning value |
| novelty | distance from previous experimental support |

A candidate can therefore be valuable even if it is not predicted to be the highest-performing point.

---

# 12. UMAP formulation map

The two-dimensional formulation map should be described as a visualization of high-dimensional formulation similarity, not as the predictive model itself.

Each formulation is represented by a multi-dimensional composition vector. UMAP builds a nearest-neighbor graph in the original feature space and finds a two-dimensional embedding that approximately preserves local fuzzy neighborhood relationships.

Conceptually, UMAP minimizes a cross-entropy of the form:

\[
C=
\sum_{i\neq j}
\left[
p_{ij}\log\frac{p_{ij}}{q_{ij}}
+
(1-p_{ij})
\log
\frac{1-p_{ij}}{1-q_{ij}}
\right].
\]

For the publication figure:

- formulation variables determine UMAP coordinates;
- measured or predicted DC2.4 transfection should **not** be used to generate the coordinates;
- response can subsequently be overlaid as point color, contour or surface height.

The exact final feature list, distance metric, `n_neighbors`, `min_dist`, `spread`, and random seed should be copied from the frozen figure-generation script into the modeling README.

---

# 13. Molecular structure information

An early script (`DatabaseImport.py`) queried PubChem to retrieve canonical SMILES and molecular information by CID.

This documents the origin of part of the molecular-structure metadata workflow.

In the final publication modeling pipeline, structural information should be described separately from historical round selection:

```text
lipid identity / SMILES
        ↓
molecular descriptors
        ↓
Morgan fingerprint
        ↓
IL1/IL2 composition-weighted molecular representation
        ↓
fold-safe feature selection
        ↓
final predictive modeling
```

The structural-feature pipeline belongs to **final model development**, not to the original R1 D-criterion calculation unless the historical design script explicitly used those descriptors.

---

# 14. Recommended manuscript wording

A concise Methods-style description consistent with the reconstructed record is:

> The experimental campaign used a sequential hybrid design strategy in which the selection rule changed with the information available at each stage. R1 was generated from a finite 216-formulation candidate space and down-selected using a greedy log-determinant D-criterion after categorical and continuous formulation-feature encoding. Following the R1 response screen, R2 restricted the design space toward experimentally promising lipid families while retaining systematic variation in ionizable-lipid ratio, helper-lipid fraction, cholesterol and PEG conditions; the final 40-formulation batch is most consistent with D-optimal-style down-selection within this response-informed mixture space. R3 then shifted toward TabPFN-assisted active refinement of high-response regions, combining model-predicted performance, uncertainty-aware exploration, controlled local perturbation and replicate validation. R4 further combined model-informed exploitation, broader boundary exploration and historical calibration. Thus, the overall workflow progressively transitioned from global design-space coverage toward local response-surface refinement while preserving selected exploratory measurements.

For the Bayesian component:

> Bayesian-style acquisition concepts were used as decision-support rules for later-round active sampling. The planning framework considered both expected improvement and upper-confidence-bound criteria, \(UCB(x)=\mu(x)+\kappa\sigma(x)\), to balance predicted response and uncertainty. Because the final experimental batches also incorporated explicit diversity, boundary and calibration constraints, later-round selection is described as constrained batch active learning rather than as a single unconstrained Bayesian-optimization ranking.

---

# 15. Recommended Results wording for “information value”

Before the retrospective audit is run, use conservative wording:

> Later experimental rounds were allocated more deliberately toward either high-response local refinement or under-characterized regions, with the aim of increasing the information obtained per experiment.

After the leakage-safe audit and random-batch comparison are complete, stronger wording can be used **only if supported by the results**, for example:

> Retrospective prior-round-only acquisition analysis showed that the selected later-round batches were enriched for candidates with higher model uncertainty, expected information gain and/or local optimization value than randomly sampled feasible batches.

Do not state a numerical efficiency improvement (for example “3× more efficient”) unless a defined random or baseline strategy has actually been simulated and compared.

---

# 16. What should be called “AI” in the paper

Recommended:

- **AI-assisted LNP formulation design**
- **model-informed sequential experimental design**
- **TabPFN-assisted active refinement**
- **data-efficient target-specific optimization**
- **hybrid DoE + active-learning workflow**

Avoid:

- calling D-optimal itself an “AI algorithm”;
- claiming that every R2–R4 point was selected by one exact Bayesian acquisition function;
- claiming “information gain increased” without the retrospective calculation;
- describing UMAP as the predictive AI model.

---

# 17. Designed / tested counts versus final modeling counts

Historical design files contain the number of formulations **designed or prepared** in each round, while the final publication dataset may contain fewer observations after:

- QC filtering;
- removal of failed preparations;
- duplicate / calibration handling;
- target normalization requirements.

Therefore repository documentation should report both:

```text
Designed/tested N
QC-retained modeling N
```

and should never silently replace one with the other.

The final modeling dataset should be treated as the authoritative source for the counts used in model-performance calculations.

---

# 18. GitHub repository structure

Recommended publication repository:

```text
README.md

data/
├── raw_design/
│   ├── R1_selected_36_formulations.xlsx
│   ├── R2_selected_40_formulations.xlsx
│   ├── R3_selected_26_formulations.xlsx
│   └── R4_selected_formulations.xlsx
│
├── processed/
│   └── R1_R4_publication_modeling_dataset.xlsx
│
└── metadata/
    ├── formulation_column_dictionary.csv
    └── lipid_structure_metadata.csv

scripts/
├── design/
│   ├── r1_greedy_dcriterion_design.py
│   ├── r2_design_reconstruction.md
│   └── r3_r4_active_learning_notes.md
│
├── audit/
│   └── historical_information_value_audit.py
│
├── structure/
│   └── pubchem_structure_import.py
│
└── modeling/
    └── publication_modeling_pipeline.py

docs/
├── experimental_design/
│   └── iterative_formulation_design.md
│
└── modeling/
    └── publication_modeling_README.md

results/
├── information_value/
└── model_benchmark/
```

---

# 19. Which historical files should be uploaded

## Recommended public/canonical files

1. **One clean R1 36-formulation workbook**
2. **The executable R1 D-criterion design code**
3. **One final corrected R2 40-formulation workbook**
4. **The clean R3 26-formulation table**
5. **The final R4 formulation table**
6. **The final R1–R4 publication modeling dataset**
7. **The frozen final modeling code**
8. **The retrospective information-value audit script**
9. **This experimental-design README**
10. **A cleaned PubChem / SMILES import utility if molecular metadata acquisition is part of the released pipeline**
11. **`requirements.txt` or environment specification**

## Do not place duplicate working copies in the main reproducibility path

Examples:

- files prefixed with `副本`;
- multiple `12.08`, `12.25`, `recalculate` versions of the same workbook;
- temporary concentration-calculation variants;
- prototype scripts containing simulated/mock fallback data unless clearly labeled;
- an old TabPFN uncertainty prototype presented as the exact final acquisition engine.

If historical preservation is desired, place these in:

```text
archive/
```

and label them explicitly as **development artifacts, not canonical analysis inputs**.

---

# 20. Provenance confidence table

| Component | Reconstruction confidence | Reason |
|---|---|---|
| R1 216-candidate universe | High | executable historical Python |
| R1 D-criterion equation | High | executable historical Python |
| R1 36-point selection | High | executable code + output workbook |
| R1 LHS as final executed selector | Low / unresolved | planning language exists, but current executable code is D-criterion |
| R2 response-informed family restriction | High | notes + R1-analysis code + final R2 compositions |
| R2 D-optimal-style 40-point down-selection | Medium-high | explicit planning + filename + final balanced parameter coverage; exact final script missing |
| R3 TabPFN-assisted strategy | High | workbook title + strategy sheet + final module table |
| Exact numerical R3 UCB/EI ranking | Medium / unresolved | acquisition concepts preserved, candidate-level ranking output missing |
| R4 TabPFN/UCB planning | High | surviving R4-generation prototype |
| R4 hybrid exploit/explore/calibration structure | High | final R4 formulation table + historical notes |
| Exact UCB ranking of every final R4 point | Medium | final batch contains additional explicit modules beyond pure top-UCB ranking |
| Retrospective EI/IG/novelty audit | High once executed | separate reproducible post-hoc analysis |

---

# 21. Final conceptual summary

The most defensible reconstruction of the complete campaign is:

\[
\boxed{
\text{R1: D-criterion global exploration}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{R2: response-restricted D-optimal / mixture-space expansion}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{R3: TabPFN-assisted uncertainty-aware local refinement}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{R4: UCB-informed exploit + explore + calibration}
}
\]

followed by final systematic model comparison and prospective validation.

The methodological novelty is therefore not that one acquisition equation was used unchanged from beginning to end. Rather, the campaign used **different experimental-design tools for different information needs**, progressively changing from global coverage to target-specific active refinement under a limited experimental budget.
