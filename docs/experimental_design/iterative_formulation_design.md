AI-Driven LNP Formulation Optimization — Algorithm & Reproducibility README
Purpose: This document describes the computational strategy, algorithms, and software implementation used across four rounds of lipid nanoparticle (LNP) formulation optimization. A reader with basic Python experience should be able to reproduce the full pipeline from scratch.

Overview
This project uses an iterative active learning framework to optimize LNP transfection efficiency in DC2.4 dendritic cells. Rather than exhaustively screening all possible formulations, a machine learning surrogate model (TabPFN) is trained on experimental data after each round, then used to predict and prioritize the next batch of candidates — dramatically reducing the number of wet-lab experiments required.
Total experimental budget: ~135 formulations across 4 rounds
Estimated efficiency gain: ~3× vs. random screening at equivalent sample size
Primary readout: DC2.4 cell transfection efficiency (RLU), normalized within each plate
Secondary readout (Round 4 onwards): HaCaT cell transfection efficiency

Formulation Space Definition
Each LNP formulation is defined by the following variables:
Variable	Type	Range / Options
Ionizable Lipid 1 (IL1)	Categorical	MC3, ALC-0315, SM102, C12-200, CCK12 (CKK-E12), DOTAP, DODAP
IL1 mol%	Continuous	5–55%
Ionizable Lipid 2 (IL2)	Categorical	Same palette as IL1
IL2 mol%	Continuous	5–55%
Total Ionizable Lipid (TIL)	Continuous	20–67% (IL1 + IL2)
Phospholipid	Categorical	DOPE, DSPC
Phospholipid mol%	Continuous	5–50%
Cholesterol mol%	Continuous	0–58%
PEG Lipid	Categorical	DMG-PEG2000, C14-PEG, PEG-Mannose
PEG mol%	Continuous	1.0–4.0%
Constraint	Mixture	All components sum to ~100 mol%


Round-by-Round Algorithm Description
Round 1 — D-Optimal Global Coverage (n = 36)
Algorithm: D-Optimal experimental design
Goal: Identify parameter boundaries; establish initial training set with maximum information content
D-Optimal design selects the set of experimental points that maximizes the determinant of the Fisher information matrix det(XTX). This is preferred over pure random sampling or Latin Hypercube Sampling (LHS) because the formulation space contains a mixture of categorical variables (lipid identity, PEG type) and mixture-constrained continuous variables (mol% sum ≈ 100%), which LHS does not handle efficiently.
Key outputs from Round 1:
SM102 identified as core active ionizable lipid
DOPE identified as superior phospholipid vs. DSPC
Initial signal: SM102+SM102 (R1-17) and ALC-0315+SM102 (R1-26) show highest normalized efficiency
Design tool: manual D-optimal grid construction
Constraints enforced: component sum ≈ 100 mol%
Coverage: all 7 IL types represented in pairwise combinations


Round 2 — Semi-Focused Space Expansion (n = 40–45)
Algorithm: Directed combinatorial expansion (human-in-the-loop)
Goal: Introduce new chemical components; explore concentration gradients systematically
Round 2 is not a local perturbation of Round 1. It introduces previously untested components:
New ionizable lipid: CCK12 (CKK-E12)
New PEG variants: PEG-Mannose, C14-PEG
Concentration gradients systematically swept:
DOPE: 5%, 15%, 25%, 35%, 50%
Cholesterol: 0%, 12.5%, 19.25%, 25.5%, 32%, 38.5%
Total IL ratio: 20–67%
Key outputs from Round 2:
CCK12+SM102 and C12-200+SM102 confirmed as dominant IL combinations
DOPE 15% identified as a "sweet spot" (mean normalized efficiency 0.871 vs. 0.148 at 5%)
DMG-PEG2000 confirmed as optimal PEG type
Top 8 formulations all have PDI ≤ 0.4 and particle size ≤ 200 nm

Round 3 — TabPFN Active Learning, Modular Optimization (n = 26)
Algorithm: TabPFN surrogate model + UCB acquisition function
Training data: R1 + R2 combined (n = 81)
Why TabPFN?
TabPFN (Tabular Prior-Data Fitted Networks) is a transformer-based model pre-trained on synthetic tabular datasets, enabling strong few-shot regression without hyperparameter tuning. At n ≈ 81 training points across ~10 features, TabPFN achieves R² ≈ 0.65 and Spearman ρ ≈ 0.81 — outperforming CatBoost and Random Forest at this sample size.
Active Learning Loop
1. Train TabPFN on all available data (R1 + R2)
2. Generate candidate pool (~3,000 formulations, grid over parameter space)
3. Predict mean and uncertainty for each candidate
4. Score candidates by UCB = μ_predicted + β × σ_predicted  (β = 0.5)
5. Select top candidates, organize into functional modules
6. Run experiments, add results to training set → repeat

The UCB acquisition function balances exploitation (high predicted efficiency) with exploration (high model uncertainty), preventing premature convergence to a local optimum.
Round 3 Module Structure
Module	n	Focus	Rationale
A: CCK12+SM102	8	IL ratio gradient × 2 DOPE concentrations	Systematic optimization of top combo
B: C12-200+SM102	6	Dense sampling near R2-19 and R2-28	Exploit two high-scoring (>0.98) anchors
C: SM102+ALC-0315	5	Ratio exploration	R2-22 (0.955) shows unexploited potential
D: SM102 high-conc	4	Validation of high-TIL regime	R2-41 result (1.0) needs confirmation
E: Replicates	3	Reproduce R2-17, R2-28, R2-22	Reproducibility required for publication

Key outputs from Round 3:
R3-11 (C12-200+SM102, norm = 1.00) confirmed as all-time best DC formulation
DOPE 15% effect validated across multiple IL combinations
C12-200+SM102 family robustly outperforms at Chol 28–38.5%, DOPE 35%

Round 4 — RSM Exploitation + D-Optimal Exploration + Validation (n = 30)
Algorithm: Three-block structured design (A: RSM / B: D-Optimal / C: Validate)
Training data: R1 + R2 + R3 combined (n ≈ 107), augmented with lance external dataset (n = 3,573, same DC cell line) via RankTransfer
Block Structure
Block A — Exploit (n = 12): Face-centered RSM
Response Surface Methodology sweep around the champion formulation (C12-200+SM102):
IL1:IL2 ratio sweep: 70:30 → 30:70 (5 points)
DOPE mol% sweep: 10%, 15%, 20%, 40%
Cholesterol fine sweep: 29.5%, 38.5%, 45%
CKK-E12/SM102 mirror formulation
Block B — Explore (n = 12): D-Optimal / Maximin space-filling
Greedy maximin algorithm seeded with two high-performing lance dataset regimes:
Regime i: high cholesterol (~47%), low PEG (~1%)
Regime ii: high phospholipid (~50%), low cholesterol (~12.5%)
Each new point is chosen to maximize the minimum distance to all existing points in the normalized feature space, ensuring maximum information gain in undersampled regions.
Block C — Validate (n = 6): Reproducibility and transfer
Replicates of R3-11 (best DC) and R3-09 (2nd best)
Transfer of lance global champion onto local lipid palette
Best CKK-E12 arm replicate
One boundary/extrapolation probe (model-disagreement point)
Key Finding from Round 4
DC2.4 and HaCaT cells show divergent responses to C12-200:SM102 ratio (Spearman ρ = 0.199, p = 0.30):
DC2.4 favors high C12-200 fraction (70:30 ratio)
HaCaT favors high SM102 fraction (30:70 ratio)
This demonstrates a cell-type-selective formulation window — a key mechanistic finding supporting DC-targeted LNP design.

Software Implementation
Installation
pip install tabpfn scikit-learn pandas numpy openpyxl scipy

For GPU acceleration (optional):
pip install tabpfn[gpu]

Required Python Version
Python ≥ 3.8 recommended. Tested on 3.9 and 3.10.
Core Dependencies
Package	Version	Purpose
tabpfn	≥ 0.1.9	Surrogate regression model
scikit-learn	≥ 1.0	Label encoding, cross-validation
pandas	≥ 1.3	Data loading and manipulation
numpy	≥ 1.21	Numerical operations, UCB scoring
openpyxl	≥ 3.0	Excel file I/O
scipy	≥ 1.7	Spearman correlation, statistical tests


Feature Engineering
Each formulation is encoded as a 13-dimensional feature vector:
feature_cols = [
    'IL1_encoded',              # LabelEncoded categorical
    'IL1_Mol_Percent',          # continuous
    'IL2_encoded',              # LabelEncoded categorical
    'IL2_Mol_Percent',          # continuous
    'Phospholipid_encoded',     # LabelEncoded categorical
    'Phospholipid_Mol_Percent', # continuous
    'Cholesterol_Mol_Percent',  # continuous
    'PEG_encoded',              # LabelEncoded categorical
    'PEG_Mol_Percent',          # continuous
    'Total_IL_Mol_Ratio',       # continuous (IL1% + IL2%)
    'IL_ratio',                 # IL1% / (IL2% + 0.01)
    'DOPE_flag',                # binary (1 if DOPE, 0 if DSPC)
    'Chol_DOPE_interact',       # Cholesterol_Mol_Percent × DOPE_flag
]

Label encoders are fit on the full lipid/PEG palette (all 7 ILs, 2 phospholipids, 3 PEG types) before splitting train/test, ensuring consistent encoding across rounds.

TabPFN Training
from tabpfn import TabPFNRegressor

model = TabPFNRegressor(device='cpu', n_estimators=16)
model.fit(X_train, y_train)  # y = normalized DC transfection efficiency

Notes:
TabPFN requires n_samples ≤ 1024 and n_features ≤ 100; both constraints are satisfied here
No hyperparameter tuning needed — this is the primary advantage for small-n settings
Use device='cuda' if a GPU is available for faster inference over large candidate pools

UCB Active Learning Selection
def ucb_select(model, candidates_df, n_select=15, beta=0.5):
    X_cand = prepare_features(candidates_df)
    
    # Estimate uncertainty via ensemble predictions
    pred_list = [model.predict(X_cand) for _ in range(10)]
    pred_mean = np.mean(pred_list, axis=0)
    pred_std  = np.std(pred_list, axis=0)
    
    # UCB score
    ucb = pred_mean + beta * pred_std
    
    candidates_df['pred_mean'] = pred_mean
    candidates_df['pred_std']  = pred_std
    candidates_df['ucb_score'] = ucb
    
    return candidates_df.nlargest(n_select, 'ucb_score')

Parameter guidance:
beta = 0.5 balances exploitation vs. exploration (increase to explore more aggressively)
n_estimators = 10 for uncertainty estimation is sufficient; increase for stability

Candidate Pool Generation
def generate_candidates():
    """
    Generate ~3,000 candidate formulations satisfying:
    - Component sum = 100 mol% (±2%)
    - PEG: 1.0–4.0%
    - Total IL: 20–67%
    - Phospholipid: 5–50%
    - Cholesterol: 0–58%
    """
    lipid_combos = [('CCK12','SM102'), ('C12-200','SM102'),
                    ('SM102','ALC-0315'), ('C12-200','CCK12')]
    
    il_ratios       = [(28,12),(24,16),(20,20),(16,24),(12,28),(8,17),(10,20)]
    phospho_configs = [('DOPE',15), ('DOPE',25), ('DOPE',35)]
    chol_levels     = [19.25, 25.5, 32.0, 38.5]
    peg_configs     = [('DMG-PEG2000',1.5), ('DMG-PEG2000',2.5), ('PEG-Mannose',1.5)]
    
    candidates = []
    for il1, il2 in lipid_combos:
        for il1p, il2p in il_ratios:
            for pl, plp in phospho_configs:
                for chol in chol_levels:
                    for peg, pegp in peg_configs:
                        total = il1p + il2p + plp + chol + pegp
                        if 98 <= total <= 102:
                            candidates.append({...})
    return pd.DataFrame(candidates)


Inter-Plate Normalization (Critical)
Raw RLU values cannot be compared directly across rounds due to inter-plate variability. Normalize within each plate before cross-round analysis:
# Within each experimental plate:
df['log_TE'] = np.log10(df['DC_Transfection_Efficiency'].clip(lower=1))
plate_max     = df['log_TE'].max()
plate_min     = df['log_TE'].min()  # theoretical minimum (log of lowest observed)
df['normalized_TE'] = (df['log_TE'] - plate_min) / (plate_max - plate_min)

Evidence for necessity: Replicates of R3-11 (raw RLU ~156,000 in Round 3) measured ~5,300 in Round 4 — a 30× difference attributable entirely to inter-plate variation. All cross-round comparisons in this study use normalized values.

Reproducing the Full Pipeline
Step 1: Install dependencies
        pip install tabpfn scikit-learn pandas numpy openpyxl scipy

Step 2: Load and merge data from all rounds
        → Use All-100-DATA-in-same-log-value.xlsx (R1–R3)
        → Use 4rd-round-LNP_Design_Batch_30.xlsx (R4)

Step 3: Apply inter-plate normalization (see above)

Step 4: Run feature engineering (LabelEncode categoricals, add interaction terms)

Step 5: Train TabPFN on all available normalized data

Step 6: Generate candidate pool (~3,000 formulations)

Step 7: Run UCB active learning selection (beta=0.5, n_select=15–30)

Step 8: Organize selected candidates into Exploit / Explore / Validate blocks
        → ~50–60% Exploit (RSM around current champion)
        → ~30–40% Explore (D-optimal / maximin space-filling)
        → ~10–15% Validate (replicates of top candidates)

Step 9: Run experiments, collect data, return to Step 2


Data Files
File	Contents	Rounds
All-100-DATA-in-same-log-value.xlsx	All formulations + DC transfection + physicochemical data	R1, R2, R3
4rd-round-LNP_Design_Batch_30.xlsx	30 formulations with DC + HaCaT readouts + design rationale	R4
Fu-Ben-LNP_Round3_Shi-Yan-She-Ji-_TabPFNCe-Lue-_5.xlsx	Round 3 design rationale, TabPFN code template, weighing calculations	R3 design
Round4_Recommendations.csv	Computer-generated Round 4 candidate list	R4 pre-experiment


Experimental Constraints (Wet-Lab Ready Formulations Only)
All algorithmically generated candidates are filtered to ensure wet-lab feasibility:
Component sum: 98–102 mol%
PEG lipid: 1.0–4.0 mol%
Total ionizable lipid: 20–67 mol%
Phospholipid: 5–50 mol%
Cholesterol: 0–58 mol%
Particle size target: < 200 nm (PDI < 0.4 preferred)
Lipid palette restricted to reagents available in-house

Citation
If using this pipeline, please cite the original LNP dataset (lance, n = 3,573) used as prior knowledge for RankTransfer and surrogate model initialization, and the TabPFN paper:
Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. ICLR 2023.
