#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical information-value + R4 stopping audit for sequential LNP design.

DIRECT-RUN PURPOSE
------------------
1) Read the SAME R1-R4 workbook/path used by the publication modeling script.
2) Reconstruct prior-round-only information value for every historical R2/R3/R4 point.
3) Score R5 candidates using R1-R4 ONLY. R5 is NEVER added to training in this script.
4) Quantify whether R4 is a defensible model-freeze / stopping point using:
   - remaining expected improvement (EI) in a fixed virtual feasible pool,
   - predictive uncertainty / information gain,
   - formulation-space coverage,
   - empirical repeatability from exact duplicate formulations,
   - and, when available, the existing cumulative grouped nested-CV summary.

RESEARCH-INTEGRITY RULE
-----------------------
Historical R2/R3/R4 scores are retrospective, prior-round-only audits.
They must not be described as the exact historical acquisition scores unless the
contemporaneous candidate-ranking outputs are recovered.

The R5 section is prospective-only:
    train = R1+R2+R3+R4
    score = R5
    R5 outcomes are NOT used for fitting, feature encoding, GP optimization,
    or stopping analysis.

Dependencies
------------
numpy, pandas, scipy, scikit-learn, openpyxl, matplotlib

Typical PyCharm use
-------------------
Just right-click this file -> Run.
No command-line arguments are required on the author's workstation.

Optional explicit run:
    python historical_information_value_audit_R4_STOP.py ^
      --input "C:\\path\\R1-4 all LNP normalized 1.35 (new).xlsx"
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# 0. DIRECT-RUN PATHS — SAME PATH AS THE PUBLICATION MODELING SCRIPT
# =============================================================================

FORCED_DATA_PATH = Path(
    r"C:\Users\ASUS\Desktop\AI screen LNP python and excel\PyCharm 有效代码\8.02 publish\R1-4 all LNP normalized 1.35 (new).xlsx"
)

FORCED_SHEET_NAME = "Round1&2&3&4"
TARGET_COLUMN = "Normalized for DC2.4"

RANDOM_STATE = 4930
KAPPA_UCB = 0.5
XI_EI = 0.01

# Same primary QC rule as the publication pipeline.
PDI_MAX = 0.5
SIZE_MIN = 30.0
SIZE_MAX = 300.0

# Virtual feasible pool used only for the post-R4 saturation / stopping audit.
VIRTUAL_POOL_N = 4000

# If the existing cumulative tree-model output is present, this script reads it.
CUMULATIVE_FILENAMES = (
    "cumulative_round_primary_ensemble_summary.csv",
    "CUMULATIVE_R1_R4_PRIMARY_ENSEMBLE.csv",
)


# =============================================================================
# 1. COLUMN ALIASES
# =============================================================================

COLUMN_ALIASES: Dict[str, List[str]] = {
    "candidate_id": [
        "candidate_id", "Candidate_ID", "Selection_Order", "Formulation_ID",
        "Sample", "ID", "No", "NO", "no", "编号", "配方编号", 0,
    ],
    "IL1": [
        "Ionizable_Lipid_1", "IL1", "ionizable lipid 1",
        "可离子化脂质1", "离子化脂质1",
    ],
    "IL2": [
        "Ionizable_Lipid_2", "IL2", "ionizable lipid 2",
        "可离子化脂质2", "离子化脂质2",
    ],
    "IL1_pct": ["IL1_Mol_Percent", "IL1_molpct", "IL1 mol%", "IL1 mol pct"],
    "IL2_pct": ["IL2_Mol_Percent", "IL2_molpct", "IL2 mol%", "IL2 mol pct"],
    "PL": ["Phospholipid", "HL", "Helper_Lipid", "磷脂"],
    "PL_pct": [
        "Phospholipid_Mol_Percent", "HL_molpct",
        "Helper_Lipid_Mol_Percent", "磷脂比例",
    ],
    "Chol_pct": [
        "Cholesterol_Mol_Percent", "CHOL_molpct", "Cholesterol", "胆固醇比例",
    ],
    "PEG": ["PEG类型", "PEG", "PEG_Lipid", "PEG lipid"],
    "PEG_pct": ["PEG_Mol_Percent", "PEG_molpct", "PEG mol%", "PEG比例"],
    "PDI": ["PDI"],
    "Size_nm": ["Particle_Size (nm)", "Particle_Size", "Size", "size", "粒径"],
    "Target": [TARGET_COLUMN],
}


LIPID_ALIASES = {
    "CCK12": "CKK-E12",
    "CKKE12": "CKK-E12",
    "CKK E12": "CKK-E12",
    "CKK-E12": "CKK-E12",
    "ALC0315": "ALC-0315",
    "ALC 0315": "ALC-0315",
    "ALC-0315": "ALC-0315",
    "SM-102": "SM102",
    "SM 102": "SM102",
    "SM102": "SM102",
    "C12200": "C12-200",
    "C12 200": "C12-200",
    "C12-200": "C12-200",
    "MC-3": "MC3",
    "DLIN-MC3-DMA": "MC3",
    "MC3": "MC3",
    "DOTAP": "DOTAP",
    "DODAP": "DODAP",
    "DOPE": "DOPE",
    "DSPC": "DSPC",
    "DMG-PEG": "DMG-PEG2000",
    "DMG-PEG-2000": "DMG-PEG2000",
    "DMGPEG2000": "DMG-PEG2000",
    "PEG2000-DMG": "DMG-PEG2000",
    "DMG-PEG2000": "DMG-PEG2000",
    "ALC-0159": "C14-PEG",
    "ALC0159": "C14-PEG",
    "C14PEG": "C14-PEG",
    "C14-PEG": "C14-PEG",
    "PEG-MANNOSE": "PEG-Mannose",
    "MANNOSE-PEG": "PEG-Mannose",
    "PEG MANNOSE": "PEG-Mannose",
    "PEG-Mannose": "PEG-Mannose",
}


FEATURE_NAMES = [
    "IL1", "IL1_pct",
    "IL2", "IL2_pct",
    "PL", "PL_pct",
    "Chol_pct",
    "PEG", "PEG_pct",
    "TotalIL_pct",
]

CATEGORICAL = ["IL1", "IL2", "PL", "PEG"]
NUMERIC = [
    "IL1_pct", "IL2_pct", "PL_pct",
    "Chol_pct", "PEG_pct", "TotalIL_pct",
]


# =============================================================================
# 2. BASIC UTILITIES
# =============================================================================

def normalize_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NONE"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "-", "无"}:
        return "NONE"

    candidates = [text]
    base = re.sub(r"\([^)]*\)", "", text).strip()
    if base and base != text:
        candidates.append(base)
    candidates.extend(x.strip() for x in re.findall(r"\(([^)]*)\)", text) if x.strip())

    for candidate in candidates:
        key = candidate.upper().replace("_", "-").replace("  ", " ").strip()
        if candidate in LIPID_ALIASES:
            return LIPID_ALIASES[candidate]
        if key in LIPID_ALIASES:
            return LIPID_ALIASES[key]
    return text


def find_column(df: pd.DataFrame, aliases: Sequence[Any]) -> Optional[Any]:
    normalized = {str(c).strip().casefold(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        key = str(alias).strip().casefold()
        if key in normalized:
            return normalized[key]
    return None


def infer_round(raw_id: Any) -> Optional[str]:
    """
    Highest round number wins for labels such as R4-25 (R3-11).
    Numeric 1..36 are R1.
    """
    if raw_id is None or (isinstance(raw_id, float) and np.isnan(raw_id)):
        return None

    text = str(raw_id).strip()

    # R1 historical numeric IDs.
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        value = int(float(text))
        if 1 <= value <= 36:
            return "R1"

    nums = [int(x) for x in re.findall(r"R\s*(\d+)", text, flags=re.I)]
    if nums:
        return f"R{max(nums)}"

    return None


def round_number(label: str) -> int:
    match = re.search(r"R\s*(\d+)", str(label), re.I)
    return int(match.group(1)) if match else 999


def resolve_sheet_name(path: Path, requested: str) -> str:
    xls = pd.ExcelFile(path)
    sheets = list(xls.sheet_names)

    if requested in sheets:
        return requested

    def compact(x: Any) -> str:
        return re.sub(r"\s+", "", str(x)).casefold()

    req = compact(requested)
    matches = [s for s in sheets if compact(s) == req]
    if len(matches) == 1:
        print(f"[Sheet] Auto-matched {requested!r} -> {matches[0]!r}")
        return matches[0]

    # Ignore a trailing "(2)" etc.
    def base(x: Any) -> str:
        return re.sub(r"\(\d+\)$", "", compact(x))

    req_base = base(requested)
    matches = [s for s in sheets if base(s) == req_base]
    if len(matches) == 1:
        print(f"[Sheet] Auto-matched sheet family {requested!r} -> {matches[0]!r}")
        return matches[0]

    raise ValueError(
        f"Cannot uniquely resolve sheet {requested!r}. Existing sheets: {sheets}"
    )


def discover_data_path(user_path: str = "") -> Path:
    if user_path:
        p = Path(os.path.expanduser(os.path.expandvars(user_path.strip().strip('"'))))
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"--input workbook not found: {p}")

    # 1) Same forced path as the working publication pipeline.
    if FORCED_DATA_PATH.is_file():
        return FORCED_DATA_PATH.resolve()

    # 2) Search next to this script.
    script_dir = Path(__file__).resolve().parent
    names = [
        "R1-4 all LNP normalized 1.35 (new).xlsx",
        "R1-4 all LNP normalized 1.35 (new) .xlsx",
    ]
    for name in names:
        candidate = script_dir / name
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the training workbook.\n"
        f"Expected direct-run path:\n  {FORCED_DATA_PATH}\n"
        "Or place the workbook beside this Python file, or pass --input explicitly."
    )


# =============================================================================
# 3. LOAD DESIGN DATA AND TRAINING DATA
# =============================================================================

def load_all_design_rows(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load R1-R5 formulation rows by column name.

    IMPORTANT:
    TotalIL_pct is calculated from IL1_pct + IL2_pct.
    We do NOT read the historical 'Total_Ionizable_Lipid_Mol_Ratio' column as
    a percentage because some later rounds store strings such as '70:30' there.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name)

    cols = {}
    for standard, aliases in COLUMN_ALIASES.items():
        col = find_column(raw, aliases)
        cols[standard] = col

    required = [
        "candidate_id", "IL1", "IL1_pct", "IL2", "IL2_pct",
        "PL", "PL_pct", "Chol_pct", "PEG", "PEG_pct", "Target",
    ]
    missing = [x for x in required if cols.get(x) is None]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Actual workbook columns: {list(raw.columns)}"
        )

    rows: List[Dict[str, Any]] = []

    for _, row in raw.iterrows():
        rid = row[cols["candidate_id"]]
        rnd = infer_round(rid)
        if rnd is None or rnd not in {"R1", "R2", "R3", "R4", "R5"}:
            continue

        rec = {
            "Source_row": int(_),
            "ID": str(rid).strip(),
            "Round": rnd,
            "IL1": normalize_name(row[cols["IL1"]]),
            "IL1_pct": pd.to_numeric(pd.Series([row[cols["IL1_pct"]]]), errors="coerce").iloc[0],
            "IL2": normalize_name(row[cols["IL2"]]),
            "IL2_pct": pd.to_numeric(pd.Series([row[cols["IL2_pct"]]]), errors="coerce").iloc[0],
            "PL": normalize_name(row[cols["PL"]]),
            "PL_pct": pd.to_numeric(pd.Series([row[cols["PL_pct"]]]), errors="coerce").iloc[0],
            "Chol_pct": pd.to_numeric(pd.Series([row[cols["Chol_pct"]]]), errors="coerce").iloc[0],
            "PEG": normalize_name(row[cols["PEG"]]),
            "PEG_pct": pd.to_numeric(pd.Series([row[cols["PEG_pct"]]]), errors="coerce").iloc[0],
            "PDI": (
                pd.to_numeric(pd.Series([row[cols["PDI"]]]), errors="coerce").iloc[0]
                if cols.get("PDI") is not None else np.nan
            ),
            "Size_nm": (
                pd.to_numeric(pd.Series([row[cols["Size_nm"]]]), errors="coerce").iloc[0]
                if cols.get("Size_nm") is not None else np.nan
            ),
            "Target": pd.to_numeric(pd.Series([row[cols["Target"]]]), errors="coerce").iloc[0],
        }

        # Single-ionizable-lipid R3 formulations can have blank IL2 / IL2%.
        if rec["IL2"] == "NONE" and not np.isfinite(rec["IL2_pct"]):
            rec["IL2_pct"] = 0.0

        rec["TotalIL_pct"] = (
            float(rec["IL1_pct"]) + float(rec["IL2_pct"])
            if np.isfinite(rec["IL1_pct"]) and np.isfinite(rec["IL2_pct"])
            else np.nan
        )

        pct_sum = (
            rec["IL1_pct"] + rec["IL2_pct"] + rec["PL_pct"]
            + rec["Chol_pct"] + rec["PEG_pct"]
            if all(np.isfinite(rec[x]) for x in [
                "IL1_pct", "IL2_pct", "PL_pct", "Chol_pct", "PEG_pct"
            ])
            else np.nan
        )
        rec["Molar_sum"] = pct_sum

        composition_ok = (
            rec["IL1"] != "NONE"
            and rec["PL"] != "NONE"
            and rec["PEG"] != "NONE"
            and all(np.isfinite(rec[x]) for x in NUMERIC)
            and np.isfinite(pct_sum)
            and 80.0 <= pct_sum <= 120.0
        )
        rec["composition_ok"] = bool(composition_ok)

        qc_reasons = []
        if not np.isfinite(rec["PDI"]):
            qc_reasons.append("PDI_missing")
        elif rec["PDI"] > PDI_MAX:
            qc_reasons.append(f"PDI>{PDI_MAX}")

        if not np.isfinite(rec["Size_nm"]):
            qc_reasons.append("size_missing")
        elif not (SIZE_MIN <= rec["Size_nm"] <= SIZE_MAX):
            qc_reasons.append(f"size_outside_{SIZE_MIN}_{SIZE_MAX}nm")

        rec["QC_pass"] = len(qc_reasons) == 0
        rec["QC_reason"] = ";".join(qc_reasons)
        rows.append(rec)

    out = pd.DataFrame(rows)

    # Historical acquisition batch = the first contiguous block for that round.
    # This prevents late reference/calibration rows (e.g. R3-labelled anchors appended
    # after the R5 block) from being misinterpreted as candidates that originally
    # selected the historical R3 experiment.
    first_pos = {}
    for rnd in ["R1", "R2", "R3", "R4", "R5"]:
        sub = out.loc[out["Round"].eq(rnd), "Source_row"]
        if len(sub):
            first_pos[rnd] = int(sub.min())

    next_round = {"R1": "R2", "R2": "R3", "R3": "R4", "R4": "R5"}
    out["historical_batch"] = True
    for rnd, nxt in next_round.items():
        if nxt in first_pos:
            cutoff = first_pos[nxt]
            mask = out["Round"].eq(rnd)
            out.loc[mask, "historical_batch"] = out.loc[mask, "Source_row"] < cutoff

    out = out.sort_values("Source_row").reset_index(drop=True)

    print("\n[All round-labelled rows]")
    print(out.groupby("Round").size().to_string())

    print("\n[Original historical acquisition blocks: composition-valid rows]")
    hist = out[out["historical_batch"] & out["composition_ok"]]
    print(hist.groupby("Round").size().to_string())

    return out


def make_training_rows(design_df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary model-development training rows:
    - R1-R4 only
    - target present
    - composition valid
    - PDI/size QC passed
    """
    train = design_df[
        design_df["Round"].isin(["R1", "R2", "R3", "R4"])
        & design_df["composition_ok"]
        & design_df["QC_pass"]
        & design_df["Target"].notna()
    ].copy().reset_index(drop=True)

    print("\n[QC-passed R1-R4 training rows]")
    print(train.groupby("Round").size().to_string())
    print(f"TOTAL = {len(train)}")

    return train


# =============================================================================
# 4. FEATURE SPACE
# =============================================================================

def fixed_categories(design_df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        col: sorted(design_df.loc[design_df["composition_ok"], col].astype(str).unique().tolist())
        for col in CATEGORICAL
    }


def make_preprocessor(categories_map: Dict[str, List[str]]) -> ColumnTransformer:
    categories = [categories_map[c] for c in CATEGORICAL]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    categories=categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_NAMES].copy()
    for c in CATEGORICAL:
        out[c] = out[c].fillna("NONE").astype(str)
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


# =============================================================================
# 5. GP / ACQUISITION FUNCTIONS
# =============================================================================

def find_white_noise_level(kernel: Any) -> Optional[float]:
    if isinstance(kernel, WhiteKernel):
        return float(kernel.noise_level)
    for attr in ("k1", "k2"):
        if hasattr(kernel, attr):
            value = find_white_noise_level(getattr(kernel, attr))
            if value is not None:
                return value
    return None


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float = XI_EI,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    safe_sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best - xi
    z = improvement / safe_sigma
    ei = improvement * norm.cdf(z) + safe_sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return np.maximum(ei, 0.0)


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if len(x) <= 1:
        return np.full(len(x), 100.0)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x))
    return 100.0 * ranks / (len(x) - 1)


def fit_gp_and_score(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    categories_map: Dict[str, List[str]],
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(train_df) < 8:
        raise ValueError("Too few prior-round training rows for GP audit.")
    if len(candidate_df) == 0:
        return pd.DataFrame(), {}

    pre = make_preprocessor(categories_map)
    X_train = pre.fit_transform(feature_frame(train_df))
    X_cand = pre.transform(feature_frame(candidate_df))
    y_train = train_df["Target"].to_numpy(dtype=float)

    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * Matern(
            length_scale=1.5,
            length_scale_bounds=(0.2, 10.0),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=0.05,
            noise_level_bounds=(1e-4, 0.5),
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )
    gp.fit(X_train, y_train)

    mu, predictive_std = gp.predict(X_cand, return_std=True)

    y_scale = float(getattr(gp, "_y_train_std", np.std(y_train, ddof=0)))
    fitted_white = find_white_noise_level(gp.kernel_)
    if fitted_white is None:
        fitted_white = 1e-3

    observation_noise_var = max(float(fitted_white) * (y_scale ** 2), 1e-12)

    # sklearn's predictive std includes the WhiteKernel contribution.
    latent_var = np.maximum(
        predictive_std ** 2 - observation_noise_var,
        1e-12,
    )
    latent_std = np.sqrt(latent_var)

    best_prior = float(np.max(y_train))
    ei = expected_improvement(mu, latent_std, best=best_prior, xi=XI_EI)
    ucb = mu + KAPPA_UCB * latent_std
    info_gain = 0.5 * np.log1p(latent_var / observation_noise_var)

    # Euclidean novelty in the same transformed feature space.
    # Batch in chunks to avoid large temporary arrays for virtual pools.
    novelty = np.empty(len(X_cand), dtype=float)
    chunk = 500
    for start in range(0, len(X_cand), chunk):
        stop = min(start + chunk, len(X_cand))
        delta = X_cand[start:stop, None, :] - X_train[None, :, :]
        novelty[start:stop] = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)

    result = candidate_df.copy().reset_index(drop=True)
    result["Prior_n"] = len(train_df)
    result["Best_prior_target"] = best_prior
    result["GP_mean"] = mu
    result["GP_latent_std"] = latent_std
    result["UCB_kappa_0.5"] = ucb
    result["Expected_Improvement"] = ei
    result["Expected_Information_Gain"] = info_gain
    result["Novelty_NN_Distance"] = novelty

    for source, name in [
        ("GP_latent_std", "Uncertainty"),
        ("UCB_kappa_0.5", "UCB"),
        ("Expected_Improvement", "EI"),
        ("Expected_Information_Gain", "IG"),
        ("Novelty_NN_Distance", "Novelty"),
    ]:
        result[f"{name}_percentile_within_batch"] = percentile_rank(result[source])

    metadata = {
        "Train_n": len(train_df),
        "Candidate_n": len(candidate_df),
        "Best_prior_target": best_prior,
        "GP_kernel": str(gp.kernel_),
        "Observation_noise_variance": observation_noise_var,
        "Target_SD_prior": float(np.std(y_train, ddof=1)),
    }

    return result, metadata


# =============================================================================
# 6. HISTORICAL POINT AUDIT + R5 PROSPECTIVE-ONLY SCORING
# =============================================================================

def historical_point_audit(
    design_df: pd.DataFrame,
    training_df: pd.DataFrame,
    categories_map: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    round_order = ["R1", "R2", "R3", "R4"]
    point_frames = []
    round_rows = []

    for target_round in ["R2", "R3", "R4"]:
        idx = round_order.index(target_round)
        prior_rounds = round_order[:idx]

        train = training_df[training_df["Round"].isin(prior_rounds)].copy()
        candidates = design_df[
            (design_df["Round"] == target_round)
            & design_df["historical_batch"]
            & design_df["composition_ok"]
        ].copy()

        result, meta = fit_gp_and_score(
            train,
            candidates,
            categories_map,
            random_state=RANDOM_STATE + idx,
        )
        result["Audit_type"] = "retrospective_prior_round_only"
        result["Training_rounds"] = "+".join(prior_rounds)
        point_frames.append(result)

        round_rows.append({
            "Round_scored": target_round,
            "Training_rounds": "+".join(prior_rounds),
            **meta,
            "Mean_uncertainty": result["GP_latent_std"].mean(),
            "Median_uncertainty": result["GP_latent_std"].median(),
            "Mean_EI": result["Expected_Improvement"].mean(),
            "Median_EI": result["Expected_Improvement"].median(),
            "Mean_IG": result["Expected_Information_Gain"].mean(),
            "Median_IG": result["Expected_Information_Gain"].median(),
            "Mean_novelty": result["Novelty_NN_Distance"].mean(),
            "Median_novelty": result["Novelty_NN_Distance"].median(),
        })

    # R5 is strictly held out from training.
    r5 = design_df[
        (design_df["Round"] == "R5")
        & design_df["composition_ok"]
    ].copy()

    if len(r5):
        r5_result, r5_meta = fit_gp_and_score(
            training_df[training_df["Round"].isin(["R1", "R2", "R3", "R4"])],
            r5,
            categories_map,
            random_state=RANDOM_STATE + 50,
        )
        r5_result["Audit_type"] = "prospective_candidate_scoring_only"
        r5_result["Training_rounds"] = "R1+R2+R3+R4"
        r5_result["Validation_role"] = "R5_HELD_OUT_NOT_USED_FOR_TRAINING"
        point_frames.append(r5_result)

        round_rows.append({
            "Round_scored": "R5",
            "Training_rounds": "R1+R2+R3+R4",
            **r5_meta,
            "Mean_uncertainty": r5_result["GP_latent_std"].mean(),
            "Median_uncertainty": r5_result["GP_latent_std"].median(),
            "Mean_EI": r5_result["Expected_Improvement"].mean(),
            "Median_EI": r5_result["Expected_Improvement"].median(),
            "Mean_IG": r5_result["Expected_Information_Gain"].mean(),
            "Median_IG": r5_result["Expected_Information_Gain"].median(),
            "Mean_novelty": r5_result["Novelty_NN_Distance"].mean(),
            "Median_novelty": r5_result["Novelty_NN_Distance"].median(),
        })

    points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    rounds = pd.DataFrame(round_rows)

    return points, rounds


# =============================================================================
# 7. EXACT-FORMULATION REPEATABILITY / EMPIRICAL NOISE CEILING
# =============================================================================

def formulation_signature(row: pd.Series, decimals: int = 4) -> str:
    pairs = [
        (normalize_name(row["IL1"]), round(float(row["IL1_pct"]), decimals)),
        (normalize_name(row["IL2"]), round(float(row["IL2_pct"]), decimals)),
    ]
    pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    return "|".join([
        f"IL:{pairs[0][0]}:{pairs[0][1]:.{decimals}f}",
        f"IL:{pairs[1][0]}:{pairs[1][1]:.{decimals}f}",
        f"PL:{normalize_name(row['PL'])}:{round(float(row['PL_pct']), decimals):.{decimals}f}",
        f"CHOL:{round(float(row['Chol_pct']), decimals):.{decimals}f}",
        f"PEG:{normalize_name(row['PEG'])}:{round(float(row['PEG_pct']), decimals):.{decimals}f}",
    ])


def repeatability_audit(training_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = training_df.copy()
    df["formulation_signature"] = df.apply(formulation_signature, axis=1)

    counts = df["formulation_signature"].value_counts()
    dup_sigs = counts[counts > 1].index.tolist()

    duplicate_rows = df[df["formulation_signature"].isin(dup_sigs)].copy()

    group_rows = []
    for sig, g in duplicate_rows.groupby("formulation_signature"):
        group_rows.append({
            "formulation_signature": sig,
            "n": len(g),
            "IDs": " | ".join(g["ID"].astype(str)),
            "Rounds": " | ".join(g["Round"].astype(str)),
            "Mean_target": g["Target"].mean(),
            "SD_target": g["Target"].std(ddof=1) if len(g) > 1 else np.nan,
            "Min_target": g["Target"].min(),
            "Max_target": g["Target"].max(),
            "Range_target": g["Target"].max() - g["Target"].min(),
        })
    group_table = pd.DataFrame(group_rows)

    if len(group_table) < 2:
        summary = pd.DataFrame([{
            "n_duplicate_groups": len(group_table),
            "n_duplicate_rows": len(duplicate_rows),
            "within_group_variance": np.nan,
            "between_group_mean_variance": np.nan,
            "estimated_signal_variance": np.nan,
            "empirical_repeatability": np.nan,
            "interpretation": "Too few duplicate formulation groups for a repeatability estimate.",
        }])
        return group_table, summary

    within_sse = 0.0
    within_df = 0
    means = []
    ns = []

    for _, g in duplicate_rows.groupby("formulation_signature"):
        means.append(float(g["Target"].mean()))
        ns.append(len(g))
        within_sse += float(((g["Target"] - g["Target"].mean()) ** 2).sum())
        within_df += len(g) - 1

    within_var = within_sse / within_df if within_df > 0 else np.nan
    between_mean_var = float(np.var(means, ddof=1))
    mean_rep = float(np.mean(ns))

    # Approximate variance decomposition:
    # Var(group means) ~= signal variance + within variance / n_rep
    signal_var = max(between_mean_var - within_var / max(mean_rep, 1.0), 0.0)
    repeatability = (
        signal_var / (signal_var + within_var)
        if (signal_var + within_var) > 0 else np.nan
    )

    summary = pd.DataFrame([{
        "n_duplicate_groups": len(group_table),
        "n_duplicate_rows": len(duplicate_rows),
        "within_group_variance": within_var,
        "between_group_mean_variance": between_mean_var,
        "estimated_signal_variance": signal_var,
        "empirical_repeatability": repeatability,
        "interpretation": (
            "Approximate repeatability/noise-ceiling diagnostic from exact-formulation repeats. "
            "This is NOT a formal universal upper bound because duplicate groups are few and "
            "can include cross-round biological/batch effects."
        ),
    }])

    return group_table, summary


# =============================================================================
# 8. FIXED VIRTUAL FEASIBLE POOL FOR SATURATION / STOPPING ANALYSIS
# =============================================================================

def generate_virtual_pool(
    design_df: pd.DataFrame,
    n: int,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate one fixed feasible pool independent of model predictions/outcomes.

    Continuous variables are sampled with Latin Hypercube Sampling within the
    experimentally observed R1-R4 ranges, then cholesterol is obtained by mixture
    closure. Categorical identities are sampled from the observed formulation palette.

    This pool is used ONLY to compare remaining acquisition opportunity after
    R1, R1+R2, R1+R2+R3 and R1-R4.
    """
    hist = design_df[
        design_df["Round"].isin(["R1", "R2", "R3", "R4"])
        & design_df["composition_ok"]
    ].copy()

    categories = fixed_categories(hist)

    total_min = max(15.0, float(hist["TotalIL_pct"].min()))
    total_max = min(70.0, float(hist["TotalIL_pct"].max()))
    pl_min = max(0.0, float(hist["PL_pct"].min()))
    pl_max = min(55.0, float(hist["PL_pct"].max()))
    peg_min = max(0.5, float(hist["PEG_pct"].min()))
    peg_max = min(5.0, float(hist["PEG_pct"].max()))
    chol_min = max(0.0, float(hist["Chol_pct"].min()))
    chol_max = min(60.0, float(hist["Chol_pct"].max()))

    frac = hist["IL1_pct"] / hist["TotalIL_pct"].replace(0, np.nan)
    frac = frac.replace([np.inf, -np.inf], np.nan).dropna()
    frac_min = max(0.05, float(frac.min()))
    frac_max = min(0.95, float(frac.max()))

    rng = np.random.default_rng(seed)

    rows = []
    batch_size = max(n * 2, 1000)
    batch_id = 0

    while len(rows) < n and batch_id < 25:
        sampler = qmc.LatinHypercube(d=4, seed=seed + batch_id)
        u = sampler.random(batch_size)

        total = total_min + u[:, 0] * (total_max - total_min)
        frac1 = frac_min + u[:, 1] * (frac_max - frac_min)
        pl = pl_min + u[:, 2] * (pl_max - pl_min)
        peg = peg_min + u[:, 3] * (peg_max - peg_min)
        chol = 100.0 - total - pl - peg

        valid = (chol >= chol_min) & (chol <= chol_max)
        idxs = np.where(valid)[0]

        for i in idxs:
            if len(rows) >= n:
                break

            il1 = rng.choice(categories["IL1"])
            il2 = rng.choice(categories["IL2"])
            # Avoid NONE as IL1. IL2 NONE is allowed if present historically.
            if il1 == "NONE":
                continue

            total_i = float(total[i])
            frac_i = float(frac1[i])

            if il2 == "NONE":
                il1_pct = total_i
                il2_pct = 0.0
            else:
                il1_pct = total_i * frac_i
                il2_pct = total_i - il1_pct

            rows.append({
                "ID": f"VIRTUAL-{len(rows)+1:05d}",
                "Round": "VIRTUAL",
                "IL1": il1,
                "IL1_pct": il1_pct,
                "IL2": il2,
                "IL2_pct": il2_pct,
                "PL": rng.choice(categories["PL"]),
                "PL_pct": float(pl[i]),
                "Chol_pct": float(chol[i]),
                "PEG": rng.choice(categories["PEG"]),
                "PEG_pct": float(peg[i]),
                "TotalIL_pct": total_i,
                "PDI": np.nan,
                "Size_nm": np.nan,
                "Target": np.nan,
                "Molar_sum": 100.0,
                "composition_ok": True,
                "QC_pass": False,
                "QC_reason": "not_synthesized",
            })

        batch_id += 1

    pool = pd.DataFrame(rows)
    if len(pool) < n:
        raise RuntimeError(f"Could generate only {len(pool)} feasible virtual candidates.")

    return pool.iloc[:n].reset_index(drop=True)


def virtual_pool_stage_audit(
    training_df: pd.DataFrame,
    pool: pd.DataFrame,
    categories_map: Dict[str, List[str]],
) -> pd.DataFrame:
    stages = [
        ("R1", ["R1"]),
        ("R1+R2", ["R1", "R2"]),
        ("R1+R2+R3", ["R1", "R2", "R3"]),
        ("R1+R2+R3+R4", ["R1", "R2", "R3", "R4"]),
    ]

    rows = []

    for i, (stage, rounds) in enumerate(stages):
        train = training_df[training_df["Round"].isin(rounds)].copy()
        scored, meta = fit_gp_and_score(
            train,
            pool,
            categories_map,
            random_state=RANDOM_STATE + 100 + i,
        )

        target_sd = max(float(meta["Target_SD_prior"]), 1e-12)

        rows.append({
            "stage": stage,
            "stage_order": i + 1,
            "n_training": len(train),
            "Best_prior_target": meta["Best_prior_target"],
            "Target_SD_prior": target_sd,
            "Mean_pool_uncertainty": scored["GP_latent_std"].mean(),
            "P95_pool_uncertainty": scored["GP_latent_std"].quantile(0.95),
            "Mean_pool_EI": scored["Expected_Improvement"].mean(),
            "P95_pool_EI": scored["Expected_Improvement"].quantile(0.95),
            "Max_pool_EI": scored["Expected_Improvement"].max(),
            "P95_EI_over_target_SD": (
                scored["Expected_Improvement"].quantile(0.95) / target_sd
            ),
            "Max_EI_over_target_SD": (
                scored["Expected_Improvement"].max() / target_sd
            ),
            "Mean_pool_IG": scored["Expected_Information_Gain"].mean(),
            "P95_pool_IG": scored["Expected_Information_Gain"].quantile(0.95),
            "Mean_pool_novelty": scored["Novelty_NN_Distance"].mean(),
            "P95_pool_novelty": scored["Novelty_NN_Distance"].quantile(0.95),
            "GP_kernel": meta["GP_kernel"],
        })

    summary = pd.DataFrame(rows)

    # Relative changes from the immediately preceding stage.
    for col in [
        "Mean_pool_uncertainty", "P95_pool_EI",
        "Max_pool_EI", "Mean_pool_IG", "Mean_pool_novelty",
    ]:
        summary[f"delta_{col}"] = summary[col].diff()
        summary[f"ratio_to_previous_{col}"] = summary[col] / summary[col].shift(1)

    return summary


# =============================================================================
# 9. OPTIONAL: READ EXISTING CUMULATIVE GROUPED NESTED-CV RESULTS
# =============================================================================

def discover_latest_cumulative_summary(data_path: Path) -> Optional[Path]:
    roots = [
        data_path.parent,
        data_path.parent.parent,
        data_path.parent.parent / "lnp_outputs",
    ]

    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for filename in CUMULATIVE_FILENAMES:
            candidates.extend(root.rglob(filename))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_cumulative_summary(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if "model" in df.columns:
        sub = df[df["model"].astype(str) == "PrimaryTreeEnsemble"].copy()
        if len(sub):
            df = sub

    if "stage_order" in df.columns:
        df = df.sort_values("stage_order")

    keep = [
        c for c in [
            "stage", "stage_order", "n_samples", "n_formulation_groups",
            "R2", "RMSE", "Spearman", "Top20_recall",
        ] if c in df.columns
    ]
    return df[keep].reset_index(drop=True)


# =============================================================================
# 10. STOPPING INTERPRETATION
# =============================================================================

def stopping_evidence(
    virtual_summary: pd.DataFrame,
    repeatability_summary: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:

    evidence = []

    final_virtual = virtual_summary.iloc[-1]
    prev_virtual = virtual_summary.iloc[-2]

    ei_ratio = (
        final_virtual["P95_pool_EI"] / prev_virtual["P95_pool_EI"]
        if prev_virtual["P95_pool_EI"] > 0 else np.nan
    )
    max_ei_sd = final_virtual["Max_EI_over_target_SD"]
    p95_ei_sd = final_virtual["P95_EI_over_target_SD"]

    evidence.append({
        "Evidence": "Remaining optimization opportunity after R4",
        "Value": f"P95 EI ratio R4-stage / R3-stage = {ei_ratio:.3f}",
        "Interpretation": (
            "A strong drop means the surrogate sees less remaining probability-weighted "
            "improvement in the defined feasible space after R4."
        ),
    })
    evidence.append({
        "Evidence": "Scale of remaining EI",
        "Value": (
            f"P95 EI / target SD = {p95_ei_sd:.4f}; "
            f"Max EI / target SD = {max_ei_sd:.4f}"
        ),
        "Interpretation": (
            "Small values indicate that even the more promising virtual candidates are "
            "predicted to offer only limited improvement relative to the current target scale."
        ),
    })

    reliability = np.nan
    if len(repeatability_summary):
        reliability = float(
            repeatability_summary.iloc[0].get("empirical_repeatability", np.nan)
        )
        evidence.append({
            "Evidence": "Empirical repeatability / noise diagnostic",
            "Value": (
                f"repeatability ≈ {reliability:.3f}"
                if np.isfinite(reliability) else "not estimable"
            ),
            "Interpretation": (
                "This is a rough experiment-level reliability estimate from exact-formulation "
                "repeats, not a formal theoretical R² ceiling."
            ),
        })

    final_r2 = np.nan
    final_spearman = np.nan
    if len(cumulative_summary) and "R2" in cumulative_summary.columns:
        final = cumulative_summary.iloc[-1]
        final_r2 = float(final.get("R2", np.nan))
        final_spearman = float(final.get("Spearman", np.nan))
        evidence.append({
            "Evidence": "Existing grouped nested-CV model performance",
            "Value": (
                f"final R² = {final_r2:.3f}; "
                f"Spearman = {final_spearman:.3f}"
            ),
            "Interpretation": (
                "This measures retrospective predictive/ranking utility under the publication "
                "cross-validation protocol. It is not a reason to chase an arbitrary R²=0.9."
            ),
        })

        if np.isfinite(reliability) and reliability > 0:
            evidence.append({
                "Evidence": "Model R² relative to empirical repeatability",
                "Value": f"R² / repeatability ≈ {final_r2 / reliability:.3f}",
                "Interpretation": (
                    "If close to 1, much of the remaining error may reflect experimental/batch "
                    "variation rather than simply too few formulation points. Treat cautiously "
                    "because the repeatability estimate is based on few duplicate groups."
                ),
            })

    # Internal transparent heuristic. Do NOT use this "score" as a manuscript statistic.
    checks = []
    checks.append(np.isfinite(ei_ratio) and ei_ratio <= 0.35)
    checks.append(np.isfinite(max_ei_sd) and max_ei_sd <= 0.05)
    checks.append(np.isfinite(final_spearman) and final_spearman >= 0.80)
    if np.isfinite(reliability) and np.isfinite(final_r2) and reliability > 0:
        checks.append(final_r2 / reliability >= 0.80)

    n_support = int(sum(bool(x) for x in checks))
    n_total = len(checks)

    if n_total >= 3 and n_support >= n_total - 1:
        internal = "STRONG SUPPORT FOR FREEZING AFTER R4"
    elif n_support >= max(2, math.ceil(n_total / 2)):
        internal = "MODERATE SUPPORT FOR FREEZING AFTER R4"
    else:
        internal = "STOPPING EVIDENCE IS INCOMPLETE; DO NOT CLAIM SATURATION YET"

    evidence.append({
        "Evidence": "Internal stopping heuristic",
        "Value": f"{internal} ({n_support}/{n_total} supportive checks)",
        "Interpretation": (
            "Internal decision aid only. Do NOT report this heuristic score as a publication statistic."
        ),
    })

    paragraph = f"""
R4 STOPPING INTERPRETATION
==========================

The scientifically defensible question is not "why didn't we force R² to 0.9?"
R²=0.9 is not a required endpoint for a noisy biological formulation task.

A defensible R4 freeze point is supported when several independent signals agree:
(1) the grouped nested-CV model has reached useful predictive/ranking performance;
(2) exact-formulation repeats indicate a non-negligible experimental/batch noise floor;
(3) after R4, the acquisition audit finds little remaining Expected Improvement within
    the defined feasible formulation space; and
(4) the next unanswered question is prospective generalization, not further retrospective fitting.

Internal audit conclusion: {internal}.

IMPORTANT:
- This does NOT prove that no formulation outside the defined candidate space can be better.
- It does NOT imply that adding samples could never improve R².
- It supports a methodological decision to FREEZE the R1-R4 model and move to prospective
  validation rather than indefinitely optimizing retrospective cross-validation.
- R5 should therefore remain held out. It should be used to test frozen predictions/ranking
  and should not be added back into model fitting for the headline prospective-validation claim.
""".strip()

    return pd.DataFrame(evidence), paragraph


# =============================================================================
# 11. PLOTS
# =============================================================================

def save_audit_plots(
    virtual_summary: pd.DataFrame,
    cumulative_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = np.arange(len(virtual_summary))
    labels = virtual_summary["stage"].astype(str).tolist()

    fig = plt.figure(figsize=(6.4, 4.3))
    plt.plot(x, virtual_summary["P95_pool_EI"], marker="o", label="P95 EI")
    plt.plot(x, virtual_summary["Max_pool_EI"], marker="o", label="Max EI")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Expected improvement")
    plt.xlabel("Cumulative training data")
    plt.title("Remaining optimization opportunity")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "stopping_expected_improvement.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(6.4, 4.3))
    plt.plot(x, virtual_summary["Mean_pool_novelty"], marker="o", label="Mean novelty")
    plt.plot(x, virtual_summary["Mean_pool_uncertainty"], marker="o", label="Mean GP uncertainty")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.xlabel("Cumulative training data")
    plt.title("Coverage and uncertainty across rounds")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "stopping_coverage_uncertainty.png", dpi=300)
    plt.close(fig)

    if len(cumulative_summary) and "R2" in cumulative_summary.columns:
        fig = plt.figure(figsize=(6.4, 4.3))
        xx = np.arange(len(cumulative_summary))
        labs = cumulative_summary["stage"].astype(str).tolist()
        plt.plot(xx, cumulative_summary["R2"], marker="o", label="Nested-CV R²")
        if "Spearman" in cumulative_summary.columns:
            plt.plot(xx, cumulative_summary["Spearman"], marker="o", label="Spearman")
        plt.xticks(xx, labs, rotation=20, ha="right")
        plt.xlabel("Cumulative training data")
        plt.ylabel("Metric")
        plt.title("Existing cumulative model performance")
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "existing_cumulative_model_performance.png", dpi=300)
        plt.close(fig)


# =============================================================================
# 12. MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical information-value and R4 stopping audit."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional override workbook. Direct-run uses the same forced path as the publication model.",
    )
    parser.add_argument(
        "--sheet",
        default=FORCED_SHEET_NAME,
    )
    parser.add_argument(
        "--output-dir",
        default="",
    )
    parser.add_argument(
        "--virtual-pool-n",
        type=int,
        default=VIRTUAL_POOL_N,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = discover_data_path(args.input)
    sheet = resolve_sheet_name(data_path, args.sheet)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_path.parent / "information_value_R4_stop_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("HISTORICAL INFORMATION VALUE + R4 STOPPING AUDIT")
    print("=" * 92)
    print(f"[Data]   {data_path}")
    print(f"[Sheet]  {sheet!r}")
    print(f"[Target] {TARGET_COLUMN}")
    print(f"[Output] {output_dir}")
    print("=" * 92)

    design_df = load_all_design_rows(data_path, sheet)
    training_df = make_training_rows(design_df)
    categories_map = fixed_categories(design_df)

    # -------------------------------------------------------------------------
    # A) Per-point historical information value
    # -------------------------------------------------------------------------
    points, round_summary = historical_point_audit(
        design_df,
        training_df,
        categories_map,
    )

    # -------------------------------------------------------------------------
    # B) Repeatability / rough empirical noise ceiling
    # -------------------------------------------------------------------------
    duplicate_groups, repeatability_summary = repeatability_audit(training_df)

    # -------------------------------------------------------------------------
    # C) Fixed virtual pool: is there much optimization opportunity left after R4?
    # -------------------------------------------------------------------------
    print(f"\n[VirtualPool] Generating {args.virtual_pool_n} feasible candidates...")
    virtual_pool = generate_virtual_pool(
        design_df,
        n=args.virtual_pool_n,
        seed=RANDOM_STATE,
    )

    virtual_summary = virtual_pool_stage_audit(
        training_df,
        virtual_pool,
        categories_map,
    )

    # -------------------------------------------------------------------------
    # D) Existing publication grouped nested-CV learning curve, if found
    # -------------------------------------------------------------------------
    cumulative_path = discover_latest_cumulative_summary(data_path)
    cumulative_summary = load_cumulative_summary(cumulative_path)

    if cumulative_path is not None:
        print(f"\n[CumulativeCV] Loaded existing result: {cumulative_path}")
    else:
        print("\n[CumulativeCV] Existing cumulative summary not found. "
              "Stopping audit will still run without it.")

    # -------------------------------------------------------------------------
    # E) Integrate the evidence
    # -------------------------------------------------------------------------
    stop_table, stop_text = stopping_evidence(
        virtual_summary,
        repeatability_summary,
        cumulative_summary,
    )

    # -------------------------------------------------------------------------
    # F) Save
    # -------------------------------------------------------------------------
    points.to_csv(
        output_dir / "historical_information_value_points_R2_R5.csv",
        index=False,
        encoding="utf-8-sig",
    )
    round_summary.to_csv(
        output_dir / "historical_information_value_round_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    virtual_summary.to_csv(
        output_dir / "R4_stopping_virtual_pool_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    duplicate_groups.to_csv(
        output_dir / "exact_formulation_repeat_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )
    repeatability_summary.to_csv(
        output_dir / "empirical_repeatability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stop_table.to_csv(
        output_dir / "R4_stopping_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    virtual_pool.to_csv(
        output_dir / "virtual_feasible_pool_for_stopping_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if len(cumulative_summary):
        cumulative_summary.to_csv(
            output_dir / "existing_cumulative_model_summary_used.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with open(
        output_dir / "R4_STOPPING_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(stop_text + "\n")

    # One combined Excel workbook for easy inspection.
    with pd.ExcelWriter(
        output_dir / "information_value_and_R4_stopping_audit.xlsx",
        engine="openpyxl",
    ) as writer:
        points.to_excel(writer, sheet_name="point_information_value", index=False)
        round_summary.to_excel(writer, sheet_name="round_summary", index=False)
        virtual_summary.to_excel(writer, sheet_name="R4_stopping_virtual", index=False)
        stop_table.to_excel(writer, sheet_name="R4_stopping_evidence", index=False)
        repeatability_summary.to_excel(writer, sheet_name="repeatability_summary", index=False)
        duplicate_groups.to_excel(writer, sheet_name="duplicate_groups", index=False)
        if len(cumulative_summary):
            cumulative_summary.to_excel(writer, sheet_name="existing_cumulative_CV", index=False)

    save_audit_plots(
        virtual_summary,
        cumulative_summary,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # G) Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("[PER-ROUND INFORMATION-VALUE SUMMARY]")
    show_cols = [
        "Round_scored", "Training_rounds", "Train_n", "Candidate_n",
        "Mean_uncertainty", "Mean_EI", "Mean_IG", "Mean_novelty",
    ]
    print(round_summary[show_cols].to_string(index=False))

    print("\n" + "=" * 92)
    print("[VIRTUAL-POOL STOPPING SUMMARY]")
    show_cols2 = [
        "stage", "n_training",
        "P95_pool_EI", "Max_pool_EI",
        "P95_EI_over_target_SD", "Max_EI_over_target_SD",
        "Mean_pool_uncertainty", "Mean_pool_novelty",
    ]
    print(virtual_summary[show_cols2].to_string(index=False))

    print("\n" + "=" * 92)
    print("[REPEATABILITY]")
    print(repeatability_summary.to_string(index=False))

    print("\n" + "=" * 92)
    print("[R4 STOPPING EVIDENCE]")
    print(stop_table.to_string(index=False))

    print("\n" + stop_text)

    print("\n" + "=" * 92)
    print("[R5 RULE]")
    print("R5 rows were scored using R1-R4 only.")
    print("R5 was NOT used for training in any calculation in this script.")
    print("=" * 92)

    print(f"\n[Done] Main workbook:\n{output_dir / 'information_value_and_R4_stopping_audit.xlsx'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 92)
        print("[ProgramError]")
        print(type(exc).__name__, ":", exc)
        print("=" * 92)
        raise
