# -*- coding: utf-8 -*-
"""
DC24_LNP_Transfection_ModelSelection_NestedCV.py

Publication-oriented modeling pipeline for DC2.4 mRNA-LNP transfection efficiency.

What this version fixes
-----------------------
1. Feature selection is performed inside each outer training fold only.
2. Hyperparameter tuning is performed inside each outer training fold only (nested CV).
3. No tiny 10% random holdout is misreported as an independent test set.
4. Core formulation variables are always retained; mRMR selects only auxiliary
   structural / fingerprint features.
5. A pre-specified local tree ensemble is evaluated to reduce single-model selection bias.
6. Optional leave-one-round-out evaluation measures across-round generalization.
7. Optional prospective external validation calculates R2, RMSE, MAE, Spearman,
   Pearson and exports predicted-vs-experimental figures.
8. Post-synthesis size/PDI/zeta variables are not used by the primary pre-experimental model.
9. Exact duplicate formulations are kept in the same outer and inner folds by default.
10. Missing size/PDI values are excluded by default, and round labels are recovered from candidate IDs.
11. Cumulative round learning curves are evaluated for R1, R1+R2, R1+R2+R3, etc.
12. Morgan fingerprints are preserved: precomputed Excel fingerprint columns are preferred, with RDKit/SMILES fallback.

Important target-scale rule
---------------------------
External RMSE is scientifically valid only when experimental values are on exactly the
same scale as the training target. By default, the script trains on the workbook column
"normolized for DC2.4" (including spelling variants). For external validation, create a
column on the same normalization scale and pass it with --external-actual-column.

Typical commands
----------------
Primary nested-CV analysis:
    python DC24_LNP_Transfection_ModelSelection_NestedCV.py \
        --data-path "C:\\path\\lnp_dc24_hacat_modeling_dataset.xlsx"

Faster test run:
    python DC24_LNP_Transfection_ModelSelection_NestedCV.py \
        --data-path "C:\\path\\lnp_dc24_hacat_modeling_dataset.xlsx" \
        --outer-repeats 1 --tune-iter 5

Prospective S15 validation, when S15 already contains the same normalized target:
    python DC24_LNP_Transfection_ModelSelection_NestedCV.py \
        --data-path "C:\\path\\lnp_dc24_hacat_modeling_dataset.xlsx" \
        --external-path "C:\\path\\S15_validation.xlsx" \
        --external-actual-column "actual_normolized_for_DC2.4"

Train directly on log10(raw RLU), then validate raw S15 RLU on the same log10 scale:
    python DC24_LNP_Transfection_ModelSelection_NestedCV.py \
        --data-path "C:\\path\\lnp_dc24_hacat_modeling_dataset.xlsx" \
        --target-mode log10_raw \
        --raw-target-column "DC_Cell_Transfection_Efficiency" \
        --external-path "C:\\path\\S15_validation.xlsx" \
        --external-actual-column "DC_Cell_Transfection_Efficiency" \
        --external-actual-scale log10_raw

Dependencies
------------
Required: numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, joblib
Optional: rdkit, xgboost, lightgbm
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HGB = True
except Exception:
    HAS_HGB = False

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))

SCRIPT_VERSION = "2.3.2-publication-safe-same-folder"
# Your Excel filename appears in Windows Explorer as "Inp_..." (capital I),
# while earlier code searched for "lnp_..." (lowercase L). Both spellings,
# plus close filename variants, are accepted below.
DEFAULT_FILE = "Inp_dc24_hacat_modeling_dataset.xlsx"
DATA_FILE_CANDIDATES = (
    "Inp_dc24_hacat_modeling_dataset.xlsx",  # capital I, as shown in Explorer
    "lnp_dc24_hacat_modeling_dataset.xlsx",  # lowercase L
    "Lnp_dc24_hacat_modeling_dataset.xlsx",
    "LNP_dc24_hacat_modeling_dataset.xlsx",
)

# -----------------------------------------------------------------------------
# Column aliases
# -----------------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "candidate_id": [
        "candidate_id", "Candidate_ID", "Selection_Order", "Formulation_ID",
        "Sample", "ID", "编号", "配方编号",
    ],
    "IL1": ["Ionizable_Lipid_1", "IL1", "ionizable lipid 1", "可离子化脂质1", "离子化脂质1"],
    "IL2": ["Ionizable_Lipid_2", "IL2", "ionizable lipid 2", "可离子化脂质2", "离子化脂质2"],
    "IL1_molpct": ["IL1_Mol_Percent", "IL1_molpct", "IL1 mol%", "IL1 mol pct"],
    "IL2_molpct": ["IL2_Mol_Percent", "IL2_molpct", "IL2 mol%", "IL2 mol pct"],
    "Phospholipid": ["Phospholipid", "HL", "Helper_Lipid", "磷脂"],
    "HL_molpct": ["Phospholipid_Mol_Percent", "HL_molpct", "Helper_Lipid_Mol_Percent", "磷脂比例"],
    "CHOL_molpct": ["Cholesterol_Mol_Percent", "CHOL_molpct", "Cholesterol", "胆固醇比例"],
    "PEG": ["PEG类型", "PEG", "PEG_Lipid", "PEG lipid"],
    "PEG_molpct": ["PEG_Mol_Percent", "PEG_molpct", "PEG mol%", "PEG比例"],
    "NP_ratio": ["N/P", "NP_ratio", "N_P_ratio", "N/P ratio"],
    "Round": ["Round", "round", "轮次"],
    "normalized_target": [
        "normolized for DC2.4", "normalized for DC2.4",
        "normolized for DC2_4", "normalized for DC2_4",
        "normolized DC2.4", "normalized DC2.4",
        "normolized", "normalized",
    ],
    "same_log_target": [
        "in same log value", "same log value", "log value",
        "DC log value", "DC same log value",
    ],
}

# -----------------------------------------------------------------------------
# Lipid identities and descriptors
# -----------------------------------------------------------------------------
LIPID_ALIASES = {
    "DLIN-MC3-DMA": "MC3", "DLIN-MC3": "MC3", "MC-3": "MC3",
    "SM-102": "SM102", "SM 102": "SM102", "SM102": "SM102",
    "C12200": "C12-200", "C12 200": "C12-200", "C12-200": "C12-200",
    "CKKE12": "CKK-E12", "CKK E12": "CKK-E12", "CCK12": "CKK-E12",
    "CKK-E12": "CKK-E12", "CKK E12 (CCK12)": "CKK-E12",
    "ALC0315": "ALC-0315", "ALC 0315": "ALC-0315", "ALC-0315": "ALC-0315",
    "DMG-PEG": "DMG-PEG2000", "DMG-PEG-2000": "DMG-PEG2000",
    "DMGPEG2000": "DMG-PEG2000", "PEG2000-DMG": "DMG-PEG2000",
    "PEG-DMG": "DMG-PEG2000", "DMG-PEG2000": "DMG-PEG2000",
    "ALC-0159": "C14-PEG", "ALC0159": "C14-PEG", "C14PEG": "C14-PEG",
    "C14-PEG": "C14-PEG",
    "PEG-MANNOSE": "PEG-Mannose", "MANNOSE-PEG": "PEG-Mannose",
    "PEG MANNOSE": "PEG-Mannose", "PEG-Mannose": "PEG-Mannose",
    "CHOL": "Cholesterol", "CHOLESTEROL": "Cholesterol",
    "DSPC": "DSPC", "DOPE": "DOPE", "DOTAP": "DOTAP",
    "DODAP": "DODAP", "MC3": "MC3",
}

# Only ionizable-lipid descriptors are used as auxiliary features.
IONIZABLE_DESCRIPTORS: Dict[str, Dict[str, float]] = {
    "MC3": {
        "head_ionizable_amine": 1, "head_quaternary": 0, "head_polyamine": 0,
        "linker_ester": 1, "linker_amide": 0, "linker_ether": 0,
        "linker_degradable": 1, "tail_count": 2, "tail_carbons": 36,
        "tail_double_bonds": 4, "tail_branched": 1, "tail_saturated": 0,
        "clogp": 13.5, "mw": 642, "tpsa": 0,
    },
    "ALC-0315": {
        "head_ionizable_amine": 1, "head_quaternary": 0, "head_polyamine": 0,
        "linker_ester": 1, "linker_amide": 0, "linker_ether": 0,
        "linker_degradable": 1, "tail_count": 2, "tail_carbons": 32,
        "tail_double_bonds": 0, "tail_branched": 1, "tail_saturated": 1,
        "clogp": 14.0, "mw": 766, "tpsa": 0,
    },
    "SM102": {
        "head_ionizable_amine": 1, "head_quaternary": 0, "head_polyamine": 0,
        "linker_ester": 1, "linker_amide": 0, "linker_ether": 0,
        "linker_degradable": 1, "tail_count": 2, "tail_carbons": 30,
        "tail_double_bonds": 0, "tail_branched": 1, "tail_saturated": 1,
        "clogp": 13.8, "mw": 710, "tpsa": 0,
    },
    "C12-200": {
        "head_ionizable_amine": 1, "head_quaternary": 0, "head_polyamine": 1,
        "linker_ester": 0, "linker_amide": 0, "linker_ether": 0,
        "linker_degradable": 0, "tail_count": 5, "tail_carbons": 60,
        "tail_double_bonds": 0, "tail_branched": 0, "tail_saturated": 1,
        "clogp": 15.0, "mw": 1108, "tpsa": 0,
    },
    "CKK-E12": {
        "head_ionizable_amine": 1, "head_quaternary": 0, "head_polyamine": 1,
        "linker_ester": 0, "linker_amide": 1, "linker_ether": 1,
        "linker_degradable": 1, "tail_count": 4, "tail_carbons": 48,
        "tail_double_bonds": 0, "tail_branched": 0, "tail_saturated": 1,
        "clogp": 12.0, "mw": 1100, "tpsa": 0,
    },
    "DOTAP": {
        "head_ionizable_amine": 0, "head_quaternary": 1, "head_polyamine": 0,
        "linker_ester": 1, "linker_amide": 0, "linker_ether": 0,
        "linker_degradable": 1, "tail_count": 2, "tail_carbons": 36,
        "tail_double_bonds": 2, "tail_branched": 0, "tail_saturated": 0,
        "clogp": 13.0, "mw": 698, "tpsa": 0,
    },
    "DODAP": {
        "head_ionizable_amine": 1, "head_quaternary": 0, "head_polyamine": 0,
        "linker_ester": 1, "linker_amide": 0, "linker_ether": 0,
        "linker_degradable": 1, "tail_count": 2, "tail_carbons": 36,
        "tail_double_bonds": 2, "tail_branched": 0, "tail_saturated": 0,
        "clogp": 13.0, "mw": 648, "tpsa": 0,
    },
}

IL_CATEGORIES = ["ALC-0315", "C12-200", "CKK-E12", "DODAP", "DOTAP", "MC3", "SM102", "UNKNOWN"]
IL2_CATEGORIES = ["NONE"] + IL_CATEGORIES
HL_CATEGORIES = ["DOPE", "DSPC", "UNKNOWN"]
PEG_CATEGORIES = ["C14-PEG", "DMG-PEG2000", "PEG-Mannose", "UNKNOWN"]

LIPID_SMILES_FALLBACK = {
    "MC3": "CCCCC/C=C\\C/C=C\\CCCCCCCCC(CCCCCCCC/C=C\\C/C=C\\CCCCC)OC(=O)CCCN(C)C",
    "DOTAP": "CCCCCCCC/C=C\\CCCCCCCCC(=O)OCC(OC(=O)CCCCCCC/C=C\\CCCCCCCC)C[N+](C)(C)C",
    "DODAP": "CCCCCCCC/C=C\\CCCCCCCCC(=O)OCC(OC(=O)CCCCCCC/C=C\\CCCCCCCC)CN(C)C",
}

# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class CVConfig:
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 4
    tune_iter: int = 12
    auxiliary_top_k: int = 8
    split_mode: str = "grouped"
    group_round_decimals: int = 4
    random_state: int = RANDOM_STATE


@dataclass
class QCConfig:
    pdi_max: float = 0.5
    size_min: float = 30.0
    size_max: float = 300.0
    require_complete: bool = True


@dataclass
class FittedModel:
    model_name: str
    estimator: BaseEstimator
    selected_features: List[str]
    best_params: Dict[str, Any]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        if isinstance(value, str):
            text = re.sub(r"[^0-9.\-eE]", "", value)
            if text in {"", "-", ".", "-."}:
                return default
            return float(text)
        return float(value)
    except Exception:
        return default


def normalize_lipid_name(name: Any) -> Optional[str]:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    text = str(name).strip()
    if not text or text.lower() in {"nan", "none", "na", "-", "无"}:
        return None

    candidates = [text]
    base = re.sub(r"\([^)]*\)", "", text).strip()
    inside = re.findall(r"\(([^)]*)\)", text)
    if base and base != text:
        candidates.append(base)
    candidates.extend(x.strip() for x in inside if x.strip())

    known = set(IONIZABLE_DESCRIPTORS) | {"DSPC", "DOPE", "Cholesterol", "DMG-PEG2000", "C14-PEG", "PEG-Mannose"}
    for candidate in candidates:
        if candidate in known:
            return candidate
        key = candidate.upper().replace("_", "-").replace("  ", " ").strip()
        if key in LIPID_ALIASES:
            return LIPID_ALIASES[key]
        for standard in known:
            if standard.upper() == key:
                return standard
    return text


def find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        key = str(alias).strip().lower()
        if key in normalized:
            return normalized[key]
    return None



def infer_round_from_candidate_id(candidate_id: Any) -> str:
    """Infer R1/R2/R3... when the explicit Round column is missing."""
    if candidate_id is None or (isinstance(candidate_id, float) and np.isnan(candidate_id)):
        return "UNKNOWN"
    text = str(candidate_id).strip()
    match = re.match(r"^R\s*(\d+)\s*[-_ ]?", text, re.I)
    if match:
        return f"R{int(match.group(1))}"
    # Purely numeric historical IDs are treated as first-round samples.
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return "R1"
    return "UNKNOWN"


def fill_missing_round_labels(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    explicit = out["Round"].apply(normalize_round_label) if "Round" in out.columns else pd.Series("UNKNOWN", index=out.index)
    inferred = out["candidate_id"].apply(infer_round_from_candidate_id)
    missing = explicit.isin(["UNKNOWN", "", "nan", "None"])
    out["Round"] = explicit.where(~missing, inferred)
    counts = out["Round"].value_counts(dropna=False).to_dict()
    print(f"[Round] Labels after explicit/inferred recovery: {counts}")
    return out


def formulation_signature(row: pd.Series, decimals: int = 4) -> str:
    """Canonical exact-formulation signature; IL1/IL2 order is normalized."""
    pairs = []
    for lipid_col, pct_col in [("IL1", "IL1_molpct"), ("IL2", "IL2_molpct")]:
        lipid = normalize_lipid_name(row.get(lipid_col)) or "NONE"
        pct = round(safe_float(row.get(pct_col), 0.0), decimals)
        if lipid != "NONE" or abs(pct) > 0:
            pairs.append((lipid, pct))
    pairs = sorted(pairs, key=lambda item: (item[0], item[1]))
    parts = [f"IL:{lipid}:{pct:.{decimals}f}" for lipid, pct in pairs]
    parts.extend([
        f"HL:{normalize_lipid_name(row.get('Phospholipid')) or 'NONE'}:{round(safe_float(row.get('HL_molpct'), 0.0), decimals):.{decimals}f}",
        f"CHOL:{round(safe_float(row.get('CHOL_molpct'), 0.0), decimals):.{decimals}f}",
        f"PEG:{normalize_lipid_name(row.get('PEG')) or 'NONE'}:{round(safe_float(row.get('PEG_molpct'), 0.0), decimals):.{decimals}f}",
    ])
    np_ratio = safe_float(row.get("NP_ratio"), np.nan)
    if np.isfinite(np_ratio):
        parts.append(f"NP:{round(np_ratio, decimals):.{decimals}f}")
    return "|".join(parts)


def make_formulation_groups(df: pd.DataFrame, decimals: int = 4) -> Tuple[np.ndarray, pd.Series]:
    signatures = df.apply(lambda row: formulation_signature(row, decimals), axis=1)
    groups, unique = pd.factorize(signatures, sort=True)
    duplicate_rows = int(signatures.duplicated(keep=False).sum())
    duplicate_groups = int((signatures.value_counts() > 1).sum())
    print(
        f"[Groups] exact formulation groups={len(unique)} | duplicate groups={duplicate_groups} "
        f"| rows in duplicate groups={duplicate_rows}"
    )
    return groups.astype(int), signatures


def find_existing_data_path(user_path: Optional[str]) -> str:
    """Find the training workbook beside this script.

    Recommended layout (both files in exactly the same folder):
        PyCharm 有效代码/
        ├─ DC24_LNP_Transfection_ModelSelection_NestedCV_同文件夹直接运行.py
        └─ Inp_dc24_hacat_modeling_dataset.xlsx

    The function also accepts the old lowercase-l spelling and, as a safety
    fallback, searches the script folder for an Excel filename containing
    dc24 + hacat + modeling + dataset.
    """

    def clean_path(value: Any) -> Path:
        cleaned = str(value).strip().strip('"').strip("'").strip()
        return Path(os.path.expandvars(os.path.expanduser(cleaned)))

    def is_excel_file(path: Path) -> bool:
        return (
            path.is_file()
            and not path.name.startswith("~$")
            and path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}
        )

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    checked: List[Path] = []

    # 1. Optional command-line override.
    if user_path:
        supplied = clean_path(user_path)
        supplied_candidates = [supplied] if supplied.is_absolute() else [
            script_dir / supplied,
            Path.cwd() / supplied,
        ]
        for candidate in supplied_candidates:
            checked.append(candidate)
            if is_excel_file(candidate):
                resolved = candidate.resolve()
                print(f"[DataPath] Using --data-path workbook: {resolved}")
                return str(resolved)
        raise FileNotFoundError(
            "--data-path was provided, but the workbook does not exist:\n"
            + "\n".join(f"  - {p}" for p in supplied_candidates)
        )

    # 2. Direct filename checks in the SAME folder as the script.
    # Parent-folder fallback is retained only in case the script is
    # accidentally left inside the old '8.02 publish' subfolder.
    search_dirs: List[Path] = [script_dir]
    if script_dir.name.casefold() == "8.02 publish".casefold():
        search_dirs.append(script_dir.parent)

    for folder in search_dirs:
        for filename in DATA_FILE_CANDIDATES:
            candidate = folder / filename
            checked.append(candidate)
            if is_excel_file(candidate):
                resolved = candidate.resolve()
                print(f"[DataPath] Using workbook beside the script: {resolved}")
                return str(resolved)

    # 3. Semantic filename match. This handles I/l confusion, spaces,
    # underscores, hyphens, and small renaming differences.
    required_tokens = ("dc24", "hacat", "modeling", "dataset")
    semantic_matches: List[Path] = []
    excel_files_seen: List[Path] = []

    for folder in search_dirs:
        if not folder.is_dir():
            continue
        try:
            files = list(folder.iterdir())
        except (OSError, PermissionError):
            continue

        for path in files:
            if not is_excel_file(path):
                continue
            excel_files_seen.append(path)
            normalized_name = re.sub(r"[^a-z0-9]+", "", path.stem.casefold())
            if all(token in normalized_name for token in required_tokens):
                semantic_matches.append(path)

    semantic_matches = list(dict.fromkeys(p.resolve() for p in semantic_matches))
    if len(semantic_matches) == 1:
        resolved = semantic_matches[0]
        print(
            "[DataPath] Exact spelling differed, but the DC24/HaCaT modeling "
            f"workbook was identified automatically: {resolved}"
        )
        return str(resolved)

    if len(semantic_matches) > 1:
        raise FileNotFoundError(
            "More than one possible DC24/HaCaT modeling workbook was found.\n"
            "Keep only the intended workbook beside the script, or pass its full "
            "path with --data-path:\n"
            + "\n".join(f"  - {p}" for p in semantic_matches)
        )

    nearby = "\n".join(f"  - {p.name}" for p in excel_files_seen) or "  (none)"
    checked_text = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(
        "Could not find the DC24/HaCaT training workbook.\n\n"
        "Put BOTH the Python file and Excel file directly in:\n"
        "  C:\\Users\\ASUS\\Desktop\\AI screen LNP python and excel\\PyCharm 有效代码"
        "\n\nAccepted Excel filenames include:\n"
        + "\n".join(f"  - {name}" for name in DATA_FILE_CANDIDATES)
        + f"\n\nPython file actually running:\n  {script_path}"
        + f"\nFolder actually searched:\n  {script_dir}"
        + f"\n\nExcel files currently visible in the searched folder(s):\n{nearby}"
        + f"\n\nExact paths checked:\n{checked_text}"
    )


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not HAS_SCIPY or len(y_true) < 3:
        return np.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if np.isfinite(value) else np.nan


def safe_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not HAS_SCIPY or len(y_true) < 3:
        return np.nan
    try:
        value = pearsonr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))[0]
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def top_fraction_recall(y_true: Sequence[float], y_pred: Sequence[float], fraction: float = 0.20) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)
    if n < 3:
        return np.nan
    k = max(1, int(np.ceil(n * fraction)))
    true_top = set(np.argsort(yt)[-k:])
    pred_top = set(np.argsort(yp)[-k:])
    return float(len(true_top & pred_top) / k)


def metric_dict(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[valid], yp[valid]
    if len(yt) == 0:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "Spearman", "Pearson", "Top20_recall"]}
    return {
        "R2": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
        "RMSE": rmse(yt, yp),
        "MAE": float(mean_absolute_error(yt, yp)),
        "Spearman": safe_spearman(yt, yp),
        "Pearson": safe_pearson(yt, yp),
        "Top20_recall": top_fraction_recall(yt, yp, 0.20),
    }


# -----------------------------------------------------------------------------
# Data standardization and QC
# -----------------------------------------------------------------------------
def standardize_formulation_columns(df_in: pd.DataFrame, require_target: bool = True) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df_in.copy()
    rename: Dict[str, str] = {}
    source_columns: Dict[str, str] = {}

    for standard, aliases in COLUMN_ALIASES.items():
        if standard in {"normalized_target", "same_log_target"}:
            continue
        source = find_column(df, aliases)
        if source is not None:
            rename[source] = standard
            source_columns[standard] = source
    df = df.rename(columns=rename)

    for column in ["IL1", "IL2", "Phospholipid", "PEG"]:
        if column not in df.columns:
            df[column] = None
        df[column] = df[column].apply(normalize_lipid_name)

    for column in ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct", "NP_ratio"]:
        if column not in df.columns:
            df[column] = 0.0 if column != "NP_ratio" else np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Round" not in df.columns:
        df["Round"] = np.nan
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"sample_{i+1:03d}" for i in range(len(df))]

    if require_target and len(df) == 0:
        raise ValueError("No rows were loaded from the workbook.")
    return df, source_columns


def attach_training_target(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    target_mode: str,
    raw_target_column: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    out = df.copy()

    if target_mode == "normalized":
        source = find_column(original_df, COLUMN_ALIASES["normalized_target"])
        if source is None:
            source = find_column(original_df, COLUMN_ALIASES["same_log_target"])
            if source is None:
                raise ValueError(
                    "No normalized DC2.4 target column was found. Use --target-mode log10_raw "
                    "with --raw-target-column, or add a normalized target column."
                )
            print(f"[TargetWarning] Normalized target was not found; falling back to '{source}'.")
        values = pd.to_numeric(original_df[source], errors="coerce")
        out["TE"] = values.values
        label = source
    elif target_mode == "log10_raw":
        if not raw_target_column:
            raise ValueError("--raw-target-column is required when --target-mode log10_raw.")
        source = find_column(original_df, [raw_target_column])
        if source is None:
            raise ValueError(f"Raw target column '{raw_target_column}' was not found.")
        raw = pd.to_numeric(original_df[source], errors="coerce")
        out["TE"] = np.log10(raw.clip(lower=1e-12))
        label = f"log10({source})"
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    out = out.dropna(subset=["TE"]).reset_index(drop=True)
    return out, label


def drop_invalid_rows(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_in.copy().reset_index(drop=True)
    bad = pd.Series(False, index=df.index)
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)

    known_il = set(IONIZABLE_DESCRIPTORS)
    known_hl = {"DOPE", "DSPC"}
    known_peg = {"C14-PEG", "DMG-PEG2000", "PEG-Mannose"}

    checks = [
        ("IL1", lambda x: x in known_il, "unknown_IL1"),
        ("IL2", lambda x: x is None or x in known_il, "unknown_IL2"),
        ("Phospholipid", lambda x: x in known_hl, "unknown_phospholipid"),
        ("PEG", lambda x: x in known_peg, "unknown_PEG"),
    ]
    for column, predicate, reason in checks:
        mask = ~df[column].apply(predicate)
        bad |= mask
        reasons = reasons.mask(mask, reasons + reason + ";")

    pct_cols = ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct"]
    total = df[pct_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    mask = ((total < 80) | (total > 120)).fillna(True)
    bad |= mask
    reasons = reasons.mask(mask, reasons + "molar_sum_outside_80_120;")

    removed = df.loc[bad].copy()
    if len(removed):
        removed["__invalid_reason__"] = reasons.loc[bad].str.rstrip(";").values
    kept = df.loc[~bad].reset_index(drop=True)
    print(f"[Cleaning] {len(df)} -> {len(kept)} rows; removed {len(removed)} invalid/template rows.")
    return kept, removed


def find_qc_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    pdi_col = None
    size_col = None
    zeta_col = None
    for column in df.columns:
        name = str(column).lower()
        if pdi_col is None and ("pdi" in name or "polydispers" in name or "多分散" in name):
            pdi_col = column
        if size_col is None and any(k in name for k in ["particle_size", "z-ave", "zave", "diameter", "size", "粒径"]):
            size_col = column
        if zeta_col is None and any(k in name for k in ["zeta", "potential", "电位"]):
            zeta_col = column
    return pdi_col, size_col, zeta_col


def apply_qc_filter(df_in: pd.DataFrame, qc: QCConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_in.copy().reset_index(drop=True)
    pdi_col, size_col, _ = find_qc_columns(df)
    if pdi_col is None and size_col is None:
        print("[QC] No size/PDI columns were found; no QC exclusion was applied.")
        return df, df.iloc[0:0].copy()

    bad = pd.Series(False, index=df.index)
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)
    if pdi_col is not None:
        pdi = pd.to_numeric(df[pdi_col], errors="coerce")
        missing = pdi.isna() if qc.require_complete else pd.Series(False, index=df.index)
        mask = missing | (pdi > qc.pdi_max)
        bad |= mask
        reasons = reasons.mask(missing, reasons + "PDI_missing;")
        reasons = reasons.mask((pdi > qc.pdi_max).fillna(False), reasons + f"PDI>{qc.pdi_max};")
    if size_col is not None:
        size = pd.to_numeric(df[size_col], errors="coerce")
        missing = size.isna() if qc.require_complete else pd.Series(False, index=df.index)
        out_of_range = ((size < qc.size_min) | (size > qc.size_max)).fillna(False)
        mask = missing | out_of_range
        bad |= mask
        reasons = reasons.mask(missing, reasons + "size_missing;")
        reasons = reasons.mask(out_of_range, reasons + f"size_outside_{qc.size_min}_{qc.size_max}nm;")

    removed = df.loc[bad].copy()
    if len(removed):
        removed["__QC_removed_reason__"] = reasons.loc[bad].str.rstrip(";").values
    kept = df.loc[~bad].reset_index(drop=True)
    print(f"[QC] {len(df)} -> {len(kept)} rows; removed {len(removed)} QC failures.")
    return kept, removed


# -----------------------------------------------------------------------------
# Optional Morgan fingerprints
# -----------------------------------------------------------------------------
def looks_like_smiles(text: Any) -> bool:
    value = str(text).strip()
    if len(value) < 4:
        return False
    allowed = set("CNOSPFIBrClcnospfibr()[]=#@+-/\\.%0123456789H")
    fraction = sum(char in allowed for char in value) / max(len(value), 1)
    return fraction > 0.85 and ("C" in value or "c" in value)


def detect_name_smiles_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    smiles_col = next((c for c in df.columns if "smiles" in str(c).lower()), None)
    if smiles_col is None:
        best_col, best_score = None, 0.0
        for column in df.columns:
            values = df[column].dropna().astype(str).head(30)
            if len(values) == 0:
                continue
            score = float(np.mean([looks_like_smiles(v) for v in values]))
            if score > best_score:
                best_col, best_score = column, score
        if best_score >= 0.5:
            smiles_col = best_col

    name_col = None
    for column in df.columns:
        if column == smiles_col:
            continue
        name = str(column).lower()
        if any(k in name for k in ["lipid", "name", "compound", "molecule", "脂质", "名称", "abbrev"]):
            name_col = column
            break
    if name_col is None:
        for column in df.columns:
            if column == smiles_col:
                continue
            values = df[column].dropna().astype(str)
            if len(values) and values.str.len().mean() < 40:
                name_col = column
                break
    return name_col, smiles_col


def load_smiles_map(workbook_path: str, sheet_hint: str = "SMILES NAME") -> Dict[str, str]:
    smiles_map = dict(LIPID_SMILES_FALLBACK)
    try:
        xls = pd.ExcelFile(workbook_path)
    except Exception:
        return smiles_map

    ordered = ([sheet_hint] if sheet_hint in xls.sheet_names else []) + list(xls.sheet_names)
    seen: set[str] = set()
    for sheet in ordered:
        if sheet in seen:
            continue
        seen.add(sheet)
        try:
            table = pd.read_excel(workbook_path, sheet_name=sheet)
        except Exception:
            continue
        name_col, smiles_col = detect_name_smiles_columns(table)
        if name_col is None or smiles_col is None:
            continue
        count = 0
        for _, row in table.iterrows():
            lipid = normalize_lipid_name(row.get(name_col))
            smiles = str(row.get(smiles_col)).strip() if pd.notna(row.get(smiles_col)) else ""
            if lipid in IONIZABLE_DESCRIPTORS and smiles and smiles.lower() not in {"nan", "none", "-"}:
                smiles_map[lipid] = smiles
                count += 1
        if count:
            print(f"[SMILES] Loaded {count} ionizable-lipid structures from sheet '{sheet}'.")
            break
    return smiles_map


def build_morgan_map(smiles_map: Dict[str, str], n_bits: int = 128, radius: int = 2) -> Dict[str, np.ndarray]:
    """Generate Morgan fingerprints from SMILES when RDKit is available."""
    try:
        from rdkit import Chem, DataStructs
        try:
            from rdkit.Chem import rdFingerprintGenerator
            use_new = True
        except Exception:
            from rdkit.Chem import AllChem
            use_new = False
    except Exception:
        print("[Morgan] RDKit is unavailable; trying precomputed Excel fingerprints instead.")
        return {}

    result: Dict[str, np.ndarray] = {}
    for lipid, smiles in smiles_map.items():
        if lipid not in IONIZABLE_DESCRIPTORS:
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        if use_new:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = generator.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        array = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        result[lipid] = array.astype(float)
    print(f"[Morgan] Generated {n_bits}-bit fingerprints for {len(result)} ionizable lipids with RDKit.")
    return result


def detect_lipid_name_column(table: pd.DataFrame) -> Optional[str]:
    """Detect a lipid-name column even when a sheet contains no SMILES column."""
    preferred = [
        "lipid", "lipid_name", "name", "compound", "molecule",
        "脂质", "名称", "成分", "abbrev", "abbreviation",
    ]
    normalized = {str(c).strip().lower(): c for c in table.columns}
    for key in preferred:
        if key in normalized:
            return normalized[key]
    for column in table.columns:
        name = str(column).lower()
        if any(key in name for key in preferred):
            return column
    # Fallback: a short text column with recognizable lipid names.
    best_column = None
    best_matches = 0
    for column in table.columns:
        values = table[column].dropna().astype(str).head(50)
        matches = sum(normalize_lipid_name(v) in IONIZABLE_DESCRIPTORS for v in values)
        if matches > best_matches:
            best_matches = matches
            best_column = column
    return best_column if best_matches >= 2 else None


def load_precomputed_morgan_from_excel(
    workbook_path: str,
    sheet_hint: str = "SMILES NAME",
) -> Tuple[Dict[str, np.ndarray], int, str]:
    """
    Load precomputed Morgan/ECFP bit columns from the workbook.

    Accepted column names include fp_0, bit-0, Morgan 0, ECFP_0, etc. This keeps
    fingerprint features available on computers without RDKit.
    """
    try:
        xls = pd.ExcelFile(workbook_path)
    except Exception:
        return {}, 128, "none"

    ordered = ([sheet_hint] if sheet_hint in xls.sheet_names else []) + list(xls.sheet_names)
    seen: set[str] = set()
    pattern = re.compile(r"^(fp|bit|morgan|ecfp)[_\- ]?(\d+)$", re.I)
    for sheet in ordered:
        if sheet in seen:
            continue
        seen.add(sheet)
        try:
            table = pd.read_excel(workbook_path, sheet_name=sheet)
        except Exception:
            continue
        name_col = detect_lipid_name_column(table)
        if name_col is None:
            continue
        indexed_columns: List[Tuple[int, Any]] = []
        for column in table.columns:
            match = pattern.fullmatch(str(column).strip())
            if match:
                indexed_columns.append((int(match.group(2)), column))
        if len(indexed_columns) < 8:
            continue
        indexed_columns.sort(key=lambda item: item[0])
        bit_columns = [column for _, column in indexed_columns]
        result: Dict[str, np.ndarray] = {}
        for _, row in table.iterrows():
            lipid = normalize_lipid_name(row.get(name_col))
            if lipid not in IONIZABLE_DESCRIPTORS:
                continue
            vector = pd.to_numeric(row[bit_columns], errors="coerce").fillna(0.0).values.astype(float)
            result[lipid] = vector
        if result:
            n_bits = len(bit_columns)
            print(
                f"[MorganExcel] Loaded {n_bits}-bit precomputed fingerprints for "
                f"{len(result)} ionizable lipids from sheet '{sheet}'."
            )
            return result, n_bits, f"excel:{sheet}"
    return {}, 128, "none"


def prepare_morgan_features(
    workbook_path: str,
    smiles_map: Dict[str, str],
    requested_bits: int = 128,
    radius: int = 2,
) -> Tuple[Dict[str, np.ndarray], int, str]:
    """Prefer precomputed Excel fingerprints; otherwise generate them with RDKit."""
    excel_map, excel_bits, excel_source = load_precomputed_morgan_from_excel(workbook_path)
    if excel_map:
        return excel_map, excel_bits, excel_source
    rdkit_map = build_morgan_map(smiles_map, n_bits=requested_bits, radius=radius)
    if rdkit_map:
        return rdkit_map, requested_bits, "rdkit_smiles"
    print(
        "[Morgan] No precomputed fingerprint columns were found and RDKit is unavailable. "
        "The Morgan block is retained in the code but cannot be used in this run. "
        "Install RDKit or add fp_0...fp_N columns to the workbook."
    )
    return {}, requested_bits, "unavailable"


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------
def canonical_category(value: Optional[str], categories: Sequence[str], none_label: Optional[str] = None) -> str:
    if value is None and none_label is not None:
        return none_label
    return value if value in categories else "UNKNOWN"


def one_hot_fixed(values: Sequence[str], categories: Sequence[str], prefix: str, index: pd.Index) -> pd.DataFrame:
    categorical = pd.Categorical(values, categories=list(categories))
    dummies = pd.get_dummies(categorical, prefix=prefix, dtype=float)
    dummies.index = index
    expected = [f"{prefix}_{category}" for category in categories]
    return dummies.reindex(columns=expected, fill_value=0.0)


def build_core_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        il1 = safe_float(row.get("IL1_molpct"), 0.0)
        il2 = safe_float(row.get("IL2_molpct"), 0.0)
        hl = safe_float(row.get("HL_molpct"), 0.0)
        chol = safe_float(row.get("CHOL_molpct"), 0.0)
        peg = safe_float(row.get("PEG_molpct"), 0.0)
        total_il = il1 + il2
        rows.append({
            "IL1_molpct": il1,
            "IL2_molpct": il2,
            "IL_total_molpct": total_il,
            "IL1_fraction_in_IL": il1 / total_il if total_il > 0 else 0.0,
            "HL_molpct": hl,
            "CHOL_molpct": chol,
            "PEG_molpct": peg,
            "totalIL_x_PEG": total_il * peg / 100.0,
            "CHOL_x_HL": chol * hl / 100.0,
            "ILHL_total": total_il + hl,
            "HL_fraction_in_ILHL": hl / (total_il + hl) if total_il + hl > 0 else 0.0,
        })
    numeric = pd.DataFrame(rows, index=df.index).fillna(0.0)

    il1_values = [canonical_category(v, IL_CATEGORIES) for v in df["IL1"]]
    il2_values = [canonical_category(v, IL2_CATEGORIES, none_label="NONE") for v in df["IL2"]]
    hl_values = [canonical_category(v, HL_CATEGORIES) for v in df["Phospholipid"]]
    peg_values = [canonical_category(v, PEG_CATEGORIES) for v in df["PEG"]]

    categorical = pd.concat([
        one_hot_fixed(il1_values, IL_CATEGORIES, "IL1", df.index),
        one_hot_fixed(il2_values, IL2_CATEGORIES, "IL2", df.index),
        one_hot_fixed(hl_values, HL_CATEGORIES, "Phospholipid", df.index),
        one_hot_fixed(peg_values, PEG_CATEGORIES, "PEG", df.index),
    ], axis=1)

    return pd.concat([numeric, categorical], axis=1).astype(float)


def weighted_descriptor_features(df: pd.DataFrame) -> pd.DataFrame:
    descriptor_keys = list(next(iter(IONIZABLE_DESCRIPTORS.values())).keys())
    records: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        pairs = [
            (normalize_lipid_name(row.get("IL1")), safe_float(row.get("IL1_molpct"), 0.0)),
            (normalize_lipid_name(row.get("IL2")), safe_float(row.get("IL2_molpct"), 0.0)),
        ]
        valid = [(lipid, weight) for lipid, weight in pairs if lipid in IONIZABLE_DESCRIPTORS and weight > 0]
        total = sum(weight for _, weight in valid)
        feature: Dict[str, float] = {}
        for key in descriptor_keys:
            feature[f"ILdesc_{key}"] = (
                sum(IONIZABLE_DESCRIPTORS[lipid][key] * weight for lipid, weight in valid) / total
                if total > 0 else 0.0
            )
        tail_c = feature.get("ILdesc_tail_carbons", 0.0)
        tail_db = feature.get("ILdesc_tail_double_bonds", 0.0)
        feature["ILdesc_unsaturation_density"] = tail_db / tail_c if tail_c > 0 else 0.0
        records.append(feature)
    return pd.DataFrame(records, index=df.index).fillna(0.0).astype(float)


def build_morgan_features(df: pd.DataFrame, morgan_map: Dict[str, np.ndarray], n_bits: int = 128) -> pd.DataFrame:
    if not morgan_map:
        return pd.DataFrame(index=df.index)
    records: List[np.ndarray] = []
    for _, row in df.iterrows():
        accumulator = np.zeros(n_bits, dtype=float)
        weight_sum = 0.0
        for lipid_column, pct_column in [("IL1", "IL1_molpct"), ("IL2", "IL2_molpct")]:
            lipid = normalize_lipid_name(row.get(lipid_column))
            weight = safe_float(row.get(pct_column), 0.0)
            if lipid in morgan_map and weight > 0:
                vector = np.asarray(morgan_map[lipid], dtype=float)
                if len(vector) != n_bits:
                    vector = np.pad(vector, (0, max(0, n_bits - len(vector))))[:n_bits]
                accumulator += vector * weight
                weight_sum += weight
        records.append(accumulator / weight_sum if weight_sum > 0 else accumulator)
    return pd.DataFrame(records, columns=[f"fp_{i}" for i in range(n_bits)], index=df.index)


def build_feature_blocks(df: pd.DataFrame, morgan_map: Dict[str, np.ndarray], n_bits: int = 128) -> Tuple[pd.DataFrame, pd.DataFrame]:
    core = build_core_features(df)
    descriptors = weighted_descriptor_features(df)
    fingerprints = build_morgan_features(df, morgan_map, n_bits=n_bits)
    auxiliary = pd.concat([descriptors, fingerprints], axis=1).replace([np.inf, -np.inf], 0).fillna(0.0)
    print(f"[Features] core={core.shape[1]} | auxiliary={auxiliary.shape[1]} | total={core.shape[1] + auxiliary.shape[1]}")
    return core, auxiliary


# -----------------------------------------------------------------------------
# Fold-safe auxiliary feature selection
# -----------------------------------------------------------------------------
def mrmr_select_auxiliary(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    if X.shape[1] == 0 or k <= 0:
        return []

    # Remove features that are constant in the training fold.
    variable = [column for column in X.columns if float(np.nanstd(X[column].values.astype(float))) > 1e-12]
    if not variable:
        return []
    Xv = X[variable].replace([np.inf, -np.inf], 0).fillna(0.0)
    k = min(k, Xv.shape[1])

    relevance_values = mutual_info_regression(Xv.values, y.values.astype(float), random_state=RANDOM_STATE)
    relevance = dict(zip(Xv.columns, relevance_values))

    # Pre-filter to reduce runtime and noise from many fingerprint bits.
    pre_n = min(max(4 * k, 24), Xv.shape[1])
    preselected = sorted(Xv.columns, key=lambda c: relevance.get(c, 0.0), reverse=True)[:pre_n]
    selected: List[str] = []
    remaining = list(preselected)

    arrays = {column: Xv[column].values.astype(float) for column in preselected}
    while remaining and len(selected) < k:
        best_feature = None
        best_score = -np.inf
        for feature in remaining:
            if not selected:
                score = relevance.get(feature, 0.0)
            else:
                redundancy: List[float] = []
                for chosen in selected:
                    a, b = arrays[feature], arrays[chosen]
                    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                        redundancy.append(0.0)
                    else:
                        corr = np.corrcoef(a, b)[0, 1]
                        redundancy.append(0.0 if not np.isfinite(corr) else abs(float(corr)))
                score = relevance.get(feature, 0.0) - float(np.mean(redundancy))
            if score > best_score:
                best_score = score
                best_feature = feature
        if best_feature is None:
            break
        selected.append(best_feature)
        remaining.remove(best_feature)
    return selected


def combine_selected_features(
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    indices: Sequence[int],
    selected_aux: Sequence[str],
) -> pd.DataFrame:
    columns = list(X_core.columns) + list(selected_aux)
    combined = pd.concat([X_core.iloc[list(indices)], X_aux.iloc[list(indices)][list(selected_aux)]], axis=1)
    return combined.reindex(columns=columns, fill_value=0.0).astype(float)


# -----------------------------------------------------------------------------
# Model definitions and nested CV
# -----------------------------------------------------------------------------
def model_specs(include_optional_boosters: bool = True) -> Dict[str, Tuple[BaseEstimator, Dict[str, Sequence[Any]]]]:
    specs: Dict[str, Tuple[BaseEstimator, Dict[str, Sequence[Any]]]] = {
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300, 500, 700, 900],
                "max_features": [0.35, 0.5, 0.7, "sqrt"],
                "min_samples_leaf": [1, 2, 3, 4],
                "max_depth": [3, 4, 5, 6, None],
            },
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [300, 500, 700, 900],
                "max_features": [0.35, 0.5, 0.7, "sqrt"],
                "min_samples_leaf": [1, 2, 3, 4],
                "max_depth": [3, 4, 5, 6, None],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [120, 200, 300, 500],
                "learning_rate": [0.01, 0.02, 0.03, 0.05],
                "max_depth": [2, 3],
                "min_samples_leaf": [2, 3, 4, 5],
                "subsample": [0.7, 0.85, 1.0],
            },
        ),
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
        ),
    }
    if HAS_HGB:
        specs["HistGradientBoosting"] = (
            HistGradientBoostingRegressor(early_stopping=False, random_state=RANDOM_STATE),
            {
                "max_iter": [150, 250, 350, 500],
                "learning_rate": [0.01, 0.02, 0.03, 0.05],
                "max_depth": [3, 4, 5, None],
                "max_leaf_nodes": [7, 15, 31],
                "min_samples_leaf": [5, 8, 10, 12],
                "l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0],
            },
        )

    if include_optional_boosters:
        try:
            from xgboost import XGBRegressor
            specs["XGBoost"] = (
                XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
                {
                    "n_estimators": [150, 300, 500],
                    "learning_rate": [0.01, 0.02, 0.03, 0.05],
                    "max_depth": [2, 3, 4],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                    "min_child_weight": [1, 2, 3, 5],
                },
            )
        except Exception:
            pass
        try:
            from lightgbm import LGBMRegressor
            specs["LightGBM"] = (
                LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
                {
                    "n_estimators": [150, 300, 500],
                    "learning_rate": [0.01, 0.02, 0.03, 0.05],
                    "num_leaves": [7, 15, 31],
                    "min_child_samples": [5, 8, 12, 16],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                },
            )
        except Exception:
            pass
    return specs


def fit_tuned_estimator(
    estimator: BaseEstimator,
    param_dist: Dict[str, Sequence[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: CVConfig,
    seed_offset: int = 0,
    groups_train: Optional[Sequence[int]] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    inner_splits = min(config.inner_folds, len(y_train))
    if inner_splits < 2:
        fitted = clone(estimator).fit(X_train.values, y_train.values)
        return fitted, {}

    fit_groups = None
    if groups_train is not None:
        groups_array = np.asarray(groups_train)
        n_unique_groups = len(np.unique(groups_array))
        grouped_splits = min(config.inner_folds, n_unique_groups)
        if grouped_splits >= 2:
            inner_cv = GroupKFold(n_splits=grouped_splits)
            fit_groups = groups_array
        else:
            inner_cv = KFold(
                n_splits=inner_splits, shuffle=True,
                random_state=config.random_state + seed_offset,
            )
    else:
        inner_cv = KFold(
            n_splits=inner_splits, shuffle=True,
            random_state=config.random_state + seed_offset,
        )
    total_combinations = int(np.prod([len(values) for values in param_dist.values()])) if param_dist else 1
    n_iter = min(config.tune_iter, total_combinations)

    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="r2",
        cv=inner_cv,
        random_state=config.random_state + seed_offset,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train.values, y_train.values, groups=fit_groups)
    return search.best_estimator_, dict(search.best_params_)


def make_outer_splits(
    n_samples: int,
    config: CVConfig,
    groups: Optional[Sequence[int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = min(config.outer_folds, n_samples)
    if n_splits < 2:
        raise ValueError("Too few samples for nested cross-validation.")

    if config.split_mode == "grouped":
        if groups is None:
            raise ValueError("Grouped nested CV requires formulation group labels.")
        group_array = np.asarray(groups)
        unique_groups = np.unique(group_array)
        n_splits = min(n_splits, len(unique_groups))
        if n_splits < 2:
            raise ValueError("Too few unique formulation groups for grouped CV.")
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        # Repeated shuffled group assignment. No formulation group crosses train/test.
        for repeat in range(config.outer_repeats):
            rng = np.random.default_rng(config.random_state + repeat)
            shuffled = unique_groups.copy()
            rng.shuffle(shuffled)
            group_folds = np.array_split(shuffled, n_splits)
            for test_groups in group_folds:
                test_mask = np.isin(group_array, test_groups)
                train_idx = np.where(~test_mask)[0]
                test_idx = np.where(test_mask)[0]
                splits.append((train_idx, test_idx))
        return splits

    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=config.outer_repeats,
        random_state=config.random_state,
    )
    dummy = np.zeros((n_samples, 1))
    return [(train, test) for train, test in splitter.split(dummy)]


def nested_cv_single_model(
    model_name: str,
    estimator: BaseEstimator,
    param_dist: Dict[str, Sequence[Any]],
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    y: pd.Series,
    sample_ids: Sequence[str],
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: CVConfig,
    formulation_groups: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(y)
    prediction_lists: List[List[float]] = [[] for _ in range(n)]
    fold_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []

    for fold_number, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        selected_aux = mrmr_select_auxiliary(
            X_aux.iloc[train_idx], y_train, k=config.auxiliary_top_k
        )
        X_train = combine_selected_features(X_core, X_aux, train_idx, selected_aux)
        X_test = combine_selected_features(X_core, X_aux, test_idx, selected_aux)

        train_groups = None if formulation_groups is None else np.asarray(formulation_groups)[train_idx]
        fitted, best_params = fit_tuned_estimator(
            estimator, param_dist, X_train, y_train, config, seed_offset=fold_number,
            groups_train=train_groups,
        )
        pred_train = np.asarray(fitted.predict(X_train.values), dtype=float)
        pred_test = np.asarray(fitted.predict(X_test.values), dtype=float)

        for row_index, prediction in zip(test_idx, pred_test):
            prediction_lists[int(row_index)].append(float(prediction))

        test_metrics = metric_dict(y_test.values, pred_test)
        train_metrics = metric_dict(y_train.values, pred_train)
        fold_rows.append({
            "model": model_name,
            "fold": fold_number,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "R2_overfit_gap": train_metrics["R2"] - test_metrics["R2"],
            "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            "n_core_features": X_core.shape[1],
            "n_aux_features": len(selected_aux),
        })
        for rank, feature in enumerate(selected_aux, start=1):
            feature_rows.append({
                "model": model_name,
                "fold": fold_number,
                "aux_rank": rank,
                "feature": feature,
            })

        print(
            f"[NestedCV] {model_name:<22} fold {fold_number:02d}/{len(outer_splits)} "
            f"R2={test_metrics['R2']:.3f} RMSE={test_metrics['RMSE']:.3f} "
            f"Spearman={test_metrics['Spearman']:.3f} aux={len(selected_aux)}"
        )

    oof_rows: List[Dict[str, Any]] = []
    for i, values in enumerate(prediction_lists):
        if not values:
            raise RuntimeError(f"Sample index {i} did not receive an outer-fold prediction.")
        oof_rows.append({
            "sample_index": i,
            "candidate_id": str(sample_ids[i]),
            "actual": float(y.iloc[i]),
            "predicted": float(np.mean(values)),
            "prediction_sd_across_repeats": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_predictions": len(values),
            "model": model_name,
        })
    return pd.DataFrame(oof_rows), pd.DataFrame(fold_rows), pd.DataFrame(feature_rows)


def summarize_oof(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> Dict[str, Any]:
    metrics = metric_dict(oof["actual"].values, oof["predicted"].values)
    return {
        "model": str(oof["model"].iloc[0]),
        **metrics,
        "Fold_R2_mean": float(fold_metrics["test_R2"].mean()),
        "Fold_R2_std": float(fold_metrics["test_R2"].std(ddof=1)),
        "Fold_RMSE_mean": float(fold_metrics["test_RMSE"].mean()),
        "Fold_RMSE_std": float(fold_metrics["test_RMSE"].std(ddof=1)),
        "Fold_Spearman_mean": float(fold_metrics["test_Spearman"].mean()),
        "Fold_Spearman_std": float(fold_metrics["test_Spearman"].std(ddof=1)),
        "Mean_overfit_gap": float(fold_metrics["R2_overfit_gap"].mean()),
    }


def select_primary_ensemble_members(model_names: Sequence[str]) -> List[str]:
    # Pre-specified local tree ensemble. This is not selected after seeing S15.
    preferred = ["RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting"]
    members = [name for name in preferred if name in model_names]
    return members[:3] if len(members) >= 3 else members


def build_ensemble_oof(all_oof: pd.DataFrame, members: Sequence[str]) -> pd.DataFrame:
    subset = all_oof[all_oof["model"].isin(members)].copy()
    pivot = subset.pivot_table(index=["sample_index", "candidate_id", "actual"], columns="model", values="predicted")
    missing = [member for member in members if member not in pivot.columns]
    if missing:
        raise ValueError(f"Cannot build ensemble; missing OOF predictions for {missing}.")
    ensemble = pivot[list(members)].mean(axis=1).rename("predicted").reset_index()
    ensemble["prediction_sd_across_models"] = pivot[list(members)].std(axis=1).values
    ensemble["n_models"] = len(members)
    ensemble["model"] = "PrimaryTreeEnsemble"
    return ensemble


# -----------------------------------------------------------------------------
# Cumulative round learning-curve evaluation
# -----------------------------------------------------------------------------
def round_sort_key(label: str) -> Tuple[int, str]:
    match = re.search(r"R\s*(\d+)", str(label), re.I)
    return (int(match.group(1)), str(label)) if match else (10**9, str(label))


def save_cumulative_learning_curve(summary: pd.DataFrame, output_path: str) -> None:
    """Save R2, RMSE and Spearman of the primary ensemble across cumulative rounds."""
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    subset = summary[summary["model"] == "PrimaryTreeEnsemble"].copy()
    if subset.empty:
        return
    subset = subset.sort_values("stage_order")
    x = np.arange(len(subset))
    labels = subset["stage"].astype(str).tolist()

    fig = plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, subset["R2"].values, marker="o", label="R²")
    plt.plot(x, subset["Spearman"].values, marker="o", label="Spearman")
    plt.plot(x, subset["RMSE"].values, marker="o", label="RMSE")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.xlabel("Cumulative training rounds")
    plt.ylabel("Metric value")
    plt.title("Cumulative-round model performance")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def cumulative_round_nested_cv(
    df: pd.DataFrame,
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    y: pd.Series,
    specs: Dict[str, Tuple[BaseEstimator, Dict[str, Sequence[Any]]]],
    ensemble_members: Sequence[str],
    config: CVConfig,
    output_dir: str,
    full_summary: pd.DataFrame,
    full_oof: pd.DataFrame,
    full_fold_metrics: pd.DataFrame,
    full_fold_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate cumulative datasets: R1, R1+R2, R1+R2+R3, ...

    Every non-final stage is evaluated with the same fold-safe feature selection,
    grouped nested CV, hyperparameter tuning and pre-specified tree ensemble as the
    main analysis. The final all-round stage reuses the already-computed full result.
    """
    rounds = df["Round"].apply(normalize_round_label)
    valid_rounds = sorted(
        [r for r in rounds.unique() if r not in {"UNKNOWN", "", "nan", "None"}],
        key=round_sort_key,
    )
    if not valid_rounds:
        print("[CumulativeRounds] No usable round labels; analysis skipped.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_frames: List[pd.DataFrame] = []
    oof_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    feature_frames: List[pd.DataFrame] = []

    for stage_order, end_round in enumerate(valid_rounds, start=1):
        included_rounds = valid_rounds[:stage_order]
        stage = "+".join(included_rounds)
        mask = rounds.isin(included_rounds).values
        indices = np.where(mask)[0]
        if len(indices) < 15:
            print(f"[CumulativeRounds] Skipping {stage}: only {len(indices)} samples.")
            continue

        # Reuse the full analysis for the last cumulative stage to avoid duplicate computation.
        is_full_stage = stage_order == len(valid_rounds) and len(indices) == len(df)
        if is_full_stage:
            stage_summary = full_summary.copy()
            stage_oof = full_oof.copy()
            stage_folds = full_fold_metrics.copy()
            stage_features = full_fold_features.copy()
            stage_groups, _ = make_formulation_groups(df.iloc[indices].reset_index(drop=True), config.group_round_decimals)
        else:
            stage_df = df.iloc[indices].reset_index(drop=True)
            stage_y = y.iloc[indices].reset_index(drop=True)
            stage_core = X_core.iloc[indices].reset_index(drop=True)
            stage_aux = X_aux.iloc[indices].reset_index(drop=True)
            stage_ids = stage_df["candidate_id"].astype(str).tolist()
            stage_groups, _ = make_formulation_groups(stage_df, config.group_round_decimals)
            outer_groups = stage_groups if config.split_mode == "grouped" else None
            stage_splits = make_outer_splits(len(stage_y), config, groups=outer_groups)

            stage_oof_parts: List[pd.DataFrame] = []
            stage_fold_parts: List[pd.DataFrame] = []
            stage_feature_parts: List[pd.DataFrame] = []
            stage_summary_rows: List[Dict[str, Any]] = []
            print(
                f"\n[CumulativeRounds] Stage={stage} | n={len(stage_y)} | "
                f"groups={len(np.unique(stage_groups))}"
            )
            for model_name, (estimator, param_dist) in specs.items():
                oof, folds, features = nested_cv_single_model(
                    model_name, estimator, param_dist,
                    stage_core, stage_aux, stage_y, stage_ids, stage_splits, config,
                    formulation_groups=stage_groups if config.split_mode == "grouped" else None,
                )
                stage_oof_parts.append(oof)
                stage_fold_parts.append(folds)
                stage_feature_parts.append(features)
                stage_summary_rows.append(summarize_oof(oof, folds))

            stage_oof = pd.concat(stage_oof_parts, ignore_index=True)
            stage_folds = pd.concat(stage_fold_parts, ignore_index=True)
            stage_features = (
                pd.concat(stage_feature_parts, ignore_index=True)
                if stage_feature_parts else pd.DataFrame()
            )
            ensemble_oof = build_ensemble_oof(stage_oof, ensemble_members)
            ensemble_metrics = metric_dict(ensemble_oof["actual"], ensemble_oof["predicted"])
            stage_summary_rows.append({
                "model": "PrimaryTreeEnsemble",
                **ensemble_metrics,
                "Fold_R2_mean": np.nan,
                "Fold_R2_std": np.nan,
                "Fold_RMSE_mean": np.nan,
                "Fold_RMSE_std": np.nan,
                "Fold_Spearman_mean": np.nan,
                "Fold_Spearman_std": np.nan,
                "Mean_overfit_gap": np.nan,
            })
            stage_oof = pd.concat([stage_oof, ensemble_oof], ignore_index=True, sort=False)
            stage_summary = pd.DataFrame(stage_summary_rows)

        if "publication_score" not in stage_summary.columns:
            stage_summary["publication_score"] = (
                0.55 * stage_summary["R2"].fillna(-1).clip(-1, 1)
                + 0.30 * stage_summary["Spearman"].fillna(-1).clip(-1, 1)
                + 0.15 * (1.0 / (1.0 + stage_summary["RMSE"].clip(lower=0)))
            )

        metadata = {
            "stage": stage,
            "stage_order": stage_order,
            "included_rounds": ",".join(included_rounds),
            "n_samples": len(indices),
            "n_formulation_groups": len(np.unique(stage_groups)),
        }
        for key, value in metadata.items():
            stage_summary[key] = value
            stage_oof[key] = value
            stage_folds[key] = value
            if not stage_features.empty:
                stage_features[key] = value

        summary_frames.append(stage_summary)
        oof_frames.append(stage_oof)
        fold_frames.append(stage_folds)
        if not stage_features.empty:
            feature_frames.append(stage_features)

        ensemble_stage = stage_oof[stage_oof["model"] == "PrimaryTreeEnsemble"].copy()
        if not ensemble_stage.empty:
            metrics = metric_dict(ensemble_stage["actual"], ensemble_stage["predicted"])
            safe_stage = re.sub(r"[^A-Za-z0-9_+\-]", "_", stage)
            save_parity_plot(
                ensemble_stage["actual"], ensemble_stage["predicted"],
                os.path.join(output_dir, f"cumulative_{safe_stage}_primary_ensemble_parity.png"),
                f"Cumulative rounds {stage}: primary tree ensemble", metrics,
            )
            save_residual_plot(
                ensemble_stage["actual"], ensemble_stage["predicted"],
                os.path.join(output_dir, f"cumulative_{safe_stage}_primary_ensemble_residuals.png"),
                f"Cumulative rounds {stage}: residuals",
            )
            print(
                f"[CumulativeRounds] {stage} ensemble: R2={metrics['R2']:.3f}, "
                f"RMSE={metrics['RMSE']:.3f}, Spearman={metrics['Spearman']:.3f}"
            )

    cumulative_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    cumulative_oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    cumulative_folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    cumulative_features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    save_cumulative_learning_curve(
        cumulative_summary,
        os.path.join(output_dir, "cumulative_round_learning_curve.png"),
    )
    return cumulative_summary, cumulative_oof, cumulative_folds, cumulative_features


# -----------------------------------------------------------------------------
# Leave-one-round-out evaluation
# -----------------------------------------------------------------------------
def normalize_round_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"
    text = str(value).strip()
    match = re.search(r"R\s*(\d+)", text, re.I)
    return f"R{match.group(1)}" if match else text


def leave_one_round_out_ensemble(
    df: pd.DataFrame,
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    y: pd.Series,
    specs: Dict[str, Tuple[BaseEstimator, Dict[str, Sequence[Any]]]],
    members: Sequence[str],
    config: CVConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rounds = df["Round"].apply(normalize_round_label)
    formulation_groups, _ = make_formulation_groups(df, config.group_round_decimals)
    valid_rounds = [r for r in sorted(rounds.unique()) if r not in {"UNKNOWN", "", "nan"}]
    if len(valid_rounds) < 2:
        print("[RoundCV] Fewer than two usable rounds; leave-one-round-out evaluation was skipped.")
        return pd.DataFrame(), pd.DataFrame()

    prediction_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    for round_index, test_round in enumerate(valid_rounds, start=1):
        test_idx = np.where(rounds.values == test_round)[0]
        train_idx = np.where(rounds.values != test_round)[0]
        if len(test_idx) < 2 or len(train_idx) < 10:
            print(f"[RoundCV] Skipping {test_round}: n_train={len(train_idx)}, n_test={len(test_idx)}")
            continue

        selected_aux = mrmr_select_auxiliary(X_aux.iloc[train_idx], y.iloc[train_idx], config.auxiliary_top_k)
        X_train = combine_selected_features(X_core, X_aux, train_idx, selected_aux)
        X_test = combine_selected_features(X_core, X_aux, test_idx, selected_aux)

        member_predictions: List[np.ndarray] = []
        for member_number, member in enumerate(members, start=1):
            estimator, param_dist = specs[member]
            fitted, _ = fit_tuned_estimator(
                estimator, param_dist, X_train, y.iloc[train_idx], config,
                seed_offset=5000 + round_index * 10 + member_number,
                groups_train=formulation_groups[train_idx],
            )
            member_predictions.append(np.asarray(fitted.predict(X_test.values), dtype=float))
        ensemble_prediction = np.mean(np.vstack(member_predictions), axis=0)
        metrics = metric_dict(y.iloc[test_idx].values, ensemble_prediction)
        fold_rows.append({
            "test_round": test_round,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **metrics,
            "n_aux_features": len(selected_aux),
            "ensemble_members": ",".join(members),
        })
        for idx, prediction in zip(test_idx, ensemble_prediction):
            prediction_rows.append({
                "sample_index": int(idx),
                "candidate_id": str(df.iloc[idx]["candidate_id"]),
                "Round": test_round,
                "actual": float(y.iloc[idx]),
                "predicted": float(prediction),
            })
        print(
            f"[RoundCV] held out {test_round}: R2={metrics['R2']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, Spearman={metrics['Spearman']:.3f}"
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


# -----------------------------------------------------------------------------
# Final deployment model
# -----------------------------------------------------------------------------
def fit_final_models(
    specs: Dict[str, Tuple[BaseEstimator, Dict[str, Sequence[Any]]]],
    members: Sequence[str],
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    y: pd.Series,
    config: CVConfig,
    formulation_groups: Optional[Sequence[int]] = None,
) -> Tuple[List[FittedModel], List[str]]:
    selected_aux = mrmr_select_auxiliary(X_aux, y, config.auxiliary_top_k)
    indices = np.arange(len(y))
    X_full = combine_selected_features(X_core, X_aux, indices, selected_aux)
    fitted_models: List[FittedModel] = []
    for member_number, member in enumerate(members, start=1):
        estimator, param_dist = specs[member]
        fitted, best_params = fit_tuned_estimator(
            estimator, param_dist, X_full, y, config, seed_offset=9000 + member_number,
            groups_train=formulation_groups,
        )
        fitted_models.append(FittedModel(
            model_name=member,
            estimator=fitted,
            selected_features=list(X_full.columns),
            best_params=best_params,
        ))
        print(f"[FinalFit] {member} fitted with {X_full.shape[1]} features.")
    return fitted_models, selected_aux


def predict_with_final_models(
    fitted_models: Sequence[FittedModel],
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    member_predictions: Dict[str, np.ndarray] = {}
    for fitted in fitted_models:
        feature_frame = pd.concat([X_core, X_aux], axis=1).reindex(columns=fitted.selected_features, fill_value=0.0)
        member_predictions[fitted.model_name] = np.asarray(
            fitted.estimator.predict(feature_frame.values.astype(float)), dtype=float
        )
    member_table = pd.DataFrame(member_predictions, index=X_core.index)
    return (
        member_table.mean(axis=1).values,
        member_table.std(axis=1).fillna(0.0).values,
        member_table,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_parity_plot(
    actual: Sequence[float],
    predicted: Sequence[float],
    output_path: str,
    title: str,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Plot] matplotlib is unavailable; plots were skipped.")
        return
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
    actual_array = actual_array[valid]
    predicted_array = predicted_array[valid]
    if len(actual_array) == 0:
        return
    metrics = metrics or metric_dict(actual_array, predicted_array)
    lower = min(float(np.min(actual_array)), float(np.min(predicted_array)))
    upper = max(float(np.max(actual_array)), float(np.max(predicted_array)))
    padding = max((upper - lower) * 0.05, 1e-6)

    fig = plt.figure(figsize=(5.4, 5.0))
    plt.scatter(actual_array, predicted_array, alpha=0.8)
    plt.plot([lower - padding, upper + padding], [lower - padding, upper + padding], linestyle="--")
    plt.xlabel("Experimental target")
    plt.ylabel("Predicted target")
    plt.title(title)
    text = (
        f"R² = {metrics.get('R2', np.nan):.3f}\n"
        f"RMSE = {metrics.get('RMSE', np.nan):.3f}\n"
        f"MAE = {metrics.get('MAE', np.nan):.3f}\n"
        f"Spearman = {metrics.get('Spearman', np.nan):.3f}"
    )
    plt.text(0.04, 0.96, text, transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_residual_plot(actual: Sequence[float], predicted: Sequence[float], output_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residual = actual_array - predicted_array
    valid = np.isfinite(predicted_array) & np.isfinite(residual)
    if not np.any(valid):
        return
    fig = plt.figure(figsize=(5.6, 4.4))
    plt.scatter(predicted_array[valid], residual[valid], alpha=0.8)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted target")
    plt.ylabel("Residual (experimental - predicted)")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# External prospective validation
# -----------------------------------------------------------------------------
def prepare_external_actual(
    original_df: pd.DataFrame,
    column_name: str,
    scale: str,
    training_target_mode: str,
) -> pd.Series:
    source = find_column(original_df, [column_name])
    if source is None:
        raise ValueError(f"External actual column '{column_name}' was not found.")
    values = pd.to_numeric(original_df[source], errors="coerce")

    if scale == "same_target":
        return values
    if scale == "log10_raw":
        if training_target_mode != "log10_raw":
            raise ValueError(
                "External scale 'log10_raw' can only be used when the training target mode is also log10_raw. "
                "For a normalized training target, first compute the identical normalized external target "
                "and use --external-actual-scale same_target."
            )
        return np.log10(values.clip(lower=1e-12))
    raise ValueError(f"Unsupported external actual scale: {scale}")


def run_external_validation(
    external_path: str,
    sheet_name: Any,
    actual_column: str,
    actual_scale: str,
    training_target_mode: str,
    qc: QCConfig,
    fitted_models: Sequence[FittedModel],
    morgan_map: Dict[str, np.ndarray],
    morgan_n_bits: int,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = pd.read_excel(external_path, sheet_name=sheet_name)
    standardized, _ = standardize_formulation_columns(original, require_target=False)
    standardized["external_actual"] = prepare_external_actual(
        original, actual_column, actual_scale, training_target_mode
    ).values

    valid_target = standardized["external_actual"].notna()
    standardized = standardized.loc[valid_target].reset_index(drop=True)
    if len(standardized) < 2:
        raise ValueError("External validation requires at least two rows with actual values.")

    standardized, invalid_removed = drop_invalid_rows(standardized)
    standardized, qc_removed = apply_qc_filter(standardized, qc)
    if len(standardized) < 2:
        raise ValueError("Fewer than two external samples remained after cleaning/QC.")

    X_core, X_aux = build_feature_blocks(standardized, morgan_map, n_bits=morgan_n_bits)
    prediction, prediction_sd, member_table = predict_with_final_models(fitted_models, X_core, X_aux)

    result = standardized.copy()
    result["predicted_target"] = prediction
    result["prediction_sd_across_models"] = prediction_sd
    result["residual"] = result["external_actual"] - result["predicted_target"]
    result["absolute_error"] = result["residual"].abs()
    for column in member_table.columns:
        result[f"pred_{column}"] = member_table[column].values

    metrics = metric_dict(result["external_actual"].values, result["predicted_target"].values)
    metrics_table = pd.DataFrame([{
        "validation_set": os.path.basename(external_path),
        "n": len(result),
        **metrics,
        "actual_column": actual_column,
        "actual_scale": actual_scale,
    }])

    save_parity_plot(
        result["external_actual"], result["predicted_target"],
        os.path.join(output_dir, "external_validation_parity.png"),
        "Prospective external validation", metrics,
    )
    save_residual_plot(
        result["external_actual"], result["predicted_target"],
        os.path.join(output_dir, "external_validation_residuals.png"),
        "Prospective external validation residuals",
    )
    return result, metrics_table, pd.concat([
        invalid_removed.assign(removal_stage="invalid_or_template"),
        qc_removed.assign(removal_stage="QC"),
    ], ignore_index=True)


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def save_excel(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication-safe nested-CV LNP modeling pipeline.")
    parser.add_argument(
        "--data-path", default="",
        help=(
            "Optional override for the training workbook. Without this argument, "
            "the script automatically uses the DC24/HaCaT modeling Excel file "
            "stored beside the Python script."
        ),
    )
    parser.add_argument("--sheet-name", default="0", help="Training sheet name or zero-based index; default 0.")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument("--target-mode", choices=["normalized", "log10_raw"], default="normalized")
    parser.add_argument("--raw-target-column", default="DC_Cell_Transfection_Efficiency")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--tune-iter", type=int, default=12)
    parser.add_argument("--aux-top-k", type=int, default=8)
    parser.add_argument(
        "--outer-split-mode", choices=["grouped", "random"], default="grouped",
        help="grouped keeps identical formulations in one fold; random reproduces interpolation-style CV.",
    )
    parser.add_argument("--group-round-decimals", type=int, default=4)
    parser.add_argument("--no-optional-boosters", action="store_true")
    parser.add_argument("--skip-round-cv", action="store_true")
    parser.add_argument("--skip-cumulative-round-analysis", action="store_true")
    parser.add_argument("--pdi-max", type=float, default=0.5)
    parser.add_argument("--size-min", type=float, default=30.0)
    parser.add_argument("--size-max", type=float, default=300.0)
    parser.add_argument(
        "--allow-missing-qc", action="store_true",
        help="Keep rows with missing size/PDI. Publication default is to exclude missing QC values.",
    )
    parser.add_argument("--external-path", default="", help="Optional prospective validation workbook.")
    parser.add_argument("--external-sheet", default="0", help="External sheet name or index.")
    parser.add_argument("--external-actual-column", default="actual_normolized_for_DC2.4")
    parser.add_argument("--external-actual-scale", choices=["same_target", "log10_raw"], default="same_target")
    return parser.parse_args()


def parse_sheet_argument(value: Any) -> Any:
    text = str(value)
    return int(text) if text.isdigit() else text


def main() -> None:
    args = parse_args()
    start_time = time.time()

    config = CVConfig(
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        inner_folds=args.inner_folds,
        tune_iter=args.tune_iter,
        auxiliary_top_k=args.aux_top_k,
        split_mode=args.outer_split_mode,
        group_round_decimals=args.group_round_decimals,
    )
    qc = QCConfig(
        args.pdi_max, args.size_min, args.size_max,
        require_complete=not args.allow_missing_qc,
    )

    data_path = find_existing_data_path(args.data_path or None)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(data_path),
        "lnp_outputs",
        "publication_safe_nestedCV_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Version] {SCRIPT_VERSION}")
    print(f"[Data] {data_path}")
    print(f"[Output] {output_dir}")

    training_sheet = parse_sheet_argument(args.sheet_name)
    original = pd.read_excel(data_path, sheet_name=training_sheet)
    standardized, source_columns = standardize_formulation_columns(original)
    standardized, target_label = attach_training_target(
        standardized, original, args.target_mode, args.raw_target_column
    )
    standardized, invalid_removed = drop_invalid_rows(standardized)
    standardized, qc_removed = apply_qc_filter(standardized, qc)
    standardized = fill_missing_round_labels(standardized)
    if len(standardized) < 20:
        raise ValueError(f"Only {len(standardized)} training rows remained; too few for this pipeline.")

    y = standardized["TE"].astype(float).reset_index(drop=True)
    standardized = standardized.reset_index(drop=True)
    sample_ids = standardized["candidate_id"].astype(str).tolist()
    formulation_groups, formulation_signatures = make_formulation_groups(
        standardized, config.group_round_decimals
    )
    standardized["formulation_signature"] = formulation_signatures.values
    standardized["formulation_group"] = formulation_groups
    print(
        f"[Target] {target_label} | N={len(y)} | mean={y.mean():.4f} | "
        f"SD={y.std(ddof=1):.4f} | min={y.min():.4f} | max={y.max():.4f}"
    )

    smiles_map = load_smiles_map(data_path)
    morgan_map, morgan_n_bits, morgan_source = prepare_morgan_features(
        data_path, smiles_map, requested_bits=128, radius=2
    )
    X_core, X_aux = build_feature_blocks(standardized, morgan_map, n_bits=morgan_n_bits)
    print(f"[Morgan] source={morgan_source} | n_bits={morgan_n_bits} | n_lipids={len(morgan_map)}")

    specs = model_specs(include_optional_boosters=not args.no_optional_boosters)
    print(f"[Models] {list(specs)}")
    outer_groups = formulation_groups if config.split_mode == "grouped" else None
    outer_splits = make_outer_splits(len(y), config, groups=outer_groups)
    print(
        f"[NestedCV] mode={config.split_mode} | outer={config.outer_folds} folds x {config.outer_repeats} repeats "
        f"({len(outer_splits)} outer fits/model), inner={config.inner_folds}, tune_iter={config.tune_iter}"
    )

    all_oof_frames: List[pd.DataFrame] = []
    all_fold_frames: List[pd.DataFrame] = []
    all_feature_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for model_name, (estimator, param_dist) in specs.items():
        oof, fold_metrics, fold_features = nested_cv_single_model(
            model_name, estimator, param_dist,
            X_core, X_aux, y, sample_ids, outer_splits, config,
            formulation_groups=formulation_groups if config.split_mode == "grouped" else None,
        )
        all_oof_frames.append(oof)
        all_fold_frames.append(fold_metrics)
        all_feature_frames.append(fold_features)
        summary_rows.append(summarize_oof(oof, fold_metrics))

    all_oof = pd.concat(all_oof_frames, ignore_index=True)
    all_fold_metrics = pd.concat(all_fold_frames, ignore_index=True)
    all_fold_features = pd.concat(all_feature_frames, ignore_index=True) if all_feature_frames else pd.DataFrame()

    model_names = list(specs)
    ensemble_members = select_primary_ensemble_members(model_names)
    if len(ensemble_members) < 2:
        raise RuntimeError("Fewer than two primary ensemble members were available.")
    ensemble_oof = build_ensemble_oof(all_oof, ensemble_members)
    ensemble_metrics = metric_dict(ensemble_oof["actual"], ensemble_oof["predicted"])
    summary_rows.append({
        "model": "PrimaryTreeEnsemble",
        **ensemble_metrics,
        "Fold_R2_mean": np.nan,
        "Fold_R2_std": np.nan,
        "Fold_RMSE_mean": np.nan,
        "Fold_RMSE_std": np.nan,
        "Fold_Spearman_mean": np.nan,
        "Fold_Spearman_std": np.nan,
        "Mean_overfit_gap": np.nan,
    })
    all_oof = pd.concat([all_oof, ensemble_oof], ignore_index=True, sort=False)

    summary = pd.DataFrame(summary_rows)
    summary["publication_score"] = (
        0.55 * summary["R2"].fillna(-1).clip(-1, 1)
        + 0.30 * summary["Spearman"].fillna(-1).clip(-1, 1)
        + 0.15 * (1.0 / (1.0 + summary["RMSE"].clip(lower=0)))
    )
    summary = summary.sort_values("publication_score", ascending=False).reset_index(drop=True)
    print("\n[Honest nested-CV summary]")
    print(summary.to_string(index=False))
    print(f"\n[Primary ensemble] {ensemble_members}")

    save_parity_plot(
        ensemble_oof["actual"], ensemble_oof["predicted"],
        os.path.join(output_dir, "nested_cv_primary_ensemble_parity.png"),
        "Nested CV: primary tree ensemble", ensemble_metrics,
    )
    save_residual_plot(
        ensemble_oof["actual"], ensemble_oof["predicted"],
        os.path.join(output_dir, "nested_cv_primary_ensemble_residuals.png"),
        "Nested CV residuals: primary tree ensemble",
    )

    cumulative_summary = pd.DataFrame()
    cumulative_oof = pd.DataFrame()
    cumulative_folds = pd.DataFrame()
    cumulative_features = pd.DataFrame()
    if not args.skip_cumulative_round_analysis:
        cumulative_summary, cumulative_oof, cumulative_folds, cumulative_features = cumulative_round_nested_cv(
            standardized, X_core, X_aux, y, specs, ensemble_members, config, output_dir,
            full_summary=summary,
            full_oof=all_oof,
            full_fold_metrics=all_fold_metrics,
            full_fold_features=all_fold_features,
        )

    round_predictions = pd.DataFrame()
    round_metrics = pd.DataFrame()
    if not args.skip_round_cv:
        round_predictions, round_metrics = leave_one_round_out_ensemble(
            standardized, X_core, X_aux, y, specs, ensemble_members, config
        )
        if len(round_predictions):
            round_overall = metric_dict(round_predictions["actual"], round_predictions["predicted"])
            save_parity_plot(
                round_predictions["actual"], round_predictions["predicted"],
                os.path.join(output_dir, "leave_one_round_out_parity.png"),
                "Leave-one-round-out validation", round_overall,
            )

    fitted_models, final_aux_features = fit_final_models(
        specs, ensemble_members, X_core, X_aux, y, config,
        formulation_groups=formulation_groups,
    )
    bundle = {
        "script_version": SCRIPT_VERSION,
        "target_mode": args.target_mode,
        "target_label": target_label,
        "core_feature_columns": list(X_core.columns),
        "auxiliary_feature_columns": list(X_aux.columns),
        "selected_auxiliary_features": final_aux_features,
        "ensemble_members": ensemble_members,
        "fitted_models": fitted_models,
        "morgan_map": morgan_map,
        "morgan_n_bits": morgan_n_bits,
        "morgan_source": morgan_source,
        "qc": qc,
        "cv_config": config,
    }
    joblib.dump(bundle, os.path.join(output_dir, "final_model_bundle.joblib"))

    external_result = pd.DataFrame()
    external_metrics = pd.DataFrame()
    external_removed = pd.DataFrame()
    if args.external_path:
        external_result, external_metrics, external_removed = run_external_validation(
            external_path=args.external_path,
            sheet_name=parse_sheet_argument(args.external_sheet),
            actual_column=args.external_actual_column,
            actual_scale=args.external_actual_scale,
            training_target_mode=args.target_mode,
            qc=qc,
            fitted_models=fitted_models,
            morgan_map=morgan_map,
            morgan_n_bits=morgan_n_bits,
            output_dir=output_dir,
        )
        print("\n[Prospective external validation]")
        print(external_metrics.to_string(index=False))

    feature_frequency = pd.DataFrame()
    if len(all_fold_features):
        feature_frequency = (
            all_fold_features.groupby(["model", "feature"], as_index=False)
            .agg(selection_count=("fold", "count"), mean_rank=("aux_rank", "mean"))
            .sort_values(["model", "selection_count", "mean_rank"], ascending=[True, False, True])
        )

    run_info = pd.DataFrame([
        {"item": "script_version", "value": SCRIPT_VERSION},
        {"item": "data_path", "value": data_path},
        {"item": "target_mode", "value": args.target_mode},
        {"item": "target_label", "value": target_label},
        {"item": "n_training_after_cleaning_QC", "value": len(standardized)},
        {"item": "outer_folds", "value": config.outer_folds},
        {"item": "outer_repeats", "value": config.outer_repeats},
        {"item": "outer_split_mode", "value": config.split_mode},
        {"item": "n_exact_formulation_groups", "value": len(np.unique(formulation_groups))},
        {"item": "qc_require_complete", "value": qc.require_complete},
        {"item": "inner_folds", "value": config.inner_folds},
        {"item": "tune_iter", "value": config.tune_iter},
        {"item": "core_feature_count", "value": X_core.shape[1]},
        {"item": "morgan_source", "value": morgan_source},
        {"item": "morgan_n_bits", "value": morgan_n_bits},
        {"item": "morgan_lipid_count", "value": len(morgan_map)},
        {"item": "cumulative_round_analysis", "value": not args.skip_cumulative_round_analysis},
        {"item": "auxiliary_top_k", "value": config.auxiliary_top_k},
        {"item": "primary_ensemble_members", "value": ",".join(ensemble_members)},
        {"item": "runtime_minutes", "value": (time.time() - start_time) / 60.0},
        {
            "item": "external_scale_rule",
            "value": "External RMSE is valid only when external actual values use the identical training-target scale.",
        },
    ])

    save_excel(os.path.join(output_dir, "publication_safe_model_results.xlsx"), {
        "model_summary": summary,
        "nested_cv_oof": all_oof,
        "nested_cv_folds": all_fold_metrics,
        "fold_selected_aux": all_fold_features,
        "aux_selection_frequency": feature_frequency,
        "cumulative_round_summary": cumulative_summary,
        "cumulative_round_oof": cumulative_oof,
        "cumulative_round_folds": cumulative_folds,
        "cumulative_round_features": cumulative_features,
        "round_cv_predictions": round_predictions,
        "round_cv_metrics": round_metrics,
        "external_validation": external_result,
        "external_metrics": external_metrics,
        "external_removed": external_removed,
        "training_data_used": standardized,
        "invalid_removed": invalid_removed,
        "qc_removed": qc_removed,
        "final_aux_features": pd.DataFrame({"selected_auxiliary_feature": final_aux_features}),
        "run_info": run_info,
    })

    summary.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False, encoding="utf-8-sig")
    ensemble_oof.to_csv(os.path.join(output_dir, "nested_cv_primary_ensemble_oof.csv"), index=False, encoding="utf-8-sig")
    if len(cumulative_summary):
        cumulative_summary.to_csv(
            os.path.join(output_dir, "cumulative_round_model_summary.csv"),
            index=False, encoding="utf-8-sig",
        )
        cumulative_summary[
            cumulative_summary["model"] == "PrimaryTreeEnsemble"
        ].to_csv(
            os.path.join(output_dir, "cumulative_round_primary_ensemble_summary.csv"),
            index=False, encoding="utf-8-sig",
        )
    if len(external_result):
        external_result.to_csv(os.path.join(output_dir, "external_validation_predictions.csv"), index=False, encoding="utf-8-sig")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "target_mode": args.target_mode,
        "target_label": target_label,
        "n_training": len(standardized),
        "outer_split_mode": config.split_mode,
        "n_exact_formulation_groups": int(len(np.unique(formulation_groups))),
        "primary_ensemble_members": ensemble_members,
        "primary_nested_cv_metrics": json_safe(ensemble_metrics),
        "morgan_source": morgan_source,
        "morgan_n_bits": morgan_n_bits,
        "cumulative_round_stages": (
            cumulative_summary["stage"].drop_duplicates().tolist()
            if len(cumulative_summary) else []
        ),
        "selected_auxiliary_features_final": final_aux_features,
        "output_dir": os.path.abspath(output_dir),
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("[Done] Publication-safe nested-CV analysis completed.")
    print(f"Results: {os.path.join(output_dir, 'publication_safe_model_results.xlsx')}")
    print(f"Model bundle: {os.path.join(output_dir, 'final_model_bundle.joblib')}")
    print(f"Runtime: {(time.time() - start_time) / 60.0:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 88)
        print("[ProgramError]")
        print(type(exc).__name__, ":", exc)
        print("=" * 88)
        raise
