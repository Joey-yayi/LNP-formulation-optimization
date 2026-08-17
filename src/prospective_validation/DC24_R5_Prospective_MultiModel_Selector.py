# -*- coding: utf-8 -*-
"""
DC24_R5_Prospective_MultiModel_Selector.py
==========================================

Prospective R5 candidate selection for the DC2.4 mRNA-LNP project.

Scientific purpose
------------------
This script is NOT another retrospective benchmark.  It is a deployment / candidate-
selection script intended to be run BEFORE the R5 wet-lab experiment.

It freezes the R1-R4 target-specific model, generates or reads a feasible candidate pool,
predicts the SAME candidate pool with several strong model families, and selects a small
prospective panel that can test both:

1) whether the model can enrich for genuinely high-transfection formulations; and
2) whether it can rank unseen formulations across high / intermediate / low response.

Default R5 panel (12 NEW formulations)
---------------------------------------
    H1-H5 : 5 high-confidence, multi-model consensus-high candidates
    H6-H7 : 2 consensus-high but deliberately more compositionally diverse candidates
    M1-M2 : 2 genuinely intermediate candidates selected using BOTH percentile rank
            and absolute multi-model mean prediction (default target ≈0.65)
    L1-L2 : 2 low predicted controls selected around a low but in-domain prediction
            level (default target ≈0.42), excluding duplicated IL1=IL2 identities
    D1    : 1 controlled model-disagreement / information-rich candidate

Thus 7/12 candidates are intentionally selected from the high-predicted region, while the
remaining 5 preserve enough response range to test prospective ranking.

Model families
--------------
Tree models (fit once on the frozen R1-R4 data using the same feature engineering as the
publication pipeline):
    RandomForest, XGBoost, GradientBoosting, HistGradientBoosting, LightGBM

Foundation/tabular models:
    TabPFN, TabFM

The script combines models using PERCENTILE RANKS across the same prospective candidate
pool rather than raw prediction averages.  This is important because different model
families can have different calibration/compression of the normalized response scale.

Important publication guardrails
--------------------------------
* R5 measured outcomes must NOT be used anywhere in candidate selection.
* Run this script, save/hash the frozen prediction key, then perform the wet-lab experiment.
* Keep R5 experimental conditions as close as possible to R1-R4 (cell state, P1415, mRNA,
  LNP preparation, dialysis/centrifugation, storage interval, dose and readout timing).
* For n≈12, Spearman / Top-k enrichment / high-hit rate are generally more stable headline
  validation quantities than R² alone.  R²/RMSE/MAE may still be reported when the measured
  R5 target is on exactly the same normalization scale.

Environment strategy
--------------------
The parent process is expected to run in the user's normal tree-model environment.
It fits the tree models first.  If TabPFN/TabFM are requested, the script automatically
looks for an existing `.venv_tabfm` environment and launches a small CHILD process there.
Only the already-created numeric R1-R4 feature matrix and candidate feature matrix are sent
to that child.  This avoids mixing incompatible NumPy/PyTorch environments.

TabPFN privacy note
-------------------
If the TabPFN client backend is used, the numeric R1-R4 training features/targets and R5
candidate features are sent to Prior Labs for hosted inference.  Use --tabpfn-backend local
or --skip-tabpfn when remote transfer is unsuitable.  No token is hard-coded in this file.

Typical direct run
------------------
    python DC24_R5_Prospective_MultiModel_Selector.py

Explicit paths
--------------
    python DC24_R5_Prospective_MultiModel_Selector.py ^
      --train-excel "C:\\path\\R1-4 all LNP normalized 1.35 (new).xlsx" ^
      --tree-script "C:\\path\\DC24_R1R4_ModelSelection_Publication_DirectRun.py"

Use a pre-designed candidate workbook instead of virtual generation
---------------------------------------------------------------
    python DC24_R5_Prospective_MultiModel_Selector.py ^
      --candidate-source excel ^
      --candidate-excel "C:\\path\\R5_candidate_pool.xlsx"

Optional post-experiment evaluation
-----------------------------------
After the experiment, add a column with the SAME normalized DC2.4 target and run:

    python DC24_R5_Prospective_MultiModel_Selector.py ^
      --evaluate-only ^
      --frozen-dir "C:\\path\\R5_prospective_YYYYMMDD_HHMMSS" ^
      --actual-excel "C:\\path\\R5_measured.xlsx" ^
      --actual-column "Normalized for DC2.4"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

warnings.filterwarnings("once")

SEED = 20260817
rng_global = np.random.default_rng(SEED)

SCRIPT_VERSION = "1.1.0-R5-prospective-multimodel-consensus-validation-balanced"
TARGET_COLUMN = "Normalized for DC2.4"
DEFAULT_SHEET = "Round1&2&3&4"

# Model families pre-specified for R5 candidate ranking.  These are the families that
# were approximately R² >= 0.68 in the final R1-R4 retrospective benchmark.
# ExtraTrees and Ridge are intentionally not individual R5 consensus voters here.
TREE_MODELS_WANTED = [
    "RandomForest",
    "XGBoost",
    "GradientBoosting",
    "HistGradientBoosting",
    "LightGBM",
]
FOUNDATION_MODELS_WANTED = ["TabPFN", "TabFM"]

# Broad wet-lab feasibility limits used previously in this project.  Candidate generation
# is additionally clipped by the observed R1-R4 distribution, so these are outer guards.
HARD_TOTAL_IL_MIN = 20.0
HARD_TOTAL_IL_MAX = 67.0
HARD_HL_MIN = 5.0
HARD_HL_MAX = 50.0
HARD_CHOL_MIN = 0.0
HARD_CHOL_MAX = 58.0
HARD_PEG_MIN = 1.0
HARD_PEG_MAX = 4.0

FORMULATION_COLUMNS = [
    "candidate_id",
    "IL1", "IL1_molpct",
    "IL2", "IL2_molpct",
    "Phospholipid", "HL_molpct",
    "CHOL_molpct",
    "PEG", "PEG_molpct",
    "NP_ratio",
]

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "candidate_id": ["candidate_id", "Candidate_ID", "Selection_Order", "Formulation_ID", "Sample", "ID", "No", "NO", "编号", "配方编号"],
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
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        if isinstance(x, str):
            s = re.sub(r"[^0-9.\-eE]", "", x)
            if s in {"", "-", ".", "-."}:
                return default
            return float(s)
        return float(x)
    except Exception:
        return default


def percentile_rank(series: pd.Series) -> pd.Series:
    """0=lowest, 1=highest; average ranks for ties."""
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True)


def parse_sheet(value: Any) -> Any:
    text = str(value)
    return int(text) if text.isdigit() else text


def find_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    normalized = {str(c).strip().casefold(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        key = str(n).strip().casefold()
        if key in normalized:
            return normalized[key]
    return None


def standardize_candidate_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a user-supplied prospective candidate workbook onto pipeline column names."""
    out = pd.DataFrame(index=raw.index)
    for canonical, aliases in COLUMN_ALIASES.items():
        source = find_col(raw, aliases)
        if source is None:
            if canonical == "candidate_id":
                out[canonical] = [f"POOL-{i+1:05d}" for i in range(len(raw))]
            elif canonical == "NP_ratio":
                out[canonical] = np.nan
            else:
                out[canonical] = np.nan
        else:
            out[canonical] = raw[source].values

    for c in ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct", "NP_ratio"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["IL1", "IL2", "Phospholipid", "PEG", "candidate_id"]:
        out[c] = out[c].astype(object)
    return out.reset_index(drop=True)


def _default_home_candidates(filename: str) -> List[Path]:
    home = Path.home()
    roots = [
        home / "Desktop",
        home / "桌面",
        home / "Downloads",
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "桌面",
    ]
    direct = []
    for root in roots:
        direct.extend([
            root / filename,
            root / "AI screen LNP python and excel" / "PyCharm 有效代码" / "8.02 publish" / filename,
            root / "AI screen LNP python and excel" / "PyCharm 有效代码" / filename,
        ])
    return direct


def resolve_training_workbook(user_path: str) -> str:
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Training workbook does not exist: {p}")

    filename = "R1-4 all LNP normalized 1.35 (new).xlsx"
    for p in _default_home_candidates(filename):
        if p.is_file():
            return str(p.resolve())

    # Bounded recursive search in likely project roots.
    home = Path.home()
    roots = [home / "Desktop", home / "桌面", home / "Downloads"]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            hits = list(root.glob(f"**/{filename}"))
        except Exception:
            hits = []
        if hits:
            return str(max(hits, key=lambda x: x.stat().st_mtime).resolve())
    raise FileNotFoundError(
        f"Could not find {filename!r}. Pass it explicitly with --train-excel."
    )


def resolve_tree_script(user_path: str) -> str:
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Tree pipeline script does not exist: {p}")

    filenames = [
        "DC24_R1R4_ModelSelection_Publication_DirectRun.py",
        "R1-4_DC2.4_tree_model_compare_R2.py",
        "DC24_R1R4_NORMALIZED1_35_PUBLICATION_FIXED.py",
        "DC24_R1R4_ModelSelection_Publication_GitHub.py",
    ]
    found: List[Path] = []
    for name in filenames:
        for p in _default_home_candidates(name):
            if p.is_file():
                found.append(p)
    if found:
        # Prefer list order, then most recent among same filename.
        for name in filenames:
            same = [p for p in found if p.name == name]
            if same:
                return str(max(same, key=lambda x: x.stat().st_mtime).resolve())

    home = Path.home()
    roots = [home / "Desktop", home / "桌面", home / "Downloads"]
    for name in filenames:
        for root in roots:
            if not root.is_dir():
                continue
            try:
                hits = list(root.glob(f"**/{name}"))
            except Exception:
                hits = []
            if hits:
                return str(max(hits, key=lambda x: x.stat().st_mtime).resolve())

    raise FileNotFoundError(
        "Could not find the final R1-R4 tree-model script. Pass it with --tree-script."
    )


def load_pipeline_module(path: str):
    spec = importlib.util.spec_from_file_location("dc24_r1r4_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dc24_r1r4_pipeline"] = mod
    spec.loader.exec_module(mod)
    required = [
        "standardize_formulation_columns",
        "attach_training_target",
        "drop_invalid_rows",
        "apply_qc_filter",
        "fill_missing_round_labels",
        "make_formulation_groups",
        "load_smiles_map",
        "prepare_morgan_features",
        "build_feature_blocks",
        "mrmr_select_auxiliary",
        "combine_selected_features",
        "model_specs",
        "fit_tuned_estimator",
        "formulation_signature",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(
            "Selected tree pipeline is incompatible with R5 selector. Missing: "
            + ", ".join(missing)
        )
    return mod


# -----------------------------------------------------------------------------
# Training data + exact feature representation
# -----------------------------------------------------------------------------
@dataclass
class TrainingContext:
    pipeline_path: str
    workbook_path: str
    sheet_name: str
    target_label: str
    standardized: pd.DataFrame
    y: pd.Series
    groups: np.ndarray
    signatures: pd.Series
    X_core: pd.DataFrame
    X_aux: pd.DataFrame
    selected_aux: List[str]
    X_selected: pd.DataFrame
    morgan_map: Dict[str, Any]
    morgan_n_bits: int
    morgan_source: str
    config: Any


def resolve_training_sheet(pipeline, workbook: str, requested: Any) -> Any:
    if hasattr(pipeline, "resolve_training_sheet_name"):
        return pipeline.resolve_training_sheet_name(workbook, requested)
    xls = pd.ExcelFile(workbook)
    req = str(requested).strip()
    exact = [s for s in xls.sheet_names if str(s).strip() == req]
    if len(exact) == 1:
        return exact[0]
    folded = [s for s in xls.sheet_names if re.sub(r"\s+", "", str(s)).casefold() == re.sub(r"\s+", "", req).casefold()]
    if len(folded) == 1:
        return folded[0]
    if isinstance(requested, int):
        return requested
    raise ValueError(f"Could not uniquely resolve sheet {requested!r}. Available: {xls.sheet_names}")


def build_training_context(args: argparse.Namespace) -> Tuple[Any, TrainingContext]:
    workbook = resolve_training_workbook(args.train_excel)
    tree_script = resolve_tree_script(args.tree_script)
    pipeline = load_pipeline_module(tree_script)

    sheet = resolve_training_sheet(pipeline, workbook, parse_sheet(args.sheet_name))
    original = pd.read_excel(workbook, sheet_name=sheet)
    exact = find_col(original, [TARGET_COLUMN])
    if exact is None:
        raise ValueError(
            f"Exact target {TARGET_COLUMN!r} is absent from sheet {sheet!r}. "
            "R5 selection is intentionally locked to the final 1.35 DC2.4 target."
        )

    standardized, _ = pipeline.standardize_formulation_columns(original)
    standardized, target_label = pipeline.attach_training_target(
        standardized, original, "normalized", "DC_Cell_Transfection_Efficiency"
    )

    qc = pipeline.QCConfig(
        pdi_max=args.pdi_max,
        size_min=args.size_min,
        size_max=args.size_max,
        require_complete=not args.allow_missing_qc,
    )
    standardized, _ = pipeline.drop_invalid_rows(standardized)
    standardized, _ = pipeline.apply_qc_filter(standardized, qc)
    standardized = pipeline.fill_missing_round_labels(standardized)
    standardized = standardized.reset_index(drop=True)

    # Freeze to R1-R4 only. If round labels are unavailable in a particular imported
    # module, the exact 104 QC-passed target rows are still used; no R5 response exists
    # in this workbook by design.
    if "Round" in standardized.columns and standardized["Round"].astype(str).str.match(r"R[1-4]", na=False).any():
        keep = standardized["Round"].astype(str).str.match(r"R[1-4]", na=False)
        standardized = standardized.loc[keep].reset_index(drop=True)

    if len(standardized) < 80:
        raise ValueError(f"Only {len(standardized)} R1-R4 rows remained. Expected approximately 104.")

    y = pd.to_numeric(standardized["TE"], errors="coerce")
    valid = y.notna()
    standardized = standardized.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True).astype(float)

    groups, signatures = pipeline.make_formulation_groups(standardized, args.group_round_decimals)
    standardized["formulation_signature"] = signatures.values
    standardized["formulation_group"] = groups

    smiles_map = pipeline.load_smiles_map(workbook)
    morgan_map, morgan_n_bits, morgan_source = pipeline.prepare_morgan_features(
        workbook, smiles_map, requested_bits=128, radius=2
    )
    X_core, X_aux = pipeline.build_feature_blocks(
        standardized, morgan_map, n_bits=morgan_n_bits
    )
    X_core = X_core.reset_index(drop=True)
    X_aux = X_aux.reset_index(drop=True)

    selected_aux = pipeline.mrmr_select_auxiliary(X_aux, y, k=args.aux_top_k)
    X_selected = pipeline.combine_selected_features(
        X_core, X_aux, np.arange(len(y)), selected_aux
    ).reset_index(drop=True)

    config = pipeline.CVConfig(
        outer_folds=5,
        outer_repeats=1,
        inner_folds=args.inner_folds,
        tune_iter=args.tune_iter,
        auxiliary_top_k=args.aux_top_k,
        split_mode="grouped",
        group_round_decimals=args.group_round_decimals,
        random_state=args.seed,
    )

    print("\n" + "=" * 92)
    print("[R5 TRAINING SNAPSHOT]")
    print(f"Workbook : {workbook}")
    print(f"Sheet    : {sheet!r}")
    print(f"Pipeline : {tree_script}")
    print(f"Target   : {target_label}")
    print(f"Rows     : {len(standardized)} | formulation groups={len(np.unique(groups))}")
    print(f"Features : core={X_core.shape[1]} | aux={X_aux.shape[1]} | selected_aux={len(selected_aux)} | final={X_selected.shape[1]}")
    print(f"Morgan   : source={morgan_source} | bits={morgan_n_bits}")
    print("=" * 92)

    return pipeline, TrainingContext(
        pipeline_path=tree_script,
        workbook_path=workbook,
        sheet_name=str(sheet),
        target_label=str(target_label),
        standardized=standardized,
        y=y,
        groups=np.asarray(groups),
        signatures=signatures.reset_index(drop=True),
        X_core=X_core,
        X_aux=X_aux,
        selected_aux=list(selected_aux),
        X_selected=X_selected,
        morgan_map=morgan_map,
        morgan_n_bits=int(morgan_n_bits),
        morgan_source=str(morgan_source),
        config=config,
    )


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------
def _clip_observed(values: pd.Series, hard_lo: float, hard_hi: float, qlo: float = 0.02, qhi: float = 0.98) -> Tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(x) < 5:
        return hard_lo, hard_hi
    lo = max(hard_lo, float(x.quantile(qlo)))
    hi = min(hard_hi, float(x.quantile(qhi)))
    if lo >= hi:
        return hard_lo, hard_hi
    return lo, hi


def _weighted_choice(values: Sequence[Any], weights: Optional[Sequence[float]], rng: np.random.Generator) -> Any:
    vals = list(values)
    if not vals:
        return None
    if weights is None:
        return vals[int(rng.integers(0, len(vals)))]
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return vals[int(rng.integers(0, len(vals)))]
    w = w / w.sum()
    return vals[int(rng.choice(len(vals), p=w))]


def _round_half(x: float) -> float:
    return round(float(x) * 2.0) / 2.0


def _make_candidate(
    il1: Any, il2: Any, total_il: float, il1_fraction: float,
    hl: Any, hl_pct: float, peg: Any, peg_pct: float,
    np_ratio: float, source: str, anchor_id: str = "", anchor_y: float = np.nan,
) -> Optional[Dict[str, Any]]:
    total_il = float(np.clip(total_il, HARD_TOTAL_IL_MIN, HARD_TOTAL_IL_MAX))
    hl_pct = float(np.clip(hl_pct, HARD_HL_MIN, HARD_HL_MAX))
    peg_pct = float(np.clip(peg_pct, HARD_PEG_MIN, HARD_PEG_MAX))

    il2_text = "" if il2 is None else str(il2).strip()
    single = il2 is None or il2_text.casefold() in {"", "none", "nan", "na", "-"}
    if single:
        il1_pct = total_il
        il2_pct = 0.0
        il2 = None
    else:
        frac = float(np.clip(il1_fraction, 0.15, 0.85))
        il1_pct = total_il * frac
        il2_pct = total_il - il1_pct

    il1_pct = _round_half(il1_pct)
    il2_pct = _round_half(il2_pct)
    total_il = il1_pct + il2_pct
    hl_pct = _round_half(hl_pct)
    peg_pct = round(float(peg_pct) * 4.0) / 4.0
    chol = round((100.0 - total_il - hl_pct - peg_pct) * 2.0) / 2.0

    # Re-close mixture exactly after rounding by adjusting cholesterol.
    chol = round((100.0 - total_il - hl_pct - peg_pct) * 2.0) / 2.0
    if not (HARD_CHOL_MIN <= chol <= HARD_CHOL_MAX):
        return None
    if not (HARD_TOTAL_IL_MIN <= total_il <= HARD_TOTAL_IL_MAX):
        return None

    return {
        "candidate_id": "",
        "IL1": il1,
        "IL1_molpct": il1_pct,
        "IL2": il2,
        "IL2_molpct": il2_pct,
        "Phospholipid": hl,
        "HL_molpct": hl_pct,
        "CHOL_molpct": chol,
        "PEG": peg,
        "PEG_molpct": peg_pct,
        "NP_ratio": np_ratio,
        "candidate_source": source,
        "anchor_id": anchor_id,
        "anchor_TE": anchor_y,
    }


def generate_virtual_pool(train: pd.DataFrame, y: pd.Series, n_target: int, seed: int) -> pd.DataFrame:
    """Generate a mixed local-refinement + broad feasible prospective library."""
    rng = np.random.default_rng(seed)
    df = train.copy().reset_index(drop=True)
    df["_y"] = y.values
    df["_total_il"] = pd.to_numeric(df["IL1_molpct"], errors="coerce").fillna(0) + pd.to_numeric(df["IL2_molpct"], errors="coerce").fillna(0)

    total_lo, total_hi = _clip_observed(df["_total_il"], HARD_TOTAL_IL_MIN, HARD_TOTAL_IL_MAX)
    hl_lo, hl_hi = _clip_observed(df["HL_molpct"], HARD_HL_MIN, HARD_HL_MAX)
    peg_lo, peg_hi = _clip_observed(df["PEG_molpct"], HARD_PEG_MIN, HARD_PEG_MAX)

    # Available chemistry is taken from the actually observed R1-R4 palette.
    il1_values = [x for x in df["IL1"].dropna().astype(str).unique().tolist() if x and x.casefold() != "nan"]
    il2_values = [x for x in df["IL2"].dropna().astype(str).unique().tolist() if x and x.casefold() not in {"nan", "none", ""}]
    hl_values = [x for x in df["Phospholipid"].dropna().astype(str).unique().tolist() if x and x.casefold() != "nan"]
    peg_values = [x for x in df["PEG"].dropna().astype(str).unique().tolist() if x and x.casefold() != "nan"]

    if not il1_values or not hl_values or not peg_values:
        raise ValueError("Training data did not contain a usable lipid / helper / PEG palette.")

    np_values = pd.to_numeric(df.get("NP_ratio", pd.Series(np.nan, index=df.index)), errors="coerce").dropna()
    np_default = float(np_values.median()) if len(np_values) else np.nan

    # Unique top anchors.  These create a dense local library around experimentally
    # successful regions, which is essential because the user explicitly wants >=6 high candidates.
    anchors = df.sort_values("_y", ascending=False).copy()
    anchor_key = anchors.apply(
        lambda r: f"{r.get('IL1')}|{r.get('IL2')}|{r.get('Phospholipid')}|{r.get('PEG')}|"
                  f"{safe_float(r.get('IL1_molpct'),0):.2f}|{safe_float(r.get('IL2_molpct'),0):.2f}|"
                  f"{safe_float(r.get('HL_molpct'),0):.2f}|{safe_float(r.get('CHOL_molpct'),0):.2f}|{safe_float(r.get('PEG_molpct'),0):.2f}",
        axis=1,
    )
    anchors = anchors.loc[~anchor_key.duplicated()].head(min(14, len(anchors))).reset_index(drop=True)
    anchor_weights = np.exp(np.linspace(1.5, 0.0, len(anchors)))

    rows: List[Dict[str, Any]] = []
    n_local = int(round(n_target * 0.65))
    attempts = 0
    while len(rows) < n_local and attempts < n_target * 30:
        attempts += 1
        ai = int(rng.choice(len(anchors), p=anchor_weights / anchor_weights.sum()))
        a = anchors.iloc[ai]
        total0 = safe_float(a.get("_total_il"), np.nan)
        hl0 = safe_float(a.get("HL_molpct"), np.nan)
        peg0 = safe_float(a.get("PEG_molpct"), np.nan)
        if not all(np.isfinite([total0, hl0, peg0])):
            continue
        il1p0 = safe_float(a.get("IL1_molpct"), 0.0)
        il2p0 = safe_float(a.get("IL2_molpct"), 0.0)
        frac0 = il1p0 / max(il1p0 + il2p0, 1e-8)

        total = float(np.clip(total0 + rng.normal(0, 4.0), total_lo, total_hi))
        frac = float(np.clip(frac0 + rng.normal(0, 0.085), 0.15, 0.85))
        hl_pct = float(np.clip(hl0 + rng.normal(0, 4.0), hl_lo, hl_hi))
        peg_pct = float(np.clip(peg0 + rng.normal(0, 0.35), peg_lo, peg_hi))

        hl = a.get("Phospholipid")
        peg = a.get("PEG")
        # A small fraction of local points probes nearby chemistry without turning
        # the high candidate pool into a purely identical family.
        if rng.random() < 0.08 and len(hl_values) > 1:
            hl = _weighted_choice(hl_values, None, rng)
        if rng.random() < 0.08 and len(peg_values) > 1:
            peg = _weighted_choice(peg_values, None, rng)

        row = _make_candidate(
            a.get("IL1"), a.get("IL2"), total, frac,
            hl, hl_pct, peg, peg_pct, np_default,
            source="local_top_refinement",
            anchor_id=str(a.get("candidate_id", "")),
            anchor_y=float(a.get("_y", np.nan)),
        )
        if row is not None:
            rows.append(row)

    # Broad pool: mostly observed IL pair families, with limited novel recombination.
    pair_table = (
        df.assign(
            _il1=df["IL1"].astype(str),
            _il2=df["IL2"].where(df["IL2"].notna(), None),
        )
        .groupby(["_il1", "_il2"], dropna=False)
        .agg(mean_y=("_y", "mean"), n=("_y", "size"))
        .reset_index()
    )
    pair_scores = np.asarray(pair_table["mean_y"], dtype=float)
    pair_scores = np.nan_to_num(pair_scores, nan=np.nanmedian(pair_scores) if len(pair_scores) else 0.5)
    pair_weights = 0.35 + (pair_scores - pair_scores.min()) / max(pair_scores.max() - pair_scores.min(), 1e-9)

    while len(rows) < n_target and attempts < n_target * 80:
        attempts += 1
        if rng.random() < 0.88 and len(pair_table):
            pi = int(rng.choice(len(pair_table), p=pair_weights / pair_weights.sum()))
            il1 = pair_table.iloc[pi]["_il1"]
            il2 = pair_table.iloc[pi]["_il2"]
            if pd.isna(il2) or str(il2).casefold() in {"nan", "none", ""}:
                il2 = None
        else:
            il1 = _weighted_choice(il1_values, None, rng)
            il2 = _weighted_choice(il2_values, None, rng) if (il2_values and rng.random() < 0.85) else None
            # A dual-ionizable formulation must contain two distinct ionizable-lipid
            # identities.  The v1.0 selector allowed a small fraction of IL1 == IL2
            # pairs (e.g. SM102 + SM102).  Those are chemically redundant and make
            # prospective interpretation unnecessarily awkward, so v1.1 removes them.
            if il2 == il1:
                alt = [x for x in il2_values if x != il1]
                il2 = _weighted_choice(alt, None, rng) if alt else None

        total = float(rng.uniform(total_lo, total_hi))
        frac = float(rng.beta(2.2, 2.2))
        frac = float(np.clip(frac, 0.18, 0.82))
        hl_pct = float(rng.uniform(hl_lo, hl_hi))
        peg_pct = float(rng.uniform(peg_lo, peg_hi))
        hl = _weighted_choice(hl_values, None, rng)
        peg = _weighted_choice(peg_values, None, rng)

        row = _make_candidate(
            il1, il2, total, frac, hl, hl_pct, peg, peg_pct,
            np_default, source="broad_feasible_exploration"
        )
        if row is not None:
            rows.append(row)

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError("Virtual candidate generation produced no feasible formulations.")
    pool["candidate_id"] = [f"V-{i+1:05d}" for i in range(len(pool))]
    return pool.reset_index(drop=True)


def formulation_feasibility_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Total_IL_molpct"] = out["IL1_molpct"].fillna(0) + out["IL2_molpct"].fillna(0)
    out["component_sum"] = (
        out["Total_IL_molpct"] + out["HL_molpct"] + out["CHOL_molpct"] + out["PEG_molpct"]
    )
    mask = (
        out["Total_IL_molpct"].between(HARD_TOTAL_IL_MIN, HARD_TOTAL_IL_MAX)
        & out["HL_molpct"].between(HARD_HL_MIN, HARD_HL_MAX)
        & out["CHOL_molpct"].between(HARD_CHOL_MIN, HARD_CHOL_MAX)
        & out["PEG_molpct"].between(HARD_PEG_MIN, HARD_PEG_MAX)
        & out["component_sum"].between(99.0, 101.0)
        & out["IL1"].notna()
        & out["Phospholipid"].notna()
        & out["PEG"].notna()
    )
    return out.loc[mask].reset_index(drop=True)


def remove_exact_training_duplicates(pipeline, pool: pd.DataFrame, train_signatures: Sequence[str], decimals: int) -> pd.DataFrame:
    train_set = set(str(s) for s in train_signatures)
    sig = pool.apply(lambda r: pipeline.formulation_signature(r, decimals), axis=1)
    keep = ~sig.astype(str).isin(train_set)
    out = pool.loc[keep].copy()
    out["formulation_signature"] = sig.loc[keep].astype(str).values
    out = out.drop_duplicates("formulation_signature", keep="first").reset_index(drop=True)
    return out


def load_candidate_pool(args: argparse.Namespace, pipeline, ctx: TrainingContext) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if args.candidate_source in {"excel", "combined"}:
        if not args.candidate_excel:
            raise ValueError("--candidate-source excel/combined requires --candidate-excel")
        raw = pd.read_excel(args.candidate_excel, sheet_name=parse_sheet(args.candidate_sheet))
        ext = standardize_candidate_columns(raw)
        ext["candidate_source"] = "user_candidate_excel"
        ext["anchor_id"] = ""
        ext["anchor_TE"] = np.nan
        parts.append(ext)
        print(f"[CandidatePool] User workbook: n={len(ext)}")

    if args.candidate_source in {"virtual", "combined"}:
        virt = generate_virtual_pool(ctx.standardized, ctx.y, args.max_virtual_candidates, args.seed)
        parts.append(virt)
        print(f"[CandidatePool] Virtual library generated: n={len(virt)}")

    pool = pd.concat(parts, ignore_index=True) if parts else generate_virtual_pool(
        ctx.standardized, ctx.y, args.max_virtual_candidates, args.seed
    )
    pool = formulation_feasibility_filter(pool)
    pool = remove_exact_training_duplicates(
        pipeline, pool, ctx.signatures, args.group_round_decimals
    )
    pool["candidate_id"] = [f"POOL-{i+1:05d}" for i in range(len(pool))]
    print(f"[CandidatePool] After feasibility + exact training-duplicate removal: n={len(pool)}")
    if len(pool) < 200:
        print("[CandidatePool] WARNING: candidate pool is small; ranking-range selection may be constrained.")
    return pool.reset_index(drop=True)


def build_candidate_features(pipeline, ctx: TrainingContext, pool: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    Xc_core, Xc_aux = pipeline.build_feature_blocks(
        pool, ctx.morgan_map, n_bits=ctx.morgan_n_bits
    )
    Xc_core = Xc_core.reset_index(drop=True).reindex(columns=ctx.X_core.columns, fill_value=0.0)
    Xc_aux = Xc_aux.reset_index(drop=True).reindex(columns=ctx.X_aux.columns, fill_value=0.0)
    Xc_selected = pipeline.combine_selected_features(
        Xc_core, Xc_aux, np.arange(len(pool)), ctx.selected_aux
    ).reset_index(drop=True)
    Xc_selected = Xc_selected.reindex(columns=ctx.X_selected.columns, fill_value=0.0).astype(float)
    return Xc_core, Xc_aux, Xc_selected


# -----------------------------------------------------------------------------
# Tree-model deployment predictions
# -----------------------------------------------------------------------------
def fit_tree_models(pipeline, ctx: TrainingContext, X_pool: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = pipeline.model_specs(include_optional_boosters=True)
    pred = pd.DataFrame(index=X_pool.index)
    status_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 92)
    print("[TREE DEPLOYMENT FITS] Frozen R1-R4 -> prospective pool")
    print("Requested:", TREE_MODELS_WANTED)
    print("=" * 92)

    for i, name in enumerate(TREE_MODELS_WANTED, start=1):
        if name not in specs:
            status_rows.append({"model": name, "status": "unavailable", "note": "not present in imported model_specs"})
            print(f"[Tree] {name:<22} unavailable in current environment; skipped")
            continue
        estimator, param_dist = specs[name]
        started = time.time()
        try:
            fitted, best_params = pipeline.fit_tuned_estimator(
                estimator,
                param_dist,
                ctx.X_selected,
                ctx.y,
                ctx.config,
                seed_offset=12000 + i,
                groups_train=ctx.groups,
            )
            p = np.asarray(fitted.predict(X_pool.values.astype(float)), dtype=float).reshape(-1)
            pred[f"pred_{name}"] = p
            status_rows.append({
                "model": name,
                "status": "completed",
                "runtime_min": (time.time() - started) / 60.0,
                "best_params": json.dumps(json_safe(best_params), ensure_ascii=False, sort_keys=True),
            })
            print(f"[Tree] {name:<22} completed | pred range={np.nanmin(p):.3f}..{np.nanmax(p):.3f}")
        except Exception as exc:
            status_rows.append({
                "model": name,
                "status": "failed",
                "runtime_min": (time.time() - started) / 60.0,
                "note": f"{type(exc).__name__}: {exc}",
            })
            print(f"[Tree] {name:<22} FAILED: {type(exc).__name__}: {exc}")

    if pred.empty:
        raise RuntimeError("No requested tree model produced predictions.")
    return pred, pd.DataFrame(status_rows)


def add_rank_columns(frame: pd.DataFrame, prediction_columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for c in prediction_columns:
        out[f"rankpct_{c.removeprefix('pred_')}"] = percentile_rank(out[c])
    return out


def build_tree_prefilter(scored: pd.DataFrame, max_pool: int, seed: int) -> pd.DataFrame:
    """Create one common prospective pool for all models, preserving top/mid/low/disagreement."""
    rng = np.random.default_rng(seed + 99)
    rank_cols = [c for c in scored.columns if c.startswith("rankpct_")]
    if not rank_cols:
        raise RuntimeError("No tree rank columns were available for candidate prefilter.")
    x = scored.copy()
    x["tree_consensus_rank"] = x[rank_cols].mean(axis=1)
    x["tree_rank_disagreement"] = x[rank_cols].std(axis=1).fillna(0.0)

    if len(x) <= max_pool:
        return x.reset_index(drop=True)

    chosen: set[int] = set()
    # 40% strongest tree consensus points.
    n_top = max(200, int(max_pool * 0.40))
    chosen.update(x.nlargest(n_top, "tree_consensus_rank").index.tolist())

    # 20% highest disagreement points.
    n_dis = max(100, int(max_pool * 0.20))
    chosen.update(x.nlargest(n_dis, "tree_rank_disagreement").index.tolist())

    # Remaining capacity is stratified across the entire tree-consensus distribution,
    # ensuring genuinely intermediate and low candidates survive to TabPFN/TabFM.
    remaining = max_pool - len(chosen)
    if remaining > 0:
        try:
            bins = pd.qcut(x["tree_consensus_rank"], q=10, duplicates="drop")
            groups = x.groupby(bins, observed=True)
            per = max(1, math.ceil(remaining / max(1, len(groups))))
            strat = []
            for _, g in groups:
                avail = g.loc[~g.index.isin(chosen)]
                if len(avail):
                    take = min(per, len(avail))
                    strat.extend(rng.choice(avail.index.to_numpy(), size=take, replace=False).tolist())
            chosen.update(strat)
        except Exception:
            pass

    if len(chosen) < max_pool:
        avail = x.loc[~x.index.isin(chosen)]
        take = min(max_pool - len(chosen), len(avail))
        if take:
            chosen.update(rng.choice(avail.index.to_numpy(), size=take, replace=False).tolist())

    # If union exceeded max_pool, keep top + disagreement anchors, then deterministic sample.
    idx = list(chosen)
    if len(idx) > max_pool:
        protected = set(x.nlargest(min(n_top, max_pool // 2), "tree_consensus_rank").index.tolist())
        protected.update(x.nlargest(min(n_dis, max_pool // 4), "tree_rank_disagreement").index.tolist())
        rest = [i for i in idx if i not in protected]
        need = max_pool - len(protected)
        if need > 0 and rest:
            rest = rng.choice(np.asarray(rest), size=min(need, len(rest)), replace=False).tolist()
        idx = list(protected) + list(rest)

    out = x.loc[idx].copy().reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Applicability domain (AD)
# -----------------------------------------------------------------------------
def add_applicability_domain(
    scored: pd.DataFrame,
    X_train: pd.DataFrame,
    X_pool: pd.DataFrame,
    k: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train.values.astype(float))
    Xp = scaler.transform(X_pool.values.astype(float))

    k_train = min(k + 1, len(Xt))
    nn_train = NearestNeighbors(n_neighbors=k_train).fit(Xt)
    dt, _ = nn_train.kneighbors(Xt)
    # First neighbor is self; leave it out.
    train_mean = dt[:, 1:].mean(axis=1) if dt.shape[1] > 1 else dt[:, 0]
    ad95 = float(np.quantile(train_mean, 0.95))
    ad99 = float(np.quantile(train_mean, 0.99))

    k_pool = min(k, len(Xt))
    nn = NearestNeighbors(n_neighbors=k_pool).fit(Xt)
    dp, _ = nn.kneighbors(Xp)
    pool_mean = dp.mean(axis=1)

    out = scored.copy()
    out["AD_mean5NN_distance"] = pool_mean
    out["AD_ratio_to_train95"] = pool_mean / max(ad95, 1e-12)
    out["AD_in95"] = pool_mean <= ad95
    out["AD_in125pct95"] = pool_mean <= 1.25 * ad95
    return out, {"train_AD95": ad95, "train_AD99": ad99, "k": k}, Xp


# -----------------------------------------------------------------------------
# Foundation-model child process
# -----------------------------------------------------------------------------
def _foundation_transient(exc: Exception) -> bool:
    msg = str(exc).casefold()
    return any(t in msg for t in (
        "ssl", "eof", "server disconnected", "connection", "connecterror",
        "getaddrinfo", "timed out", "timeout", "network", "httpx",
        "temporarily unavailable", "remote protocol", "readerror", "502", "503",
    ))


def init_tabpfn(backend: str):
    errors = []
    token = (os.environ.get("PRIORLABS_API_KEY", "") or os.environ.get("TABPFN_TOKEN", "")).strip()
    os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    if backend in {"auto", "client"}:
        try:
            import tabpfn_client
            from tabpfn_client import TabPFNRegressor as ClientRegressor
            if token:
                setter = getattr(tabpfn_client, "set_access_token", None)
                if callable(setter):
                    setter(token)
                os.environ["PRIORLABS_API_KEY"] = token
                os.environ["TABPFN_TOKEN"] = token
            return (lambda: ClientRegressor()), "client"
        except Exception as exc:
            errors.append(f"client: {type(exc).__name__}: {exc}")
            if backend == "client":
                raise
    if backend in {"auto", "local"}:
        try:
            from tabpfn import TabPFNRegressor
            return (lambda: TabPFNRegressor()), "local"
        except Exception as exc:
            errors.append(f"local: {type(exc).__name__}: {exc}")
            if backend == "local":
                raise
    raise RuntimeError("TabPFN unavailable: " + " | ".join(errors))


def init_tabfm():
    from tabfm import TabFMRegressor
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    print("[FoundationChild] Loading TabFM v1.0.0 PyTorch regression weights...")
    pretrained = tabfm_v1_0_0.load(model_type="regression")
    return (lambda: TabFMRegressor(model=pretrained)), "tabfm_v1.0.0_pytorch"


def fit_predict_retry(factory, X_train, y_train, X_pool, model_name: str, retries: int = 3) -> np.ndarray:
    last = None
    for attempt in range(1, retries + 1):
        try:
            m = factory()
            m.fit(X_train, y_train)
            p = np.asarray(m.predict(X_pool), dtype=float).reshape(-1)
            if len(p) != len(X_pool):
                raise RuntimeError(f"{model_name} returned {len(p)} predictions for {len(X_pool)} rows")
            return p
        except Exception as exc:
            last = exc
            if not _foundation_transient(exc) or attempt == retries:
                raise
            wait = 5 * attempt
            print(f"[FoundationChild] {model_name} transient error; retry {attempt}/{retries} after {wait}s")
            time.sleep(wait)
    raise last  # pragma: no cover


def foundation_child_main(args: argparse.Namespace) -> None:
    payload = Path(args.payload_dir).resolve()
    X_train = pd.read_csv(payload / "X_train_selected.csv")
    y_train = pd.read_csv(payload / "y_train.csv")["TE"].to_numpy(dtype=float)
    X_pool = pd.read_csv(payload / "X_foundation_pool_selected.csv")

    out = pd.DataFrame(index=np.arange(len(X_pool)))
    status = []

    if not args.skip_tabpfn:
        started = time.time()
        try:
            factory, backend = init_tabpfn(args.tabpfn_backend)
            p = fit_predict_retry(factory, X_train, y_train, X_pool, "TabPFN", retries=3)
            out["pred_TabPFN"] = p
            status.append({"model": "TabPFN", "status": "completed", "backend": backend, "runtime_min": (time.time()-started)/60})
            print(f"[FoundationChild] TabPFN completed | range={p.min():.3f}..{p.max():.3f}")
        except Exception as exc:
            status.append({"model": "TabPFN", "status": "failed", "backend": "", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FoundationChild] TabPFN FAILED: {type(exc).__name__}: {exc}")

    if not args.skip_tabfm:
        started = time.time()
        try:
            factory, backend = init_tabfm()
            p = fit_predict_retry(factory, X_train, y_train, X_pool, "TabFM", retries=1)
            out["pred_TabFM"] = p
            status.append({"model": "TabFM", "status": "completed", "backend": backend, "runtime_min": (time.time()-started)/60})
            print(f"[FoundationChild] TabFM completed | range={p.min():.3f}..{p.max():.3f}")
        except Exception as exc:
            status.append({"model": "TabFM", "status": "failed", "backend": "", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FoundationChild] TabFM FAILED: {type(exc).__name__}: {exc}")

    out.insert(0, "pool_row", np.arange(len(out)))
    out.to_csv(payload / "foundation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(status).to_csv(payload / "foundation_status.csv", index=False, encoding="utf-8-sig")
    print(f"[FoundationChild] Results saved in {payload}")


def candidate_tabfm_interpreters() -> List[Path]:
    home = Path.home()
    starts = [Path.cwd(), Path(__file__).resolve().parent, home]
    candidates: List[Path] = []

    def add(root: Path) -> None:
        candidates.extend([
            root / ".venv_tabfm" / "Scripts" / "python.exe",
            root / ".venv_tabfm" / "Scripts" / "python",
            root / ".venv_tabfm" / "bin" / "python",
        ])

    for s in starts:
        for root in [s, *list(s.parents)[:6]]:
            add(root)
    search_roots = [home / "Desktop", home / "桌面", home / "Documents", home / "OneDrive"]
    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            for env in root.glob("**/.venv_tabfm"):
                try:
                    if len(env.relative_to(root).parts) <= 6:
                        add(env.parent)
                except Exception:
                    pass
        except Exception:
            pass
    unique = []
    seen = set()
    for c in candidates:
        key = os.path.normcase(os.path.abspath(str(c)))
        if c.is_file() and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def clean_child_env(python_exe: Path) -> Dict[str, str]:
    env = os.environ.copy()
    for key in ["PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__"]:
        env.pop(key, None)
    env["VIRTUAL_ENV"] = str(python_exe.parent.parent)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    env["PATH"] = str(python_exe.parent) + os.pathsep + env.get("PATH", "")
    return env


def interpreter_has_tabfm(python_exe: Path) -> bool:
    try:
        r = subprocess.run(
            [str(python_exe), "-I", "-c", "import numpy,pandas,sklearn,tabfm; print('ok')"],
            env=clean_child_env(python_exe),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def run_foundation_child(args: argparse.Namespace, payload_dir: Path) -> pd.DataFrame:
    if args.skip_tabpfn and args.skip_tabfm:
        return pd.DataFrame()

    # First, if the current interpreter already has TabFM and was explicitly requested,
    # it can run the child logic directly through a subprocess.  Otherwise locate the
    # user's existing isolated .venv_tabfm.
    interpreters = candidate_tabfm_interpreters()
    selected = next((p for p in interpreters if interpreter_has_tabfm(p)), None)
    if selected is None:
        print("[Foundation] No healthy .venv_tabfm interpreter found. Tree-only R5 results will still be saved.")
        return pd.DataFrame()

    cmd = [
        str(selected), "-I", str(Path(__file__).resolve()),
        "--foundation-child",
        "--payload-dir", str(payload_dir),
        "--tabpfn-backend", args.tabpfn_backend,
    ]
    if args.skip_tabpfn:
        cmd.append("--skip-tabpfn")
    if args.skip_tabfm:
        cmd.append("--skip-tabfm")

    print(f"[Foundation] Launching isolated child: {selected}")
    completed = subprocess.run(
        cmd,
        env=clean_child_env(selected),
        cwd=str(Path(__file__).resolve().parent),
        check=False,
    )
    pred_path = payload_dir / "foundation_predictions.csv"
    if not pred_path.is_file():
        print(f"[Foundation] Child exited code {completed.returncode}; no predictions file was produced.")
        return pd.DataFrame()
    return pd.read_csv(pred_path)


# -----------------------------------------------------------------------------
# Final all-model scoring + 12-point design
# -----------------------------------------------------------------------------
def add_consensus_scores(scored: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = scored.copy()
    pred_cols = [c for c in out.columns if c.startswith("pred_") and out[c].notna().any()]
    rank_cols = []
    for c in pred_cols:
        name = c.removeprefix("pred_")
        rc = f"rankpct_{name}"
        out[rc] = percentile_rank(out[c])
        rank_cols.append(rc)
    if len(rank_cols) < 2:
        raise RuntimeError("Fewer than two model prediction columns are available; cannot form a robust consensus.")

    out["consensus_rank"] = out[rank_cols].mean(axis=1)
    out["consensus_rank_median"] = out[rank_cols].median(axis=1)
    out["model_rank_disagreement"] = out[rank_cols].std(axis=1).fillna(0.0)
    out["n_models"] = len(rank_cols)
    out["n_models_top20"] = (out[rank_cols] >= 0.80).sum(axis=1)
    out["n_models_top10"] = (out[rank_cols] >= 0.90).sum(axis=1)
    out["all_model_raw_mean"] = out[pred_cols].mean(axis=1)

    tree_rank_cols = [c for c in rank_cols if c.removeprefix("rankpct_") in TREE_MODELS_WANTED]
    foundation_rank_cols = [c for c in rank_cols if c.removeprefix("rankpct_") in FOUNDATION_MODELS_WANTED]
    out["tree_consensus_rank"] = out[tree_rank_cols].mean(axis=1) if tree_rank_cols else np.nan
    out["foundation_consensus_rank"] = out[foundation_rank_cols].mean(axis=1) if foundation_rank_cols else np.nan
    return out, rank_cols


def normalize01(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=x.index)
    return (x - lo) / (hi - lo)


def same_ionizable_identity(row: pd.Series) -> bool:
    """True only when a nominal dual-IL formulation repeats the exact same IL twice."""
    a = str(row.get("IL1", "")).strip().casefold()
    b = str(row.get("IL2", "")).strip().casefold()
    if b in {"", "nan", "none", "na", "-"}:
        return False
    return a == b


def build_simple_formulation_diversity_matrix(df: pd.DataFrame) -> np.ndarray:
    """A lightweight standardized representation used only for re-selection.

    This is NOT a new predictive feature space.  It is only used to avoid choosing
    near-duplicate validation formulations when reusing an already frozen/scored
    candidate workbook.  Categorical formulation identities are one-hot encoded and
    molar fractions are standardized.
    """
    x = pd.DataFrame(index=df.index)
    for c in ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct", "NP_ratio"]:
        x[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        if x[c].notna().any():
            x[c] = x[c].fillna(x[c].median())
        else:
            x[c] = 0.0

    cats = []
    for c in ["IL1", "IL2", "Phospholipid", "PEG"]:
        s = df.get(c, pd.Series("", index=df.index)).fillna("").astype(str)
        cats.append(pd.get_dummies(s, prefix=c, dtype=float))
    X = pd.concat([x] + cats, axis=1).astype(float)
    return StandardScaler().fit_transform(X.values)


def greedy_select(
    candidates: pd.DataFrame,
    Xz: np.ndarray,
    n: int,
    base_score_col: str,
    selected_indices: Optional[List[int]] = None,
    diversity_weight: float = 0.08,
) -> List[int]:
    chosen = list(selected_indices or [])
    available = [int(i) for i in candidates.index if int(i) not in set(chosen)]
    if not available or n <= 0:
        return []

    new: List[int] = []
    base = normalize01(candidates[base_score_col])
    while len(new) < n and available:
        best_idx = None
        best_score = -np.inf
        current = chosen + new
        for idx in available:
            score = float(base.loc[idx])
            if current:
                d = np.linalg.norm(Xz[current] - Xz[idx], axis=1)
                min_d = float(np.min(d))
                # Saturate large distances; we only want to avoid near-clones, not reward OOD.
                score += diversity_weight * min(min_d / 3.0, 1.0)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        new.append(best_idx)
        available.remove(best_idx)
    return new


def select_r5_panel(
    scored: pd.DataFrame,
    Xz: np.ndarray,
    n_total: int,
    seed: int,
    medium_raw_target: float = 0.65,
    medium_raw_low: float = 0.58,
    medium_raw_high: float = 0.72,
    low_raw_target: float = 0.42,
    low_raw_low: float = 0.32,
    low_raw_high: float = 0.50,
    allow_same_ionizable_lipid: bool = False,
) -> pd.DataFrame:
    """Select a validation-balanced R5 panel.

    v1.1 changes relative to v1.0
    -----------------------------
    * High candidates remain consensus-rank driven.
    * Intermediate candidates are no longer defined by candidate-pool percentile alone.
      They must also sit near an absolute multi-model mean prediction target (default 0.65).
      This avoids calling a prediction of ~0.81 "intermediate" merely because the candidate
      pool itself is enriched for high-response formulations.
    * Low controls are targeted around ~0.42 rather than blindly taking the extreme bottom.
      This keeps them low, measurable, and within-domain.
    * Nominal dual-IL candidates with IL1 == IL2 are excluded by default.
    """
    if n_total < 10 or n_total > 15:
        raise ValueError("R5 panel size should be 10-15 for this design; default is 12.")

    x = scored.copy().reset_index(drop=True)
    if "all_model_raw_mean" not in x.columns:
        pred_cols = [c for c in x.columns if c.startswith("pred_") and x[c].notna().any()]
        if not pred_cols:
            raise RuntimeError("No model prediction columns available for absolute medium/low selection.")
        x["all_model_raw_mean"] = x[pred_cols].mean(axis=1)

    x["same_ionizable_identity"] = x.apply(same_ionizable_identity, axis=1)
    chemistry_ok = pd.Series(True, index=x.index)
    if not allow_same_ionizable_lipid:
        chemistry_ok = ~x["same_ionizable_identity"]

    ad_excess = np.maximum(
        pd.to_numeric(x["AD_ratio_to_train95"], errors="coerce").fillna(2.0) - 1.0,
        0.0,
    )
    x["high_confidence_score"] = (
        x["consensus_rank"]
        - 0.28 * x["model_rank_disagreement"]
        - 0.10 * np.minimum(ad_excess, 2.0)
    )
    x["high_diverse_score"] = x["consensus_rank"] - 0.12 * np.minimum(ad_excess, 2.0)
    x["disagreement_score"] = x["model_rank_disagreement"] - 0.08 * np.minimum(ad_excess, 2.0)

    selected: List[Tuple[int, str]] = []
    used: List[int] = []

    if n_total == 12:
        n_high_conf, n_high_div, n_mid, n_low, n_dis = 5, 2, 2, 2, 1
    else:
        n_dis = 1
        n_mid = max(1, round(n_total * 0.17))
        n_low = max(1, round(n_total * 0.17))
        n_high = n_total - n_mid - n_low - n_dis
        n_high_conf = max(1, n_high - 2)
        n_high_div = n_high - n_high_conf

    # ------------------------------------------------------------------
    # High: multi-model consensus + AD support.
    # ------------------------------------------------------------------
    majority_top20 = np.ceil(0.60 * x["n_models"].clip(lower=1)).astype(int)
    high_conf_pool = x[
        chemistry_ok
        & (x["consensus_rank"] >= 0.90)
        & (x["n_models_top20"] >= majority_top20)
        & (x["AD_ratio_to_train95"] <= 1.25)
    ].copy()
    if len(high_conf_pool) < n_high_conf:
        high_conf_pool = x[
            chemistry_ok
            & (x["consensus_rank"] >= 0.85)
            & (x["n_models_top20"] >= majority_top20)
            & (x["AD_ratio_to_train95"] <= 1.35)
        ].copy()
    if len(high_conf_pool) < n_high_conf:
        high_conf_pool = x[
            chemistry_ok
            & (x["consensus_rank"] >= 0.80)
            & (x["AD_ratio_to_train95"] <= 1.50)
        ].copy()

    hc = greedy_select(
        high_conf_pool, Xz, n_high_conf, "high_confidence_score", used,
        diversity_weight=0.06,
    )
    for j, idx in enumerate(hc, start=1):
        selected.append((idx, f"H{j}_HighConsensus"))
    used.extend(hc)

    high_div_pool = x[
        chemistry_ok
        & (x["consensus_rank"] >= 0.80)
        & (x["AD_ratio_to_train95"] <= 1.40)
        & (~x.index.isin(used))
    ].copy()
    if len(high_div_pool) < n_high_div:
        high_div_pool = x[
            chemistry_ok
            & (x["consensus_rank"] >= 0.72)
            & (x["AD_ratio_to_train95"] <= 1.55)
            & (~x.index.isin(used))
        ].copy()

    high_remaining = high_div_pool.copy()
    hd: List[int] = []
    for _ in range(n_high_div):
        if high_remaining.empty:
            break
        best = None
        best_score = -np.inf
        for idx in high_remaining.index:
            if used + hd:
                min_d = float(np.min(np.linalg.norm(Xz[used + hd] - Xz[int(idx)], axis=1)))
            else:
                min_d = 0.0
            score = (
                0.72 * float(x.loc[int(idx), "consensus_rank"])
                + 0.28 * min(min_d / 3.0, 1.0)
                - 0.08 * max(float(x.loc[int(idx), "AD_ratio_to_train95"]) - 1.0, 0.0)
            )
            if score > best_score:
                best_score = score
                best = int(idx)
        if best is None:
            break
        hd.append(best)
        high_remaining = high_remaining.drop(index=best)

    for j, idx in enumerate(hd, start=len(hc) + 1):
        selected.append((idx, f"H{j}_HighDiverse"))
    used.extend(hd)

    # ------------------------------------------------------------------
    # Intermediate: absolute response target + consensus support.
    # Default target ≈0.65 on the same normalized DC2.4 scale.
    # ------------------------------------------------------------------
    mid_pool = x[
        chemistry_ok
        & x["all_model_raw_mean"].between(medium_raw_low, medium_raw_high)
        & x["consensus_rank"].between(0.10, 0.45)
        & (x["AD_ratio_to_train95"] <= 1.25)
        & (~x.index.isin(used))
    ].copy()
    if len(mid_pool) < n_mid:
        mid_pool = x[
            chemistry_ok
            & x["all_model_raw_mean"].between(medium_raw_low - 0.06, medium_raw_high + 0.05)
            & x["consensus_rank"].between(0.06, 0.55)
            & (x["AD_ratio_to_train95"] <= 1.40)
            & (~x.index.isin(used))
        ].copy()

    mid_scale = max(medium_raw_high - medium_raw_low, 0.05)
    mid_pool["mid_target_score"] = (
        -np.abs(mid_pool["all_model_raw_mean"] - medium_raw_target) / mid_scale
        - 0.30 * mid_pool["model_rank_disagreement"]
        - 0.05 * np.abs(mid_pool["consensus_rank"] - 0.22)
    )
    mids = greedy_select(
        mid_pool, Xz, n_mid, "mid_target_score", used,
        diversity_weight=0.10,
    )
    for j, idx in enumerate(mids, start=1):
        selected.append((idx, f"M{j}_Intermediate"))
    used.extend(mids)

    # ------------------------------------------------------------------
    # Low: clearly low but not absurdly extreme/OOD.  The default target
    # around 0.42 creates a useful experimental gap from medium (~0.65).
    # ------------------------------------------------------------------
    low_pool = x[
        chemistry_ok
        & x["all_model_raw_mean"].between(low_raw_low, low_raw_high)
        & (x["consensus_rank"] <= 0.18)
        & (x["AD_ratio_to_train95"] <= 1.15)
        & (~x.index.isin(used))
    ].copy()
    if len(low_pool) < n_low:
        low_pool = x[
            chemistry_ok
            & x["all_model_raw_mean"].between(low_raw_low - 0.07, low_raw_high + 0.04)
            & (x["consensus_rank"] <= 0.25)
            & (x["AD_ratio_to_train95"] <= 1.35)
            & (~x.index.isin(used))
        ].copy()

    low_scale = max(low_raw_high - low_raw_low, 0.05)
    low_pool["low_control_score"] = (
        -np.abs(low_pool["all_model_raw_mean"] - low_raw_target) / low_scale
        - 0.30 * low_pool["model_rank_disagreement"]
        - 0.08 * low_pool["consensus_rank"]
    )
    lows = greedy_select(
        low_pool, Xz, n_low, "low_control_score", used,
        diversity_weight=0.10,
    )
    for j, idx in enumerate(lows, start=1):
        selected.append((idx, f"L{j}_LowControl"))
    used.extend(lows)

    # ------------------------------------------------------------------
    # Disagreement: informative boundary probe, still chemistry-valid and
    # not an extreme extrapolation.
    # ------------------------------------------------------------------
    dis_pool = x[
        chemistry_ok
        & (x["consensus_rank"].between(0.30, 0.85))
        & (x["AD_ratio_to_train95"] <= 1.55)
        & (~x.index.isin(used))
    ].copy()
    if dis_pool.empty:
        dis_pool = x[chemistry_ok & (~x.index.isin(used))].copy()

    dis_idx = (
        dis_pool.sort_values("disagreement_score", ascending=False)
        .head(n_dis)
        .index.astype(int).tolist()
    )
    for j, idx in enumerate(dis_idx, start=1):
        selected.append((idx, f"D{j}_ModelDisagreement"))
    used.extend(dis_idx)

    # Fallback remains chemistry-valid.
    if len(selected) < n_total:
        remaining = x.loc[
            chemistry_ok & (~x.index.isin(used))
        ].sort_values("high_confidence_score", ascending=False)
        for idx in remaining.index[: n_total - len(selected)]:
            selected.append((int(idx), "F_Fallback"))
            used.append(int(idx))

    panel = x.loc[[i for i, _ in selected]].copy()
    panel.insert(0, "selection_role", [role for _, role in selected])

    if n_total >= 12:
        n_high_selected = int(panel["selection_role"].str.contains("High", na=False).sum())
        if n_high_selected < 6:
            raise RuntimeError(
                f"Only {n_high_selected} high candidates could be selected. "
                "Do not proceed with this R5 panel; enlarge or revise the feasible candidate pool."
            )

    panel.insert(1, "selection_internal_index", [i for i, _ in selected])

    # Randomize wet-lab IDs.
    rng = np.random.default_rng(seed + 700)
    codes = [f"R5-{i+1:02d}" for i in range(len(panel))]
    perm = rng.permutation(len(codes))
    assigned = [None] * len(codes)
    for row_pos, code_pos in enumerate(perm):
        assigned[row_pos] = codes[int(code_pos)]
    panel.insert(0, "R5_code", assigned)

    role_order = {
        "HighConsensus": 0,
        "HighDiverse": 1,
        "Intermediate": 2,
        "LowControl": 3,
        "ModelDisagreement": 4,
        "Fallback": 5,
    }

    def role_key(s: str) -> int:
        for key, order in role_order.items():
            if key in s:
                return order
        return 9

    panel["_role_order"] = panel["selection_role"].map(role_key)
    panel = (
        panel.sort_values(["_role_order", "consensus_rank"], ascending=[True, False])
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )
    return panel


# -----------------------------------------------------------------------------
# Output / freezing
# -----------------------------------------------------------------------------
def historical_controls(ctx: TrainingContext) -> pd.DataFrame:
    df = ctx.standardized.copy()
    df["TE"] = ctx.y.values
    top = df.nlargest(1, "TE").copy()
    med_target = float(ctx.y.median())
    mid_idx = (ctx.y - med_target).abs().idxmin()
    mid = df.loc[[mid_idx]].copy()
    top.insert(0, "control_role", "historical_champion_same_plate_anchor")
    mid.insert(0, "control_role", "historical_mid_reference")
    cols = ["control_role", "candidate_id", "TE", "IL1", "IL1_molpct", "IL2", "IL2_molpct", "Phospholipid", "HL_molpct", "CHOL_molpct", "PEG", "PEG_molpct", "NP_ratio"]
    return pd.concat([top, mid], ignore_index=True)[cols]


def write_outputs(
    output_dir: Path,
    panel: pd.DataFrame,
    scored_pool: pd.DataFrame,
    tree_full_pool: pd.DataFrame,
    ctx: TrainingContext,
    tree_status: pd.DataFrame,
    foundation_status: pd.DataFrame,
    ad_info: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "R5_selected_12_FROZEN.csv"
    panel.to_csv(selected_path, index=False, encoding="utf-8-sig")

    # Blind wet-lab sheet: no model predictions or role labels.
    wet_cols = [
        "R5_code", "IL1", "IL1_molpct", "IL2", "IL2_molpct",
        "Phospholipid", "HL_molpct", "CHOL_molpct", "PEG", "PEG_molpct", "NP_ratio",
    ]
    wet = panel[wet_cols].sort_values("R5_code").reset_index(drop=True)

    # Prediction key: unblinded, must be frozen BEFORE experiment.
    pred_key = panel.copy()
    model_pred_cols = [c for c in pred_key.columns if c.startswith("pred_") or c.startswith("rankpct_")]
    key_cols = [
        "R5_code", "selection_role", "consensus_rank", "consensus_rank_median",
        "model_rank_disagreement", "n_models_top20", "AD_ratio_to_train95",
        "candidate_source", "anchor_id", "anchor_TE",
        "IL1", "IL1_molpct", "IL2", "IL2_molpct", "Phospholipid", "HL_molpct",
        "CHOL_molpct", "PEG", "PEG_molpct", "NP_ratio",
    ] + model_pred_cols
    key_cols = [c for c in key_cols if c in pred_key.columns]
    pred_key = pred_key[key_cols]

    selected_aux_df = pd.DataFrame({"selected_auxiliary_feature": ctx.selected_aux})
    controls = historical_controls(ctx)

    run_info = pd.DataFrame([
        {"item": "script_version", "value": SCRIPT_VERSION},
        {"item": "timestamp_local", "value": pd.Timestamp.now().isoformat()},
        {"item": "training_workbook", "value": ctx.workbook_path},
        {"item": "training_sha256", "value": sha256_file(ctx.workbook_path)},
        {"item": "tree_pipeline", "value": ctx.pipeline_path},
        {"item": "tree_pipeline_sha256", "value": sha256_file(ctx.pipeline_path)},
        {"item": "target", "value": ctx.target_label},
        {"item": "n_training", "value": len(ctx.y)},
        {"item": "n_training_groups", "value": len(np.unique(ctx.groups))},
        {"item": "n_selected_R5", "value": len(panel)},
        {"item": "selected_aux_count", "value": len(ctx.selected_aux)},
        {"item": "common_model_pool_n", "value": len(scored_pool)},
        {"item": "seed", "value": args.seed},
        {"item": "AD_train95", "value": ad_info.get("train_AD95", np.nan)},
        {"item": "candidate_source_mode", "value": args.candidate_source},
    ])

    xlsx = output_dir / "R5_prospective_selection_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pred_key.to_excel(writer, sheet_name="R5_prediction_key", index=False)
        wet.to_excel(writer, sheet_name="R5_blinded_wetlab", index=False)
        controls.to_excel(writer, sheet_name="suggested_plate_controls", index=False)
        scored_pool.to_excel(writer, sheet_name="all_model_scored_pool", index=False)
        tree_full_pool.to_excel(writer, sheet_name="tree_full_pool", index=False)
        ctx.standardized.assign(TE=ctx.y.values).to_excel(writer, sheet_name="training_R1_R4", index=False)
        selected_aux_df.to_excel(writer, sheet_name="selected_aux_features", index=False)
        tree_status.to_excel(writer, sheet_name="tree_model_status", index=False)
        foundation_status.to_excel(writer, sheet_name="foundation_status", index=False)
        run_info.to_excel(writer, sheet_name="run_info", index=False)

    # Separate blind workbook for bench use.
    with pd.ExcelWriter(output_dir / "R5_BLINDED_WETLAB_SHEET.xlsx", engine="openpyxl") as writer:
        wet.to_excel(writer, sheet_name="R5_formulations", index=False)
        controls.to_excel(writer, sheet_name="optional_controls", index=False)

    # Freeze hashes.
    hashes = {
        "selected_csv": str(selected_path),
        "selected_csv_sha256": sha256_file(selected_path),
        "selection_workbook": str(xlsx),
        "selection_workbook_sha256": sha256_file(xlsx),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    with open(output_dir / "R5_FREEZE_HASHES.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(hashes), f, ensure_ascii=False, indent=2)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "scientific_role": "prospective R5 candidate selection before wet-lab measurement",
        "selection_design": {
            "default": "7 high + 2 intermediate + 2 low + 1 model-disagreement",
            "consensus": "mean percentile rank across all available requested models",
            "high_selection": "consensus high + agreement + applicability-domain support + diversity",
            "disagreement_candidate": "controlled information-rich probe, not extreme OOD",
        },
        "models_requested": TREE_MODELS_WANTED + [m for m in FOUNDATION_MODELS_WANTED if not getattr(args, f"skip_{m.lower()}", False)],
        "training": {
            "workbook": ctx.workbook_path,
            "sheet": ctx.sheet_name,
            "target": ctx.target_label,
            "n": len(ctx.y),
            "groups": len(np.unique(ctx.groups)),
            "morgan_source": ctx.morgan_source,
            "morgan_bits": ctx.morgan_n_bits,
            "selected_aux": ctx.selected_aux,
        },
        "ad": ad_info,
        "hashes": hashes,
        "arguments": vars(args),
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("[R5 FROZEN OUTPUTS]")
    print(f"Selection workbook : {xlsx}")
    print(f"Blind wet-lab sheet: {output_dir / 'R5_BLINDED_WETLAB_SHEET.xlsx'}")
    print(f"Frozen CSV         : {selected_path}")
    print(f"SHA-256            : {hashes['selected_csv_sha256']}")
    print("IMPORTANT: do not regenerate candidate selection after seeing R5 outcomes.")
    print("=" * 92)


def _composition_match_key(df: pd.DataFrame) -> pd.Series:
    """Stable composition key for matching an old selected panel back to its scored pool."""
    parts = []
    for c in ["IL1", "IL2", "Phospholipid", "PEG"]:
        parts.append(df.get(c, pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.casefold())
    nums = []
    for c in ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct", "NP_ratio"]:
        v = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        nums.append(v.map(lambda z: "" if pd.isna(z) else f"{float(z):.4f}"))
    key = parts[0]
    for s in parts[1:] + nums:
        key = key + "|" + s
    return key


def write_reselection_outputs(
    source_workbook: Path,
    output_dir: Path,
    panel: pd.DataFrame,
    scored_pool: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "R5_selected_12_FROZEN_v1_1.csv"
    panel.to_csv(selected_path, index=False, encoding="utf-8-sig")

    wet_cols = [
        "R5_code", "IL1", "IL1_molpct", "IL2", "IL2_molpct",
        "Phospholipid", "HL_molpct", "CHOL_molpct", "PEG", "PEG_molpct", "NP_ratio",
    ]
    wet = panel[wet_cols].sort_values("R5_code").reset_index(drop=True)

    model_cols = [
        c for c in panel.columns
        if c.startswith("pred_") or c.startswith("rankpct_")
    ]
    key_cols = [
        "R5_code", "selection_role", "candidate_id",
        "consensus_rank", "consensus_rank_median", "all_model_raw_mean",
        "model_rank_disagreement", "n_models_top20", "n_models_top10",
        "AD_ratio_to_train95", "candidate_source", "anchor_id", "anchor_TE",
        "IL1", "IL1_molpct", "IL2", "IL2_molpct", "Phospholipid", "HL_molpct",
        "CHOL_molpct", "PEG", "PEG_molpct", "NP_ratio",
    ] + model_cols
    key_cols = [c for c in key_cols if c in panel.columns]
    pred_key = panel[key_cols].copy()

    workbook_out = output_dir / "R5_prospective_selection_results_v1_1.xlsx"
    with pd.ExcelWriter(workbook_out, engine="openpyxl") as writer:
        pred_key.to_excel(writer, sheet_name="R5_prediction_key_v1_1", index=False)
        wet.to_excel(writer, sheet_name="R5_blinded_wetlab", index=False)
        scored_pool.to_excel(writer, sheet_name="all_model_scored_pool", index=False)

        # Copy useful provenance sheets from the source workbook when available.
        with pd.ExcelFile(source_workbook) as src_xls:
            available_sheets = set(src_xls.sheet_names)
        for sheet in [
            "suggested_plate_controls", "training_R1_R4", "selected_aux_features",
            "tree_model_status", "foundation_status", "run_info",
        ]:
            if sheet in available_sheets:
                pd.read_excel(source_workbook, sheet_name=sheet).to_excel(
                    writer, sheet_name=sheet[:31], index=False
                )

        pd.DataFrame([
            {"item": "script_version", "value": SCRIPT_VERSION},
            {"item": "selection_mode", "value": "reselect_from_existing_scored_workbook"},
            {"item": "source_prediction_workbook", "value": str(source_workbook)},
            {"item": "source_prediction_workbook_sha256", "value": sha256_file(source_workbook)},
            {"item": "medium_raw_target", "value": args.medium_raw_target},
            {"item": "medium_raw_range", "value": f"{args.medium_raw_low}..{args.medium_raw_high}"},
            {"item": "low_raw_target", "value": args.low_raw_target},
            {"item": "low_raw_range", "value": f"{args.low_raw_low}..{args.low_raw_high}"},
            {"item": "allow_same_ionizable_lipid", "value": args.allow_same_ionizable_lipid},
        ]).to_excel(writer, sheet_name="v1_1_reselection_info", index=False)

    blind_path = output_dir / "R5_BLINDED_WETLAB_SHEET_v1_1.xlsx"
    with pd.ExcelWriter(blind_path, engine="openpyxl") as writer:
        wet.to_excel(writer, sheet_name="R5_formulations", index=False)

    hashes = {
        "source_prediction_workbook": str(source_workbook),
        "source_prediction_workbook_sha256": sha256_file(source_workbook),
        "selected_csv": str(selected_path),
        "selected_csv_sha256": sha256_file(selected_path),
        "selection_workbook": str(workbook_out),
        "selection_workbook_sha256": sha256_file(workbook_out),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    with open(output_dir / "R5_FREEZE_HASHES_v1_1.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(hashes), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("[R5 v1.1 RESELECTION FROZEN]")
    print(f"Source predictions : {source_workbook}")
    print(f"Selected CSV       : {selected_path}")
    print(f"Selection workbook : {workbook_out}")
    print(f"Blind wet-lab sheet: {blind_path}")
    print(f"SHA-256            : {hashes['selected_csv_sha256']}")
    print("No model was retrained in --reselect-workbook mode.")
    print("=" * 92)


def reselect_from_existing_workbook(args: argparse.Namespace) -> None:
    """Revise only the validation panel using an already scored prospective workbook.

    By default this preserves the existing 7 High candidates and the existing D1
    disagreement candidate, then replaces only M1/M2/L1/L2 using the v1.1 criteria.
    This is the recommended route for the user's already completed v1.0 prediction run.
    """
    source = Path(args.reselect_workbook).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reselection workbook not found: {source}")

    with pd.ExcelFile(source) as xls:
        source_sheets = set(xls.sheet_names)
    if "all_model_scored_pool" not in source_sheets:
        raise ValueError("Source workbook must contain sheet 'all_model_scored_pool'.")

    scored = pd.read_excel(source, sheet_name="all_model_scored_pool")
    scored, _ = add_consensus_scores(scored)
    if "AD_ratio_to_train95" not in scored.columns:
        raise ValueError("Source scored pool lacks AD_ratio_to_train95.")

    scored["same_ionizable_identity"] = scored.apply(same_ionizable_identity, axis=1)
    Xz = build_simple_formulation_diversity_matrix(scored)

    old_selected_path = source.parent / "R5_selected_12_FROZEN.csv"
    if old_selected_path.is_file():
        old_panel = pd.read_csv(old_selected_path)
    elif "R5_prediction_key" in source_sheets:
        old_panel = pd.read_excel(source, sheet_name="R5_prediction_key")
    else:
        old_panel = pd.DataFrame()

    fixed_rows = pd.DataFrame()
    fixed_codes_by_role = {}
    if not args.reselect_all and not old_panel.empty and "selection_role" in old_panel.columns:
        keep = old_panel["selection_role"].astype(str).str.contains("High|ModelDisagreement", regex=True, na=False)
        fixed_rows = old_panel.loc[keep].copy()
        for _, r in old_panel.iterrows():
            role = str(r.get("selection_role", ""))
            code = str(r.get("R5_code", ""))
            if role.startswith("M") or role.startswith("L"):
                fixed_codes_by_role[role.split("_")[0]] = code

    scored["_match_key"] = _composition_match_key(scored)
    fixed_indices: List[int] = []
    fixed_role_map: Dict[int, Tuple[str, str]] = {}

    if not fixed_rows.empty:
        fixed_rows["_match_key"] = _composition_match_key(fixed_rows)
        map_idx = pd.Series(scored.index, index=scored["_match_key"]).to_dict()
        for _, r in fixed_rows.iterrows():
            idx = map_idx.get(r["_match_key"])
            if idx is not None:
                idx = int(idx)
                fixed_indices.append(idx)
                fixed_role_map[idx] = (str(r["selection_role"]), str(r.get("R5_code", "")))

    if args.reselect_all or len(fixed_indices) < 6:
        # Full re-selection from frozen scores using v1.1 rules.
        panel = select_r5_panel(
            scored.drop(columns=["_match_key"], errors="ignore"),
            Xz,
            args.n_select,
            args.seed,
            medium_raw_target=args.medium_raw_target,
            medium_raw_low=args.medium_raw_low,
            medium_raw_high=args.medium_raw_high,
            low_raw_target=args.low_raw_target,
            low_raw_low=args.low_raw_low,
            low_raw_high=args.low_raw_high,
            allow_same_ionizable_lipid=args.allow_same_ionizable_lipid,
        )
    else:
        x = scored.copy().reset_index(drop=True)
        chemistry_ok = pd.Series(True, index=x.index)
        if not args.allow_same_ionizable_lipid:
            chemistry_ok = ~x["same_ionizable_identity"]

        used = list(dict.fromkeys(fixed_indices))

        # Two genuinely intermediate points.
        mid_pool = x[
            chemistry_ok
            & x["all_model_raw_mean"].between(args.medium_raw_low, args.medium_raw_high)
            & x["consensus_rank"].between(0.10, 0.45)
            & (x["AD_ratio_to_train95"] <= 1.25)
            & (~x.index.isin(used))
        ].copy()
        if len(mid_pool) < 2:
            mid_pool = x[
                chemistry_ok
                & x["all_model_raw_mean"].between(args.medium_raw_low - 0.06, args.medium_raw_high + 0.05)
                & x["consensus_rank"].between(0.06, 0.55)
                & (x["AD_ratio_to_train95"] <= 1.40)
                & (~x.index.isin(used))
            ].copy()

        mid_scale = max(args.medium_raw_high - args.medium_raw_low, 0.05)
        mid_pool["mid_target_score"] = (
            -np.abs(mid_pool["all_model_raw_mean"] - args.medium_raw_target) / mid_scale
            - 0.30 * mid_pool["model_rank_disagreement"]
            - 0.05 * np.abs(mid_pool["consensus_rank"] - 0.22)
        )
        mids = greedy_select(mid_pool, Xz, 2, "mid_target_score", used, diversity_weight=0.10)
        used.extend(mids)

        # Two low, distinct-IL, in-domain controls around ~0.42.
        low_pool = x[
            chemistry_ok
            & x["all_model_raw_mean"].between(args.low_raw_low, args.low_raw_high)
            & (x["consensus_rank"] <= 0.18)
            & (x["AD_ratio_to_train95"] <= 1.15)
            & (~x.index.isin(used))
        ].copy()
        if len(low_pool) < 2:
            low_pool = x[
                chemistry_ok
                & x["all_model_raw_mean"].between(args.low_raw_low - 0.07, args.low_raw_high + 0.04)
                & (x["consensus_rank"] <= 0.25)
                & (x["AD_ratio_to_train95"] <= 1.35)
                & (~x.index.isin(used))
            ].copy()

        low_scale = max(args.low_raw_high - args.low_raw_low, 0.05)
        low_pool["low_control_score"] = (
            -np.abs(low_pool["all_model_raw_mean"] - args.low_raw_target) / low_scale
            - 0.30 * low_pool["model_rank_disagreement"]
            - 0.08 * low_pool["consensus_rank"]
        )
        lows = greedy_select(low_pool, Xz, 2, "low_control_score", used, diversity_weight=0.10)

        rows: List[pd.DataFrame] = []
        for idx in fixed_indices:
            row = x.loc[[idx]].copy()
            role, code = fixed_role_map[idx]
            row.insert(0, "selection_role", role)
            row.insert(0, "R5_code", code)
            rows.append(row)

        mid_codes = [fixed_codes_by_role.get("M1", "R5-01"), fixed_codes_by_role.get("M2", "R5-05")]
        low_codes = [fixed_codes_by_role.get("L1", "R5-04"), fixed_codes_by_role.get("L2", "R5-03")]

        for j, idx in enumerate(mids, start=1):
            row = x.loc[[idx]].copy()
            row.insert(0, "selection_role", f"M{j}_Intermediate")
            row.insert(0, "R5_code", mid_codes[j-1])
            rows.append(row)
        for j, idx in enumerate(lows, start=1):
            row = x.loc[[idx]].copy()
            row.insert(0, "selection_role", f"L{j}_LowControl")
            row.insert(0, "R5_code", low_codes[j-1])
            rows.append(row)

        panel = pd.concat(rows, ignore_index=True)
        role_order = {
            "HighConsensus": 0, "HighDiverse": 1, "Intermediate": 2,
            "LowControl": 3, "ModelDisagreement": 4,
        }
        def _rk(s: str) -> int:
            for k, v in role_order.items():
                if k in str(s):
                    return v
            return 9
        panel["_role_order"] = panel["selection_role"].map(_rk)
        panel = panel.sort_values(
            ["_role_order", "consensus_rank"], ascending=[True, False]
        ).drop(columns=["_role_order", "_match_key"], errors="ignore").reset_index(drop=True)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source.parent / ("R5_reselected_v1_1_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    )

    print("\n" + "=" * 92)
    print("[R5 v1.1 PANEL]")
    show_cols = [
        "R5_code", "selection_role", "candidate_id",
        "IL1", "IL1_molpct", "IL2", "IL2_molpct",
        "Phospholipid", "HL_molpct", "CHOL_molpct", "PEG", "PEG_molpct",
        "consensus_rank", "all_model_raw_mean", "model_rank_disagreement",
        "AD_ratio_to_train95",
    ]
    show_cols = [c for c in show_cols if c in panel.columns]
    print(panel[show_cols].to_string(index=False))

    write_reselection_outputs(
        source_workbook=source,
        output_dir=output_dir,
        panel=panel,
        scored_pool=scored.drop(columns=["_match_key"], errors="ignore"),
        args=args,
    )


# -----------------------------------------------------------------------------
# Optional prospective evaluation after experiment
# -----------------------------------------------------------------------------
def evaluate_frozen_r5(args: argparse.Namespace) -> None:
    frozen_dir = Path(args.frozen_dir).resolve()
    frozen_csv = frozen_dir / "R5_selected_12_FROZEN.csv"
    if not frozen_csv.is_file():
        raise FileNotFoundError(f"Frozen prediction file not found: {frozen_csv}")
    if not args.actual_excel:
        raise ValueError("--evaluate-only requires --actual-excel")
    frozen = pd.read_csv(frozen_csv)
    actual = pd.read_excel(args.actual_excel, sheet_name=parse_sheet(args.actual_sheet))
    id_col = find_col(actual, ["R5_code", "candidate_id", "Candidate_ID", "ID", "Sample"])
    y_col = find_col(actual, [args.actual_column])
    if id_col is None or y_col is None:
        raise ValueError(f"Measured workbook must contain R5_code and {args.actual_column!r}")
    meas = actual[[id_col, y_col]].copy()
    meas.columns = ["R5_code", "actual"]
    meas["actual"] = pd.to_numeric(meas["actual"], errors="coerce")
    merged = frozen.merge(meas, on="R5_code", how="left")
    valid = merged.dropna(subset=["actual", "consensus_rank"]).copy()
    if len(valid) < 6:
        raise ValueError(f"Only {len(valid)} measured R5 rows could be matched.")

    y_true = valid["actual"].to_numpy(dtype=float)
    y_pred_rank = valid["consensus_rank"].to_numpy(dtype=float)
    rho = float(spearmanr(y_true, y_pred_rank).correlation) if HAS_SCIPY else np.nan

    # Raw consensus prediction is only a descriptive numerical prediction because it averages
    # differently calibrated model outputs. Per-model errors are also exported below.
    raw_pred = valid["all_model_raw_mean"].to_numpy(dtype=float) if "all_model_raw_mean" in valid else np.full(len(valid), np.nan)
    rmse = float(np.sqrt(np.mean((y_true - raw_pred) ** 2))) if np.isfinite(raw_pred).all() else np.nan
    mae = float(np.mean(np.abs(y_true - raw_pred))) if np.isfinite(raw_pred).all() else np.nan
    ss_res = float(np.sum((y_true - raw_pred) ** 2)) if np.isfinite(raw_pred).all() else np.nan
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if np.isfinite(ss_res) and ss_tot > 0 else np.nan

    def top_overlap(k: int) -> float:
        kk = min(k, len(valid))
        true_top = set(np.argsort(y_true)[-kk:])
        pred_top = set(np.argsort(y_pred_rank)[-kk:])
        return len(true_top & pred_top) / kk

    valid["role_group"] = np.select(
        [
            valid["selection_role"].str.contains("High", na=False),
            valid["selection_role"].str.contains("Intermediate", na=False),
            valid["selection_role"].str.contains("Low", na=False),
        ],
        ["High", "Intermediate", "Low"],
        default="Disagreement",
    )
    group_summary = valid.groupby("role_group", as_index=False).agg(
        n=("actual", "size"),
        mean_actual=("actual", "mean"),
        median_actual=("actual", "median"),
        mean_consensus_rank=("consensus_rank", "mean"),
    )

    summary = pd.DataFrame([{
        "n": len(valid),
        "Spearman_actual_vs_frozen_consensus_rank": rho,
        "Top3_overlap": top_overlap(3),
        "Top5_overlap": top_overlap(5),
        "R2_raw_all_model_mean": r2,
        "RMSE_raw_all_model_mean": rmse,
        "MAE_raw_all_model_mean": mae,
        "note": "R2/RMSE/MAE are valid only if actual column uses the same R1-R4 normalized target scale.",
    }])

    model_rows = []
    for c in [c for c in valid.columns if c.startswith("pred_")]:
        p = valid[c].to_numpy(dtype=float)
        if not np.isfinite(p).all():
            continue
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        model_rows.append({
            "model": c.removeprefix("pred_"),
            "R2": 1 - np.sum((y_true - p) ** 2) / sst if sst > 0 else np.nan,
            "RMSE": float(np.sqrt(np.mean((y_true - p) ** 2))),
            "MAE": float(np.mean(np.abs(y_true - p))),
            "Spearman": float(spearmanr(y_true, p).correlation) if HAS_SCIPY else np.nan,
        })
    model_metrics = pd.DataFrame(model_rows)

    out = frozen_dir / "R5_PROSPECTIVE_EVALUATION.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        valid.to_excel(writer, sheet_name="matched_results", index=False)
        group_summary.to_excel(writer, sheet_name="high_mid_low", index=False)
        model_metrics.to_excel(writer, sheet_name="per_model_metrics", index=False)
    print(f"[Evaluate] Prospective evaluation written to: {out}")
    print(summary.to_string(index=False))


# -----------------------------------------------------------------------------
# Main parent workflow
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prospective multi-model R5 LNP candidate selector")
    p.add_argument("--train-excel", default="", help="Final R1-R4 normalized 1.35 workbook")
    p.add_argument("--tree-script", default="", help="Final R1-R4 publication tree-model Python script")
    p.add_argument("--sheet-name", default=DEFAULT_SHEET)
    p.add_argument("--output-dir", default="")

    p.add_argument("--candidate-source", choices=["virtual", "excel", "combined"], default="virtual")
    p.add_argument("--candidate-excel", default="")
    p.add_argument("--candidate-sheet", default="0")
    p.add_argument("--max-virtual-candidates", type=int, default=12000)
    p.add_argument("--foundation-pool-size", type=int, default=2000)
    p.add_argument("--n-select", type=int, default=12)

    # v1.1 validation-balance targets on the same normalized DC2.4 response scale.
    p.add_argument("--medium-raw-target", type=float, default=0.65)
    p.add_argument("--medium-raw-low", type=float, default=0.58)
    p.add_argument("--medium-raw-high", type=float, default=0.72)
    p.add_argument("--low-raw-target", type=float, default=0.42)
    p.add_argument("--low-raw-low", type=float, default=0.32)
    p.add_argument("--low-raw-high", type=float, default=0.50)
    p.add_argument(
        "--allow-same-ionizable-lipid",
        action="store_true",
        help="Allow nominal dual-IL candidates with IL1 == IL2 (not recommended).",
    )

    # Re-select from an already scored v1.0 workbook WITHOUT retraining any model.
    p.add_argument(
        "--reselect-workbook",
        default="",
        help="Existing R5_prospective_selection_results.xlsx; reuses frozen predictions.",
    )
    p.add_argument(
        "--reselect-all",
        action="store_true",
        help="With --reselect-workbook, reselect all 12 instead of preserving existing High + D1.",
    )

    p.add_argument("--aux-top-k", type=int, default=8)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--tune-iter", type=int, default=12)
    p.add_argument("--group-round-decimals", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--pdi-max", type=float, default=0.5)
    p.add_argument("--size-min", type=float, default=30.0)
    p.add_argument("--size-max", type=float, default=300.0)
    p.add_argument("--allow-missing-qc", action="store_true")

    p.add_argument("--skip-tabpfn", action="store_true")
    p.add_argument("--skip-tabfm", action="store_true")
    p.add_argument("--tabpfn-backend", choices=["auto", "client", "local"], default="auto")

    # Internal child-mode args.
    p.add_argument("--foundation-child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--payload-dir", default="", help=argparse.SUPPRESS)

    # Optional post-experiment evaluation.
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--frozen-dir", default="")
    p.add_argument("--actual-excel", default="")
    p.add_argument("--actual-sheet", default="0")
    p.add_argument("--actual-column", default=TARGET_COLUMN)
    return p.parse_args()


def parent_main(args: argparse.Namespace) -> None:
    started = time.time()
    pipeline, ctx = build_training_context(args)

    pool = load_candidate_pool(args, pipeline, ctx)
    _, _, X_pool_selected = build_candidate_features(pipeline, ctx, pool)

    # Fit strong tree models on the entire frozen R1-R4 dataset and predict the full virtual pool.
    tree_pred, tree_status = fit_tree_models(pipeline, ctx, X_pool_selected)
    tree_full = pd.concat([pool.reset_index(drop=True), tree_pred.reset_index(drop=True)], axis=1)
    tree_full = add_rank_columns(tree_full, list(tree_pred.columns))

    # Reduce to a common multi-model pool while preserving top, middle, low, and disagreement.
    foundation_pool = build_tree_prefilter(tree_full, args.foundation_pool_size, args.seed)
    common_indices = foundation_pool["candidate_id"].map(
        pd.Series(np.arange(len(pool)), index=pool["candidate_id"]).to_dict()
    ).astype(int).to_numpy()
    X_common = X_pool_selected.iloc[common_indices].reset_index(drop=True)
    foundation_pool = foundation_pool.reset_index(drop=True)

    # Applicability domain is calculated before final selection.
    foundation_pool, ad_info, X_common_z = add_applicability_domain(
        foundation_pool, ctx.X_selected, X_common, k=5
    )

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        Path(ctx.workbook_path).resolve().parent / "lnp_outputs" /
        ("R5_prospective_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = output_dir / "_foundation_payload"
    payload.mkdir(parents=True, exist_ok=True)
    ctx.X_selected.to_csv(payload / "X_train_selected.csv", index=False)
    pd.DataFrame({"TE": ctx.y}).to_csv(payload / "y_train.csv", index=False)
    X_common.to_csv(payload / "X_foundation_pool_selected.csv", index=False)

    foundation_pred = run_foundation_child(args, payload)
    foundation_status_path = payload / "foundation_status.csv"
    foundation_status = pd.read_csv(foundation_status_path) if foundation_status_path.is_file() else pd.DataFrame()
    if not foundation_pred.empty:
        foundation_pred = foundation_pred.drop(columns=["pool_row"], errors="ignore")
        foundation_pool = pd.concat([foundation_pool, foundation_pred.reset_index(drop=True)], axis=1)

    # All requested available models now vote on exactly the same candidate points.
    scored, rank_cols = add_consensus_scores(foundation_pool)
    scored, ad_info, X_common_z = add_applicability_domain(scored, ctx.X_selected, X_common, k=5)

    panel = select_r5_panel(
        scored, X_common_z, args.n_select, args.seed,
        medium_raw_target=args.medium_raw_target,
        medium_raw_low=args.medium_raw_low,
        medium_raw_high=args.medium_raw_high,
        low_raw_target=args.low_raw_target,
        low_raw_low=args.low_raw_low,
        low_raw_high=args.low_raw_high,
        allow_same_ionizable_lipid=args.allow_same_ionizable_lipid,
    )

    print("\n" + "=" * 92)
    print("[R5 SELECTED PANEL]")
    show_cols = [
        "R5_code", "selection_role", "IL1", "IL1_molpct", "IL2", "IL2_molpct",
        "Phospholipid", "HL_molpct", "CHOL_molpct", "PEG", "PEG_molpct",
        "consensus_rank", "model_rank_disagreement", "n_models_top20", "AD_ratio_to_train95",
    ]
    show_cols = [c for c in show_cols if c in panel.columns]
    print(panel[show_cols].to_string(index=False))

    write_outputs(
        output_dir=output_dir,
        panel=panel,
        scored_pool=scored,
        tree_full_pool=tree_full,
        ctx=ctx,
        tree_status=tree_status,
        foundation_status=foundation_status,
        ad_info=ad_info,
        args=args,
    )
    print(f"[Runtime] {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    args = parse_args()
    if args.foundation_child:
        if not args.payload_dir:
            raise ValueError("Internal foundation child requires --payload-dir")
        foundation_child_main(args)
    elif args.evaluate_only:
        evaluate_frozen_r5(args)
    elif args.reselect_workbook:
        reselect_from_existing_workbook(args)
    else:
        parent_main(args)
