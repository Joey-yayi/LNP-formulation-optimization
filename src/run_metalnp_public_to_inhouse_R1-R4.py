#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py

MetaLNP-style public-to-in-house benchmark for the final R1-R4 DC2.4 LNP project.

What this script does
---------------------
This script is intentionally separated from your original in-house ML model.

It does NOT train:
    - inhouse_RF
    - inhouse_GB
    - inhouse_HGB
    - inhouse_TabPFN
    - any other model using only your in-house data

It ONLY trains MetaLNP-style public meta-learning models on public tasks, then
tests them on your in-house LNP data.

Two evaluation modes are reported:
A) zero-shot:
   public meta-trained model -> directly predicts all in-house rows
   mainly evaluated on percentile target, because public/in-house raw scales differ.

B) few-shot target adaptation, close to the paper's support/query idea:
   public meta-trained model -> adapt on your support set -> evaluate on your query/test set
   default: support = earlier rounds, query = latest round if Round is available.
   fallback: repeated random support/query splits.

Why this script exists
----------------------
Your own model has already been evaluated elsewhere, with CV R2 around ~0.65-0.68
and holdout R2 around ~0.78-0.82 depending on the feature/model setting.
This script produces only the MetaLNP-side numbers so you can compare externally.

Paper-faithful design choices
-----------------------------
- Public support/query tasks.
- MAML, FoMAML, MetaSGD.
- Support/query sizes default to 10/10.
- Fingerprint-style representation; default Morgan r=4 / 2048 bits when SMILES exist.
- Pure PyTorch implementation of the MAML/FoMAML/MetaSGD concept; this is a MetaLNP-style benchmark, not an exact reproduction of the original repository.
- No Chemprop/GNN dependency is required.

Recommended first run in PowerShell
------------------------------------
& "C:\Users\ASUS\Desktop\AI screen LNP python and excel\.venv\Scripts\python.exe" `
  "C:\Users\ASUS\Downloads\run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py" `
  --quick --models maml,metasgd --target "Normalized for DC2.4"

Formal DC2.4 run
----------------
& "C:\Users\ASUS\Desktop\AI screen LNP python and excel\.venv\Scripts\python.exe" `
  "C:\Users\ASUS\Downloads\run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py" `
  --models maml,fomaml,metasgd,supervised_ann `
  --n_repeats 20 `
  --target "Normalized for DC2.4"

Formal HaCaT run
----------------
& "C:\Users\ASUS\Desktop\AI screen LNP python and excel\.venv\Scripts\python.exe" `
  "C:\Users\ASUS\Downloads\run_metalnp_public_to_inhouse_R1R4_PUBLICATION.py" `
  --models maml,fomaml,metasgd,supervised_ann `
  --n_repeats 20 `
  --target "Normalized for HaCaT"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit(
        "\n[ERROR] PyTorch is required.\n"
        "Install PyTorch in the SAME environment used to run this script.\n"
        f"Original error: {exc}\n"
    )

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


# =============================================================================
# Default paths
# =============================================================================

DEFAULT_PROJECT_DIR = Path.home() / "Desktop" / "AI screen LNP python and excel" / "PyCharm 有效代码"
DEFAULT_INHOUSE_XLSX = DEFAULT_PROJECT_DIR / "8.02 publish" / "R1-4 all LNP normalized 1.35 (new).xlsx"

# Official MetaLNP processed public data path, if you cloned their repo.
DEFAULT_METALNP_REPO = Path.home() / "Documents" / "GitHub" / "MetaLNPs"
DEFAULT_PUBLIC_TRAIN = DEFAULT_METALNP_REPO / "data" / "Processed" / "siRNAho" / "train_df_task_nosirna_clean.csv"
DEFAULT_PUBLIC_VAL = DEFAULT_METALNP_REPO / "data" / "Processed" / "siRNAho" / "meta_val_stop_df_siRNA_clean.csv"
DEFAULT_PUBLIC_HOLDOUT = DEFAULT_METALNP_REPO / "data" / "Processed" / "siRNAho" / "holdout_df_task_sirna2_clean.csv"

# Optional fallback: your prepared LANCE public task folder can be passed by --public_features_csv and --public_manifest_csv.
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_DIR / "lnp_outputs" / "MetaLNP_ONLY_public_to_inhouse"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    inhouse_xlsx: str
    public_train_csv: str
    public_val_csv: str
    public_holdout_csv: str
    public_features_csv: str
    public_manifest_csv: str
    output_dir: str

    target: str = "Normalized for DC2.4"
    inhouse_main_sheet: str = "Round1&2&3&4"
    inhouse_smiles_sheet: str = "SMILES NAME"

    public_target_col: str = ""
    public_task_col: str = ""
    public_smiles_col: str = ""

    models: str = "maml,fomaml,metasgd,supervised_ann"
    n_repeats: int = 20
    seed: int = 42
    quick: bool = False

    support_size_public: int = 10
    query_size_public: int = 10
    tasks_per_batch: int = 8
    meta_iters: int = 800

    # Target adaptation settings
    split_mode: str = "random"   # publication benchmark default; optional: round
    support_rounds: str = ""     # e.g. "1,2"
    query_rounds: str = ""       # e.g. "3"
    random_support_fraction: float = 0.80

    # Model hyperparameters
    hidden_dim: int = 128
    dropout: float = 0.10
    adapt_steps: int = 3
    adapt_lr: float = 1e-3
    meta_lr: float = 3e-4
    metasgd_max_inner_lr: float = 0.02
    supervised_lr: float = 3e-4
    supervised_epochs: int = 800

    # Feature settings
    morgan_radius: int = 4
    morgan_bits: int = 2048
    use_morgan: bool = True
    use_numeric_composition: bool = True
    use_public_existing_fp: bool = True

    # Cleaning / publication-population alignment
    apply_qc_filter: bool = True
    qc_pdi_max: float = 0.5
    qc_size_min: float = 30.0
    qc_size_max: float = 300.0
    require_complete_qc: bool = True
    expected_inhouse_n: int = 104
    enforce_expected_inhouse_n: bool = True

    device: str = "cpu"


# =============================================================================
# Utilities
# =============================================================================

def clean_col(c) -> str:
    return re.sub(r"\s+", " ", str(c).strip())


def norm_token(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def read_excel_target_sheet(path: Path, target: str, preferred_sheet: str = "") -> Tuple[pd.DataFrame, str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"In-house workbook not found: {path}")
    xls = pd.ExcelFile(path)
    sheets = list(xls.sheet_names)

    if preferred_sheet:
        # Resolve harmless worksheet-name differences such as a trailing space.
        wanted = clean_col(preferred_sheet).lower()
        matches = [s for s in sheets if clean_col(s).lower() == wanted]
        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely resolve preferred sheet {preferred_sheet!r}. "
                f"Workbook sheets: {sheets}"
            )
        resolved_sheet = matches[0]
        df = pd.read_excel(path, sheet_name=resolved_sheet)
        df.columns = [clean_col(c) for c in df.columns]
        if target not in df.columns:
            raise ValueError(
                f"Sheet {resolved_sheet!r} does not contain the exact target column {target!r}."
            )
        return df, resolved_sheet, sheets

    # Exact target match
    for s in sheets:
        df = pd.read_excel(path, sheet_name=s)
        df.columns = [clean_col(c) for c in df.columns]
        if target in df.columns:
            return df, s, sheets

    # Case-insensitive match
    target_low = clean_col(target).lower()
    for s in sheets:
        df = pd.read_excel(path, sheet_name=s)
        df.columns = [clean_col(c) for c in df.columns]
        low_map = {c.lower(): c for c in df.columns}
        if target_low in low_map:
            return df.rename(columns={low_map[target_low]: target}), s, sheets

    raise ValueError(
        f"No sheet contains target column {target!r}. Workbook sheets: {sheets}"
    )


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    low = {clean_col(c).lower(): c for c in df.columns}
    for c in candidates:
        if clean_col(c).lower() in low:
            return low[clean_col(c).lower()]
    return None


def find_col_by_keywords(df: pd.DataFrame, keywords: Sequence[str], exclude: Sequence[str] = ()) -> Optional[str]:
    excl = [x.lower() for x in exclude]
    for c in df.columns:
        cl = clean_col(c).lower()
        if any(e in cl for e in excl):
            continue
        if any(k.lower() in cl for k in keywords):
            return c
    return None


def to_percentile(y: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(y, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def corr_pearson(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    if pearsonr is not None:
        return float(pearsonr(y_true, y_pred)[0])
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def corr_spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    if spearmanr is not None:
        return float(spearmanr(y_true, y_pred)[0])
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def top_precision(y_true, y_pred, frac=0.20) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return np.nan
    k = max(1, int(math.ceil(n * frac)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return float(len(true_top & pred_top) / k)


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_total = int(len(y_true))
    n_finite = int(finite.sum())

    if n_finite < 2:
        return {
            "n_test": n_total,
            "n_finite": n_finite,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan,
            "Top20_precision": np.nan,
            "valid_prediction": False,
        }

    yt = y_true[finite]
    yp = y_pred[finite]
    return {
        "n_test": n_total,
        "n_finite": n_finite,
        "R2": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
        "RMSE": rmse(yt, yp),
        "MAE": float(mean_absolute_error(yt, yp)),
        "Pearson": corr_pearson(yt, yp),
        "Spearman": corr_spearman(yt, yp),
        "Top20_precision": top_precision(yt, yp, 0.20),
        "valid_prediction": True,
    }


# =============================================================================
# Public MetaLNP data: target, task, features
# =============================================================================

LEAK_WORDS = [
    "quantified_delivery", "delivery", "response", "target", "label", "raw",
    "normalized", "normolized", "transfection", "efficiency", "rlu", "luciferase",
    "y_task", "hit_top", "rank", "score", "prediction", "pred"
]

ID_WORDS = [
    "id", "name", "smiles", "publication", "source", "split", "task", "cell",
    "cargo", "route", "readout", "criterion"
]


def detect_public_target(df: pd.DataFrame, user_col: str = "") -> str:
    if user_col and user_col in df.columns:
        return user_col
    candidates = [
        "quantified_delivery",
        "Quantified_delivery",
        "Experiment_value",
        "experiment_value",
        "y",
        "target",
        "label",
        "response",
    ]
    c = find_col(df, candidates)
    if c:
        return c
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    raise ValueError(
        "Could not detect public target column. Pass --public_target_col.\n"
        f"Numeric columns include: {numeric[:80]}"
    )


def detect_public_task_col(df: pd.DataFrame, user_col: str = "") -> Optional[str]:
    if user_col and user_col in df.columns:
        return user_col
    candidates = [
        "task_id", "Task_ID", "task", "Task", "meta_task", "task_name",
        "split_name_for_normalization", "split_name", "normalization_group",
    ]
    c = find_col(df, candidates)
    if c:
        return c

    # If no explicit task column, use an object column with repeated groups.
    object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    n = len(df)
    scored = []
    for c in object_cols:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= max(3, n // 5):
            scored.append((nunique, c))
    if scored:
        scored.sort()
        return scored[0][1]
    return None


def detect_smiles_col(df: pd.DataFrame, user_col: str = "") -> Optional[str]:
    if user_col and user_col in df.columns:
        return user_col
    c = find_col(df, [
        "SMILES", "smiles", "Smiles",
        "canonical_smiles", "Canonical_SMILES",
        "ionizable_smiles", "Ionizable_SMILES",
        "lipid_smiles", "Lipid_SMILES",
    ])
    if c:
        return c

    # Heuristic SMILES-like text column
    def looks_like_smiles(s):
        s = str(s).strip()
        if len(s) < 6:
            return 0
        allowed = set("CNOSPFIBrClcnospfibr()[]=#@+-/\\.%0123456789H")
        frac = sum(ch in allowed for ch in s) / max(len(s), 1)
        return int(frac > 0.85 and ("C" in s or "c" in s))
    best, best_score = None, 0
    for col in df.select_dtypes(exclude=[np.number]).columns:
        vals = df[col].dropna().astype(str).head(50)
        if len(vals) == 0:
            continue
        score = sum(looks_like_smiles(v) for v in vals)
        if score > best_score:
            best_score, best = score, col
    return best if best_score >= 5 else None


def detect_existing_fp_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        cl = clean_col(c).lower()
        if (
            re.match(r"^(fp|bit|morgan|ecfp|weighted_morgan)[_\- ]?\d+$", cl)
            or re.match(r"^\d+$", cl)
            or ("morgan" in cl and re.search(r"\d+$", cl))
        ):
            if pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
    if not out:
        return []
    def key(c):
        m = re.search(r"(\d+)$", str(c))
        return int(m.group(1)) if m else 10**9
    out = sorted(out, key=key)
    return out if len(out) >= 16 else []


def morgan_fp(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    if not HAS_RDKIT or pd.isna(smiles):
        return arr
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return arr
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr.astype(np.float32)


def detect_public_composition_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    # These names cover official MetaLNP-like, LNPDB-like, and LANCE-like tables.
    return {
        "ionizable_molpct": find_col_by_keywords(df, ["ionizable", "mol"], exclude=["name", "smiles", "id"])
                             or find_col_by_keywords(df, ["il", "mol"], exclude=["name", "smiles", "id"]),
        "helper_molpct": find_col_by_keywords(df, ["helper", "mol"], exclude=["name", "smiles", "id"])
                         or find_col_by_keywords(df, ["phospholipid", "mol"], exclude=["name", "smiles", "id"]),
        "cholesterol_molpct": find_col_by_keywords(df, ["chol", "mol"], exclude=["name", "smiles", "id"]),
        "peg_molpct": find_col_by_keywords(df, ["peg", "mol"], exclude=["name", "smiles", "id"]),
        "np_ratio": find_col_by_keywords(df, ["n/p"], exclude=[])
                    or find_col_by_keywords(df, ["np_ratio"], exclude=[]),
    }


def build_public_features(df: pd.DataFrame, cfg: Config, target_col: str, task_col: Optional[str], smiles_col: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    parts = []
    report = {}

    # 1. Canonical numeric composition features.
    comp = detect_public_composition_cols(df)
    report["public_composition_columns"] = comp
    comp_df = pd.DataFrame(index=df.index)
    for new_name, old in comp.items():
        if old is not None:
            comp_df[new_name] = pd.to_numeric(df[old], errors="coerce")
        else:
            comp_df[new_name] = 0.0
    comp_df = comp_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if cfg.use_numeric_composition:
        parts.append(comp_df.astype(np.float32))

    # 2. Fingerprints: prefer SMILES-generated Morgan r4/2048; fallback existing fp columns.
    fp_used = "none"
    if cfg.use_morgan and smiles_col is not None:
        fp_mat = np.vstack([morgan_fp(s, cfg.morgan_radius, cfg.morgan_bits) for s in df[smiles_col].tolist()])
        if float(np.abs(fp_mat).sum()) > 0:
            parts.append(pd.DataFrame(fp_mat, index=df.index, columns=[f"morgan_{i}" for i in range(cfg.morgan_bits)]))
            fp_used = f"SMILES_Morgan_r{cfg.morgan_radius}_{cfg.morgan_bits}"

    if fp_used == "none" and cfg.use_public_existing_fp:
        fp_cols = detect_existing_fp_cols(df)
        if fp_cols:
            fp_raw = df[fp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if fp_raw.shape[1] != cfg.morgan_bits:
                fp_pad = np.zeros((len(df), cfg.morgan_bits), dtype=np.float32)
                n = min(fp_raw.shape[1], cfg.morgan_bits)
                fp_pad[:, :n] = fp_raw[:, :n]
                fp_raw = fp_pad
            parts.append(pd.DataFrame(fp_raw, index=df.index, columns=[f"morgan_{i}" for i in range(cfg.morgan_bits)]))
            fp_used = f"existing_fp_padded_to_{cfg.morgan_bits}"

    if not parts:
        raise ValueError("No public features could be built.")

    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    report.update({
        "public_n_rows": int(len(df)),
        "public_smiles_col": smiles_col,
        "public_fingerprint_source": fp_used,
        "public_n_features": int(X.shape[1]),
    })
    return X, report


def make_public_task_ids(df: pd.DataFrame, y: np.ndarray, task_col: Optional[str], cfg: Config) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(cfg.seed)
    min_n = cfg.support_size_public + cfg.query_size_public

    if task_col is None:
        base = pd.Series(["all_public"] * len(df), index=df.index)
    else:
        base = df[task_col].astype(str).fillna("unknown")

    task_ids = np.array([""] * len(df), dtype=object)
    task_sizes = {}
    for group_name, idx_values in base.groupby(base).groups.items():
        idx = np.array(list(idx_values))
        if len(idx) < min_n:
            continue
        idx = rng.permutation(idx)
        # Split large source groups into 20-sample tasks, as in MetaLNP-like task size.
        chunk_size = min_n
        n_chunks = len(idx) // chunk_size
        for j in range(n_chunks):
            chunk = idx[j * chunk_size:(j + 1) * chunk_size]
            tid = f"{group_name}__chunk{j:03d}"
            task_ids[chunk] = tid
            task_sizes[tid] = len(chunk)

    keep = task_ids != ""
    report = {
        "base_task_col": task_col,
        "min_task_size": min_n,
        "n_public_rows_before_task_filter": int(len(df)),
        "n_public_rows_after_task_filter": int(keep.sum()),
        "n_tasks": int(len(task_sizes)),
        "task_size_min": int(min(task_sizes.values())) if task_sizes else 0,
        "task_size_max": int(max(task_sizes.values())) if task_sizes else 0,
    }
    return task_ids, report


class PublicTaskSet:
    def __init__(self, X: np.ndarray, y: np.ndarray, task_ids: np.ndarray, cfg: Config):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
        self.task_ids = np.asarray(task_ids)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.groups = {}
        min_n = cfg.support_size_public + cfg.query_size_public
        for tid in pd.unique(self.task_ids):
            idx = np.where(self.task_ids == tid)[0]
            if len(idx) >= min_n:
                self.groups[str(tid)] = idx
        self.keys = list(self.groups.keys())
        if len(self.keys) < 3:
            raise ValueError(f"Too few public tasks: {len(self.keys)}. Need at least 3 usable tasks.")

    def sample(self):
        tid = self.rng.choice(self.keys)
        idx = self.groups[tid]
        chosen = self.rng.choice(idx, size=self.cfg.support_size_public + self.cfg.query_size_public, replace=False)

        # Stratified-ish by response: alternate sorted y into support/query.
        chosen = chosen[np.argsort(self.y[chosen, 0])]
        s, q = [], []
        for i, row_idx in enumerate(chosen):
            if (i % 2 == 0 and len(s) < self.cfg.support_size_public) or len(q) >= self.cfg.query_size_public:
                s.append(row_idx)
            else:
                q.append(row_idx)

        return (
            torch.tensor(self.X[s], dtype=torch.float32),
            torch.tensor(self.y[s], dtype=torch.float32),
            torch.tensor(self.X[q], dtype=torch.float32),
            torch.tensor(self.y[q], dtype=torch.float32),
        )


# =============================================================================
# In-house target features, based on your original workbook logic
# =============================================================================

LIPID_ALIASES = {
    "DLIN-MC3-DMA": "MC3", "DLIN-MC3": "MC3", "MC-3": "MC3", "MC3": "MC3",
    "SM-102": "SM102", "SM 102": "SM102", "SM102": "SM102",
    "C12200": "C12-200", "C12 200": "C12-200", "C12-200": "C12-200",
    "CKK-E12": "CKK-E12", "CKKE12": "CKK-E12", "CKK E12": "CKK-E12", "CCK12": "CKK-E12",
    "ALC0315": "ALC-0315", "ALC 0315": "ALC-0315", "ALC-0315": "ALC-0315",
    "DMG-PEG": "DMG-PEG2000", "DMG-PEG-2000": "DMG-PEG2000", "DMGPEG2000": "DMG-PEG2000",
    "PEG2000-DMG": "DMG-PEG2000", "PEG-DMG": "DMG-PEG2000",
    "ALC-0159": "C14-PEG", "ALC0159": "C14-PEG", "C14PEG": "C14-PEG",
    "PEG-MANNOSE": "PEG-Mannose", "MANNOSE-PEG": "PEG-Mannose", "PEG MANNOSE": "PEG-Mannose",
    "CHOL": "Cholesterol", "CHOLESTEROL": "Cholesterol",
}


def normalize_lipid_name(name):
    """
    Canonicalize formulation lipid names using the same publication vocabulary
    as the final R1-R4 tree-model pipeline.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None

    s = str(name).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "-", "无"}:
        return None

    candidates = [s]
    base = re.sub(r"\([^)]*\)", "", s).strip()
    inside = re.findall(r"\(([^)]*)\)", s)
    if base and base != s:
        candidates.append(base)
    candidates.extend(x.strip() for x in inside if x.strip())

    known = {
        "MC3", "ALC-0315", "SM102", "C12-200", "CKK-E12", "DOTAP", "DODAP",
        "DSPC", "DOPE", "Cholesterol", "DMG-PEG2000", "C14-PEG", "PEG-Mannose",
    }

    for candidate in candidates:
        key = candidate.upper().replace("_", "-").replace("  ", " ").strip()
        if key in LIPID_ALIASES:
            return LIPID_ALIASES[key]
        for standard in known:
            if standard.upper() == key:
                return standard

    return s


def detect_inhouse_columns(df: pd.DataFrame, target: str) -> Dict[str, Optional[str]]:
    # Your original code's standard mapping, with fallbacks.
    mapping = {
        "IL1": find_col(df, ["Ionizable_Lipid_1", "IL1", "离子化脂质1", "Ionizable lipid 1"]),
        "IL2": find_col(df, ["Ionizable_Lipid_2", "IL2", "离子化脂质2", "Ionizable lipid 2"]),
        "IL1_molpct": find_col(df, ["IL1_Mol_Percent", "IL1_molpct", "IL1 mol%", "IL1 mol percent"]),
        "IL2_molpct": find_col(df, ["IL2_Mol_Percent", "IL2_molpct", "IL2 mol%", "IL2 mol percent"]),
        "Phospholipid": find_col(df, ["Phospholipid", "磷脂", "Helper lipid", "helper_lipid"]),
        "HL_molpct": find_col(df, ["Phospholipid_Mol_Percent", "HL_molpct", "helper_molpct", "HL mol%"]),
        "CHOL_molpct": find_col(df, ["Cholesterol_Mol_Percent", "CHOL_molpct", "cholesterol_molpct", "CHOL mol%"]),
        "PEG": find_col(df, ["PEG类型", "PEG", "PEG_lipid", "peg_lipid"]),
        "PEG_molpct": find_col(df, ["PEG_Mol_Percent", "PEG_molpct", "peg_molpct", "PEG mol%"]),
        "Round": find_col(df, ["Round", "round", "Batch", "batch"]),
        "target": target,
    }

    # Fuzzy fallbacks
    if mapping["IL1"] is None:
        mapping["IL1"] = find_col_by_keywords(df, ["ionizable", "1"], exclude=["mol", "percent", "target"])
    if mapping["IL2"] is None:
        mapping["IL2"] = find_col_by_keywords(df, ["ionizable", "2"], exclude=["mol", "percent", "target"])

    if mapping["IL1_molpct"] is None:
        mapping["IL1_molpct"] = find_col_by_keywords(df, ["il1", "mol"], exclude=[])
    if mapping["IL2_molpct"] is None:
        mapping["IL2_molpct"] = find_col_by_keywords(df, ["il2", "mol"], exclude=[])
    if mapping["HL_molpct"] is None:
        mapping["HL_molpct"] = find_col_by_keywords(df, ["phospholipid", "mol"], exclude=[]) or find_col_by_keywords(df, ["hl", "mol"], exclude=[])
    if mapping["CHOL_molpct"] is None:
        mapping["CHOL_molpct"] = find_col_by_keywords(df, ["chol", "mol"], exclude=[])
    if mapping["PEG_molpct"] is None:
        mapping["PEG_molpct"] = find_col_by_keywords(df, ["peg", "mol"], exclude=["type", "name"])

    return mapping


def detect_smiles_and_name_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    smiles_col = detect_smiles_col(df, "")
    name_col = None
    for c in df.columns:
        cl = clean_col(c).lower()
        if c != smiles_col and any(k in cl for k in ["smiles name", "lipid", "name", "compound", "molecule", "名称", "脂质"]):
            name_col = c
            break
    if name_col is None:
        # fallback to first non-smiles object column
        for c in df.select_dtypes(exclude=[np.number]).columns:
            if c != smiles_col:
                name_col = c
                break
    return name_col, smiles_col


def load_inhouse_smiles_reference(path: Path, preferred_sheet: str) -> Tuple[Dict[str, str], Dict]:
    xls = pd.ExcelFile(path)
    sheets = list(xls.sheet_names)
    order = []
    if preferred_sheet and preferred_sheet in sheets:
        order.append(preferred_sheet)
    order += [s for s in sheets if s not in order]

    smiles_map = {}
    scanned = []
    for s in order:
        try:
            df2 = pd.read_excel(path, sheet_name=s)
        except Exception:
            continue
        df2.columns = [clean_col(c) for c in df2.columns]
        name_col, smiles_col = detect_smiles_and_name_cols(df2)
        scanned.append({"sheet": s, "name_col": name_col, "smiles_col": smiles_col, "rows": int(df2.shape[0])})
        if not name_col or not smiles_col:
            continue
        for _, row in df2.iterrows():
            nm = normalize_lipid_name(row.get(name_col))
            smi = str(row.get(smiles_col, "")).strip()
            if nm and smi and smi.lower() not in {"nan", "none", "-", ""}:
                smiles_map[norm_token(nm)] = smi
        if smiles_map:
            break

    report = {
        "preferred_sheet": preferred_sheet,
        "scanned_sheets": scanned,
        "n_smiles_loaded": len(smiles_map),
        "smiles_keys_preview": list(smiles_map.keys())[:30],
    }
    return smiles_map, report


def apply_inhouse_qc(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.apply_qc_filter:
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    size_col = find_col_by_keywords(df, ["z-ave", "zave", "size", "diameter", "particle_size", "粒径"])
    pdi_col = find_col_by_keywords(df, ["pdi", "polydispers", "多分散"])
    bad = pd.Series(False, index=df.index)
    reason = pd.Series([""] * len(df), index=df.index)

    if pdi_col:
        pdi = pd.to_numeric(df[pdi_col], errors="coerce")
        b = (pdi > cfg.qc_pdi_max).fillna(False)
        bad |= b
        reason = reason.mask(b, reason + f"PDI>{cfg.qc_pdi_max};")

    if size_col:
        size = pd.to_numeric(df[size_col], errors="coerce")
        b = ((size < cfg.qc_size_min) | (size > cfg.qc_size_max)).fillna(False)
        bad |= b
        reason = reason.mask(b, reason + f"size_out_of_range;")

    removed = df[bad].copy()
    if len(removed):
        removed["__QC_reason__"] = [reason.loc[i].rstrip(";") for i in removed.index]
    kept = df[~bad].reset_index(drop=True)
    removed = removed.reset_index(drop=True)
    print(f"[In-house QC] {len(df)} -> {len(kept)} rows; removed {len(removed)} rows.")
    return kept, removed



def detect_candidate_id_column_by_values(df: pd.DataFrame) -> Optional[str]:
    """Detect the formulation/candidate ID column from R2/R3/R4 labels and numeric R1 IDs."""
    common = find_col(df, ["candidate_id", "Candidate_ID", "Candidate ID", "ID", "No", "No."])
    if common is not None:
        return common

    best_col = None
    best_score = -1
    for col in df.columns:
        values = df[col].astype(str).str.strip()
        round_hits = values.str.match(r"^R[234]\s*[-_ ]?\s*0*\d+", case=False, na=False).sum()
        numeric_hits = values.str.match(r"^\d+(?:\.0+)?$", na=False).sum()
        score = int(round_hits) * 10 + min(int(numeric_hits), 40)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_score >= 20 else None


def infer_round_from_candidate_id(candidate_id) -> str:
    """
    Recover the actual experimental round from historical labels.

    Examples:
      R1-17 (R3-28) -> R3
      R4-25 (R3-11) -> R4

    The highest R-number appearing in a replicate label is treated as the
    current experimental round.
    """
    if candidate_id is None or (isinstance(candidate_id, float) and np.isnan(candidate_id)):
        return "UNKNOWN"

    s = str(candidate_id).strip()
    numbers = [int(x) for x in re.findall(r"R\s*(\d+)", s, flags=re.I)]
    if numbers:
        return f"R{max(numbers)}"
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        return "R1"
    return "UNKNOWN"


def recover_round_labels(df: pd.DataFrame, colmap: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Add an auditable Round column without changing the default random few-shot
    benchmark. The recovered labels are available for optional --split_mode round.
    """
    out = df.copy()
    colmap = dict(colmap)

    explicit_col = colmap.get("Round")
    candidate_col = detect_candidate_id_column_by_values(out)

    if explicit_col is not None:
        explicit = out[explicit_col].astype(str).str.strip()
    else:
        explicit = pd.Series(["UNKNOWN"] * len(out), index=out.index)

    if candidate_col is not None:
        inferred = out[candidate_col].apply(infer_round_from_candidate_id)
    else:
        inferred = pd.Series(["UNKNOWN"] * len(out), index=out.index)

    resolved = []
    for exp, inf in zip(explicit, inferred):
        exp_nums = [int(x) for x in re.findall(r"R\s*(\d+)", str(exp), flags=re.I)]
        inf_nums = [int(x) for x in re.findall(r"R\s*(\d+)", str(inf), flags=re.I)]
        nums = exp_nums + inf_nums
        resolved.append(f"R{max(nums)}" if nums else "UNKNOWN")

    out["__Recovered_Round__"] = resolved
    if candidate_col is not None:
        out["__Candidate_ID__"] = out[candidate_col].astype(str)

    colmap["Round"] = "__Recovered_Round__"

    counts = out["__Recovered_Round__"].value_counts(dropna=False).to_dict()
    print(f"[RoundAudit] recovered round counts={counts}")
    if candidate_col is not None:
        print(f"[RoundAudit] candidate ID column={candidate_col!r}")

    return out, colmap


def _publication_formulation_validity_filter(
    df_in: pd.DataFrame,
    colmap: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the invalid/template-row filter used by the final R1-R4 publication pipeline.

    Rules:
      - IL1 must be one of the seven supported ionizable lipids.
      - IL2 may be empty, otherwise must be one of the seven supported ionizable lipids.
      - Helper phospholipid must be DOPE or DSPC.
      - PEG lipid must be C14-PEG, DMG-PEG2000 or PEG-Mannose.
      - Sum of IL1 + IL2 + helper lipid + cholesterol + PEG must be within 80-120 mol%.
    """
    df = df_in.copy().reset_index(drop=True)

    required = ["IL1", "IL2", "IL1_molpct", "IL2_molpct", "Phospholipid",
                "HL_molpct", "CHOL_molpct", "PEG", "PEG_molpct"]
    missing = [k for k in required if not colmap.get(k)]
    if missing:
        raise ValueError(
            "Cannot reproduce the final R1-R4 publication population because "
            f"required formulation columns were not detected: {missing}"
        )

    il1 = df[colmap["IL1"]].apply(normalize_lipid_name)
    il2 = df[colmap["IL2"]].apply(normalize_lipid_name)
    phospholipid = df[colmap["Phospholipid"]].apply(normalize_lipid_name)
    peg = df[colmap["PEG"]].apply(normalize_lipid_name)

    known_il = {"MC3", "ALC-0315", "SM102", "C12-200", "CKK-E12", "DOTAP", "DODAP"}
    known_hl = {"DOPE", "DSPC"}
    known_peg = {"C14-PEG", "DMG-PEG2000", "PEG-Mannose"}

    bad = pd.Series(False, index=df.index)
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)

    checks = [
        (~il1.isin(known_il), "unknown_IL1"),
        (~il2.apply(lambda x: x is None or x in known_il), "unknown_IL2"),
        (~phospholipid.isin(known_hl), "unknown_phospholipid"),
        (~peg.isin(known_peg), "unknown_PEG"),
    ]
    for mask, reason in checks:
        mask = mask.fillna(True)
        bad |= mask
        reasons = reasons.mask(mask, reasons + reason + ";")

    pct_series = []
    for key in ["IL1_molpct", "IL2_molpct", "HL_molpct", "CHOL_molpct", "PEG_molpct"]:
        pct_series.append(pd.to_numeric(df[colmap[key]], errors="coerce"))
    total = pd.concat(pct_series, axis=1).sum(axis=1)
    mol_bad = ((total < 80) | (total > 120)).fillna(True)
    bad |= mol_bad
    reasons = reasons.mask(mol_bad, reasons + "molar_sum_outside_80_120;")

    removed = df.loc[bad].copy()
    if len(removed):
        removed["__invalid_reason__"] = reasons.loc[bad].str.rstrip(";").values

    kept = df.loc[~bad].copy().reset_index(drop=True)
    print(f"[Cleaning] {len(df)} -> {len(kept)} rows; removed {len(removed)} invalid/template rows.")
    return kept, removed


def _publication_qc_filter(
    df_in: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the final R1-R4 QC policy:
      PDI <= 0.5
      particle size 30-300 nm
      missing PDI/size excluded by default
    """
    df = df_in.copy().reset_index(drop=True)

    pdi_col = find_col_by_keywords(df, ["pdi", "polydispers", "多分散"])
    size_col = find_col_by_keywords(
        df,
        ["particle_size", "z-ave", "zave", "diameter", "size", "粒径"],
    )

    if pdi_col is None and size_col is None:
        raise ValueError(
            "No size/PDI columns were detected. Strict publication-population "
            "alignment cannot be verified."
        )

    bad = pd.Series(False, index=df.index)
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)

    if pdi_col is not None:
        pdi = pd.to_numeric(df[pdi_col], errors="coerce")
        pdi_missing = pdi.isna() if cfg.require_complete_qc else pd.Series(False, index=df.index)
        pdi_high = (pdi > cfg.qc_pdi_max).fillna(False)
        bad |= pdi_missing | pdi_high
        reasons = reasons.mask(pdi_missing, reasons + "PDI_missing;")
        reasons = reasons.mask(pdi_high, reasons + f"PDI>{cfg.qc_pdi_max};")

    if size_col is not None:
        size = pd.to_numeric(df[size_col], errors="coerce")
        size_missing = size.isna() if cfg.require_complete_qc else pd.Series(False, index=df.index)
        size_bad = ((size < cfg.qc_size_min) | (size > cfg.qc_size_max)).fillna(False)
        bad |= size_missing | size_bad
        reasons = reasons.mask(size_missing, reasons + "size_missing;")
        reasons = reasons.mask(
            size_bad,
            reasons + f"size_outside_{cfg.qc_size_min}_{cfg.qc_size_max}nm;",
        )

    removed = df.loc[bad].copy()
    if len(removed):
        removed["__QC_removed_reason__"] = reasons.loc[bad].str.rstrip(";").values

    kept = df.loc[~bad].copy().reset_index(drop=True)
    print(f"[QC] {len(df)} -> {len(kept)} rows; removed {len(removed)} QC failures.")
    return kept, removed


def prepare_final_r1r4_inhouse_population(
    df_target_rows: pd.DataFrame,
    cfg: Config,
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Reproduce the same R1-R4 publication population used by the main tree and
    TabPFN/TabFM analyses: target-present rows -> invalid/template removal -> QC.
    """
    n_target = len(df_target_rows)
    colmap = detect_inhouse_columns(df_target_rows, target)

    valid, invalid_removed = _publication_formulation_validity_filter(df_target_rows, colmap)

    if cfg.apply_qc_filter:
        kept, qc_removed = _publication_qc_filter(valid, cfg)
    else:
        kept = valid.reset_index(drop=True)
        qc_removed = valid.iloc[0:0].copy()

    kept, colmap = recover_round_labels(kept, colmap)

    audit = pd.DataFrame([
        {"stage": "target_nonmissing", "n_rows": n_target},
        {"stage": "after_invalid_template_removal", "n_rows": len(valid)},
        {"stage": "after_QC", "n_rows": len(kept)},
    ])

    print(
        "[PopulationAudit] "
        + " -> ".join(f"{row.stage}={row.n_rows}" for row in audit.itertuples(index=False))
    )

    if cfg.enforce_expected_inhouse_n and len(kept) != int(cfg.expected_inhouse_n):
        raise ValueError(
            "Final in-house population does not match the publication dataset. "
            f"Expected {cfg.expected_inhouse_n} rows but obtained {len(kept)}. "
            "Do not use this benchmark result until the preprocessing mismatch is resolved."
        )

    return kept, invalid_removed, qc_removed, audit, colmap

def build_inhouse_features(df: pd.DataFrame, cfg: Config, colmap: Dict[str, Optional[str]], smiles_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    idx = df.index
    parts = []

    # Composition features with same canonical names as public.
    comp = pd.DataFrame(index=idx)
    il1_pct = pd.to_numeric(df[colmap["IL1_molpct"]], errors="coerce").fillna(0.0) if colmap.get("IL1_molpct") else pd.Series(0.0, index=idx)
    il2_pct = pd.to_numeric(df[colmap["IL2_molpct"]], errors="coerce").fillna(0.0) if colmap.get("IL2_molpct") else pd.Series(0.0, index=idx)

    comp["ionizable_molpct"] = il1_pct + il2_pct
    comp["helper_molpct"] = pd.to_numeric(df[colmap["HL_molpct"]], errors="coerce").fillna(0.0) if colmap.get("HL_molpct") else 0.0
    comp["cholesterol_molpct"] = pd.to_numeric(df[colmap["CHOL_molpct"]], errors="coerce").fillna(0.0) if colmap.get("CHOL_molpct") else 0.0
    comp["peg_molpct"] = pd.to_numeric(df[colmap["PEG_molpct"]], errors="coerce").fillna(0.0) if colmap.get("PEG_molpct") else 0.0
    comp["np_ratio"] = 0.0
    if cfg.use_numeric_composition:
        parts.append(comp.astype(np.float32))

    # Weighted IL1/IL2 Morgan.
    fp_source = "none"
    match_count = 0
    if cfg.use_morgan:
        n = len(df)
        fp_mat = np.zeros((n, cfg.morgan_bits), dtype=np.float32)

        denom = (il1_pct + il2_pct).to_numpy(dtype=float)
        f1 = np.where(denom > 0, il1_pct.to_numpy(dtype=float) / denom, 0.5)
        f1 = np.clip(f1, 0, 1)

        il1_vals = df[colmap["IL1"]].tolist() if colmap.get("IL1") else [None] * n
        il2_vals = df[colmap["IL2"]].tolist() if colmap.get("IL2") else [None] * n
        cache = {}

        for i, (a, b, frac) in enumerate(zip(il1_vals, il2_vals, f1)):
            a_norm = normalize_lipid_name(a)
            b_norm = normalize_lipid_name(b)
            key_a = norm_token(a_norm)
            key_b = norm_token(b_norm)

            smi_a = smiles_map.get(key_a, "")
            smi_b = smiles_map.get(key_b, "") or smi_a

            if smi_a and smi_a not in cache:
                cache[smi_a] = morgan_fp(smi_a, cfg.morgan_radius, cfg.morgan_bits)
            if smi_b and smi_b not in cache:
                cache[smi_b] = morgan_fp(smi_b, cfg.morgan_radius, cfg.morgan_bits)

            fp_a = cache.get(smi_a, np.zeros(cfg.morgan_bits, dtype=np.float32))
            fp_b = cache.get(smi_b, np.zeros(cfg.morgan_bits, dtype=np.float32))

            if np.abs(fp_a).sum() > 0 or np.abs(fp_b).sum() > 0:
                match_count += 1
            fp_mat[i, :] = float(frac) * fp_a + (1.0 - float(frac)) * fp_b

        if float(np.abs(fp_mat).sum()) > 0:
            parts.append(pd.DataFrame(fp_mat, index=idx, columns=[f"morgan_{j}" for j in range(cfg.morgan_bits)]))
            fp_source = f"weighted_IL1_IL2_SMILES_Morgan_r{cfg.morgan_radius}_{cfg.morgan_bits}"

    if not parts:
        raise ValueError("No in-house features could be built.")

    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    report = {
        "inhouse_colmap": colmap,
        "inhouse_fingerprint_source": fp_source,
        "morgan_match_rate": float(match_count / max(1, len(df))),
        "n_inhouse_features": int(X.shape[1]),
    }
    return X, report


# =============================================================================
# Meta-learning models: pure PyTorch MAML/FoMAML/MetaSGD
# =============================================================================

class FunctionalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = float(dropout)

    def forward(self, x):
        return self.functional_forward(x, OrderedDict(self.named_parameters()), training=self.training)

    def functional_forward(self, x, params: OrderedDict, training: bool):
        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=training)
        x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=training)
        x = F.linear(x, params["out.weight"], params["out.bias"])
        return x


class MetaRegressor(nn.Module):
    def __init__(self, input_dim: int, cfg: Config, algorithm: str):
        super().__init__()
        self.algorithm = algorithm.lower()
        self.model = FunctionalMLP(input_dim, cfg.hidden_dim, cfg.dropout)
        if self.algorithm == "metasgd":
            self.inner_lrs = nn.ParameterDict()
            # Important:
            # softplus(0.001) is about 0.693, not 0.001. That makes MetaSGD
            # explode immediately on high-dimensional Morgan features.
            # We therefore store the inverse-softplus value so that
            # softplus(raw_lr) starts near cfg.adapt_lr.
            init_lr = max(float(cfg.adapt_lr), 1e-8)
            init_raw = math.log(math.expm1(init_lr))
            for name, p in self.model.named_parameters():
                self.inner_lrs[name.replace(".", "__")] = nn.Parameter(torch.full_like(p, float(init_raw)))

    def init_params(self):
        return OrderedDict((n, p) for n, p in self.model.named_parameters())

    def adapt(self, xs, ys, cfg: Config, create_graph: bool):
        params = self.init_params()
        for _ in range(cfg.adapt_steps):
            loss = F.mse_loss(self.model.functional_forward(xs, params, training=True), ys)
            grads = torch.autograd.grad(
                loss,
                list(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )
            new_params = OrderedDict()
            for (name, p), g in zip(params.items(), grads):
                if not create_graph:
                    g = g.detach()
                if self.algorithm == "metasgd":
                    lr = F.softplus(self.inner_lrs[name.replace(".", "__")])
                    lr = torch.clamp(lr, min=1e-8, max=float(cfg.metasgd_max_inner_lr))
                else:
                    lr = float(cfg.adapt_lr)
                new_params[name] = p - lr * g
            params = new_params
        return params

    def pred(self, x, params: Optional[OrderedDict] = None, training=False):
        if params is None:
            params = self.init_params()
        return self.model.functional_forward(x, params, training=training)


def train_meta(algorithm: str, taskset: PublicTaskSet, input_dim: int, cfg: Config) -> MetaRegressor:
    alg = algorithm.lower()
    if alg not in {"maml", "fomaml", "metasgd"}:
        raise ValueError(algorithm)

    device = torch.device(cfg.device)
    model = MetaRegressor(input_dim, cfg, alg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.meta_lr)
    create_graph = alg != "fomaml"
    n_iter = min(cfg.meta_iters, 120) if cfg.quick else cfg.meta_iters

    model.train()
    for it in range(1, n_iter + 1):
        opt.zero_grad()
        total_loss = 0.0

        for _ in range(cfg.tasks_per_batch):
            xs, ys, xq, yq = taskset.sample()
            xs, ys, xq, yq = xs.to(device), ys.to(device), xq.to(device), yq.to(device)
            adapted = model.adapt(xs, ys, cfg, create_graph=create_graph)
            qpred = model.pred(xq, adapted, training=True)
            total_loss = total_loss + F.mse_loss(qpred, yq)

        total_loss = total_loss / cfg.tasks_per_batch
        if not torch.isfinite(total_loss):
            print(f"[WARN] {algorithm} non-finite loss at iter {it}; skipped this optimizer step.")
            opt.zero_grad(set_to_none=True)
            continue
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if it == 1 or it % 50 == 0 or it == n_iter:
            print(f"[public meta-training] {algorithm} iter {it:04d}/{n_iter} loss={float(total_loss.detach().cpu()):.6f}")

    return model


def train_supervised_ann(X_public: np.ndarray, y_public: np.ndarray, input_dim: int, cfg: Config) -> FunctionalMLP:
    device = torch.device(cfg.device)
    model = FunctionalMLP(input_dim, cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.supervised_lr)
    x = torch.tensor(X_public, dtype=torch.float32, device=device)
    y = torch.tensor(y_public.reshape(-1, 1), dtype=torch.float32, device=device)

    n_epochs = min(cfg.supervised_epochs, 120) if cfg.quick else cfg.supervised_epochs
    model.train()
    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep == 1 or ep % 100 == 0 or ep == n_epochs:
            print(f"[public supervised ANN] epoch {ep:04d}/{n_epochs} loss={float(loss.detach().cpu()):.6f}")
    return model


def predict_zero_shot(model, X: np.ndarray, model_type: str, cfg: Config) -> np.ndarray:
    device = torch.device(cfg.device)
    xt = torch.tensor(X.astype(np.float32), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        if model_type == "supervised_ann":
            pred = model(xt).detach().cpu().numpy().reshape(-1)
        else:
            pred = model.pred(xt, None, training=False).detach().cpu().numpy().reshape(-1)
    pred = np.asarray(pred, dtype=float)
    pred[~np.isfinite(pred)] = np.nan
    return pred


def predict_fewshot(model: MetaRegressor, X_support: np.ndarray, y_support: np.ndarray, X_query: np.ndarray, cfg: Config) -> np.ndarray:
    device = torch.device(cfg.device)
    xs = torch.tensor(X_support.astype(np.float32), dtype=torch.float32, device=device)
    ys = torch.tensor(y_support.reshape(-1, 1).astype(np.float32), dtype=torch.float32, device=device)
    xq = torch.tensor(X_query.astype(np.float32), dtype=torch.float32, device=device)
    model.eval()
    adapted = model.adapt(xs, ys, cfg, create_graph=False)
    with torch.no_grad():
        pred = model.pred(xq, adapted, training=False).detach().cpu().numpy().reshape(-1)
        pred = np.asarray(pred, dtype=float)
        pred[~np.isfinite(pred)] = np.nan
        return pred


def finetune_supervised_ann(base_model: FunctionalMLP, X_support: np.ndarray, y_support: np.ndarray, X_query: np.ndarray, cfg: Config) -> np.ndarray:
    # A paper-style supervised transfer baseline: initialize from public ANN, fine-tune on support.
    import copy
    device = torch.device(cfg.device)
    model = copy.deepcopy(base_model).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adapt_lr)

    xs = torch.tensor(X_support.astype(np.float32), dtype=torch.float32, device=device)
    ys = torch.tensor(y_support.reshape(-1, 1).astype(np.float32), dtype=torch.float32, device=device)
    xq = torch.tensor(X_query.astype(np.float32), dtype=torch.float32, device=device)

    for _ in range(cfg.adapt_steps * 20):
        opt.zero_grad()
        loss = F.mse_loss(model(xs), ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(xq).detach().cpu().numpy().reshape(-1)
        pred = np.asarray(pred, dtype=float)
        pred[~np.isfinite(pred)] = np.nan
        return pred


# =============================================================================
# In-house target splits
# =============================================================================

def parse_round_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def make_target_splits(df: pd.DataFrame, colmap: Dict[str, Optional[str]], cfg: Config) -> List[Tuple[int, str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    n = len(df)
    all_idx = np.arange(n)
    splits = []

    round_col = colmap.get("Round")
    mode = cfg.split_mode

    if mode == "auto":
        mode = "round" if round_col and df[round_col].nunique(dropna=True) >= 2 else "random"

    if mode == "round" and round_col:
        rounds = df[round_col].astype(str)
        if cfg.support_rounds and cfg.query_rounds:
            support_rounds = set(parse_round_list(cfg.support_rounds))
            query_rounds = set(parse_round_list(cfg.query_rounds))
            support_idx = np.where(rounds.isin(support_rounds).values)[0]
            query_idx = np.where(rounds.isin(query_rounds).values)[0]
            if len(support_idx) < 3 or len(query_idx) < 2:
                raise ValueError("Support/query round split has too few rows. Check --support_rounds and --query_rounds.")
            splits.append((0, "round_user_specified", support_idx, query_idx))
            return splits

        # Default: latest round as query/test, all earlier rounds as support.
        # If Round values are numeric-like, use numeric order; otherwise lexical.
        rr = pd.to_numeric(df[round_col], errors="coerce")
        if rr.notna().sum() > 0:
            latest = rr.max()
            query_idx = np.where((rr == latest).fillna(False).values)[0]
            support_idx = np.where((rr < latest).fillna(False).values)[0]
            split_name = f"round_auto_latest_{latest}_as_query"
        else:
            vals = sorted(rounds.dropna().unique().tolist())
            latest = vals[-1]
            query_idx = np.where(rounds.eq(latest).values)[0]
            support_idx = np.where(~rounds.eq(latest).values)[0]
            split_name = f"round_auto_latest_{latest}_as_query"

        if len(support_idx) >= 3 and len(query_idx) >= 2:
            splits.append((0, split_name, support_idx, query_idx))
            return splits
        print("[WARN] Round split too small. Falling back to repeated random splits.")

    # Random repeated support/query.
    n_reps = min(cfg.n_repeats, 3) if cfg.quick else cfg.n_repeats
    n_support = max(3, int(round(n * cfg.random_support_fraction)))
    n_support = min(n_support, n - 2)
    for rep in range(n_reps):
        perm = rng.permutation(all_idx)
        support_idx = perm[:n_support]
        query_idx = perm[n_support:]
        splits.append((rep, "random_repeated", support_idx, query_idx))
    return splits


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="MetaLNP-only public meta-training to in-house target test.")
    parser.add_argument("--inhouse_xlsx", default=str(DEFAULT_INHOUSE_XLSX))
    parser.add_argument("--public_train_csv", default=str(DEFAULT_PUBLIC_TRAIN))
    parser.add_argument("--public_val_csv", default=str(DEFAULT_PUBLIC_VAL))
    parser.add_argument("--public_holdout_csv", default=str(DEFAULT_PUBLIC_HOLDOUT))
    parser.add_argument("--public_features_csv", default="", help="Optional prepared public feature matrix, e.g. features_lance_public_long.csv")
    parser.add_argument("--public_manifest_csv", default="", help="Optional prepared public support/query manifest.")
    parser.add_argument("--output_dir", default="")

    parser.add_argument("--target", default="Normalized for DC2.4")
    parser.add_argument("--inhouse_main_sheet", default="Round1&2&3&4")
    parser.add_argument("--inhouse_smiles_sheet", default="SMILES NAME")

    parser.add_argument("--public_target_col", default="")
    parser.add_argument("--public_task_col", default="")
    parser.add_argument("--public_smiles_col", default="")

    parser.add_argument("--models", default="maml,fomaml,metasgd,supervised_ann")
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--support_size_public", type=int, default=10)
    parser.add_argument("--query_size_public", type=int, default=10)
    parser.add_argument("--tasks_per_batch", type=int, default=8)
    parser.add_argument("--meta_iters", type=int, default=800)

    parser.add_argument("--split_mode", choices=["auto", "round", "random"], default="random")
    parser.add_argument("--support_rounds", default="", help='Example: --support_rounds "1,2"')
    parser.add_argument("--query_rounds", default="", help='Example: --query_rounds "3"')
    parser.add_argument("--random_support_fraction", type=float, default=0.80)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--adapt_steps", type=int, default=3)
    parser.add_argument("--adapt_lr", type=float, default=1e-3)
    parser.add_argument("--meta_lr", type=float, default=3e-4)
    parser.add_argument("--metasgd_max_inner_lr", type=float, default=0.02,
                        help="Upper bound for MetaSGD learned inner-loop step size; prevents NaN explosion.")
    parser.add_argument("--supervised_lr", type=float, default=3e-4)
    parser.add_argument("--supervised_epochs", type=int, default=800)

    parser.add_argument("--morgan_radius", type=int, default=4)
    parser.add_argument("--morgan_bits", type=int, default=2048)
    parser.add_argument("--no_morgan", action="store_true")
    parser.add_argument("--no_numeric_composition", action="store_true")
    parser.add_argument("--no_public_existing_fp", action="store_true")

    parser.add_argument("--no_qc_filter", action="store_true")
    parser.add_argument("--qc_pdi_max", type=float, default=0.5)
    parser.add_argument("--qc_size_min", type=float, default=30.0)
    parser.add_argument("--qc_size_max", type=float, default=300.0)
    parser.add_argument("--allow_missing_qc", action="store_true",
                        help="Keep rows with missing PDI/size. Publication default excludes them.")
    parser.add_argument("--expected_inhouse_n", type=int, default=104)
    parser.add_argument("--allow_unexpected_inhouse_n", action="store_true",
                        help="Disable the 104-row publication-population safety check.")

    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = Config(
        inhouse_xlsx=args.inhouse_xlsx,
        public_train_csv=args.public_train_csv,
        public_val_csv=args.public_val_csv,
        public_holdout_csv=args.public_holdout_csv,
        public_features_csv=args.public_features_csv,
        public_manifest_csv=args.public_manifest_csv,
        output_dir=args.output_dir,
        target=args.target,
        inhouse_main_sheet=args.inhouse_main_sheet,
        inhouse_smiles_sheet=args.inhouse_smiles_sheet,
        public_target_col=args.public_target_col,
        public_task_col=args.public_task_col,
        public_smiles_col=args.public_smiles_col,
        models=args.models,
        n_repeats=args.n_repeats,
        seed=args.seed,
        quick=args.quick,
        support_size_public=args.support_size_public,
        query_size_public=args.query_size_public,
        tasks_per_batch=args.tasks_per_batch,
        meta_iters=args.meta_iters,
        split_mode=args.split_mode,
        support_rounds=args.support_rounds,
        query_rounds=args.query_rounds,
        random_support_fraction=args.random_support_fraction,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        adapt_steps=args.adapt_steps,
        adapt_lr=args.adapt_lr,
        meta_lr=args.meta_lr,
        metasgd_max_inner_lr=args.metasgd_max_inner_lr,
        supervised_lr=args.supervised_lr,
        supervised_epochs=args.supervised_epochs,
        morgan_radius=args.morgan_radius,
        morgan_bits=args.morgan_bits,
        use_morgan=not args.no_morgan,
        use_numeric_composition=not args.no_numeric_composition,
        use_public_existing_fp=not args.no_public_existing_fp,
        apply_qc_filter=not args.no_qc_filter,
        qc_pdi_max=args.qc_pdi_max,
        qc_size_min=args.qc_size_min,
        qc_size_max=args.qc_size_max,
        require_complete_qc=not args.allow_missing_qc,
        expected_inhouse_n=args.expected_inhouse_n,
        enforce_expected_inhouse_n=not args.allow_unexpected_inhouse_n,
        device=args.device,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(cfg.output_dir) if cfg.output_dir else DEFAULT_OUTPUT_ROOT / f"run_{timestamp}"
    outdir = mkdir(outdir)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print("=" * 100)
    print("[MetaLNP-style public meta-training -> final 104-row R1-R4 in-house benchmark | PUBLICATION]")
    print(f"Public train : {cfg.public_train_csv}")
    print(f"In-house     : {cfg.inhouse_xlsx}")
    print(f"Target       : {cfg.target}")
    print(f"Output       : {outdir}")
    print(f"RDKit        : {HAS_RDKIT}")
    print(f"Models       : {cfg.models}")
    print(f"Quick mode   : {cfg.quick}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Load public data
    # -------------------------------------------------------------------------
    public_df = safe_read_csv(Path(cfg.public_train_csv))
    public_df.columns = [clean_col(c) for c in public_df.columns]

    pub_target = detect_public_target(public_df, cfg.public_target_col)
    pub_task_col = detect_public_task_col(public_df, cfg.public_task_col)
    pub_smiles_col = detect_smiles_col(public_df, cfg.public_smiles_col)

    print(f"[Public detected] target={pub_target}")
    print(f"[Public detected] task_col={pub_task_col}")
    print(f"[Public detected] smiles_col={pub_smiles_col}")

    public_df = public_df[pd.to_numeric(public_df[pub_target], errors="coerce").notna()].copy().reset_index(drop=True)
    y_public_raw = pd.to_numeric(public_df[pub_target], errors="coerce").to_numpy(dtype=float)

    # Meta-training target is task-wise percentile. This avoids public/in-house raw scale mismatch.
    temp_task_ids, temp_task_report = make_public_task_ids(public_df, y_public_raw, pub_task_col, cfg)
    keep_public = temp_task_ids != ""
    public_df = public_df.loc[keep_public].reset_index(drop=True)
    y_public_raw = y_public_raw[keep_public]
    public_task_ids = temp_task_ids[keep_public]

    if len(public_df) == 0 or len(pd.unique(public_task_ids)) < 3:
        raise RuntimeError(
            "No enough public MetaLNP tasks remained after filtering. "
            "Check --public_task_col, --support_size_public, and --query_size_public."
        )

    y_public_percentile = np.zeros_like(y_public_raw, dtype=np.float32)
    for tid in pd.unique(public_task_ids):
        idx = np.where(public_task_ids == tid)[0]
        y_public_percentile[idx] = to_percentile(y_public_raw[idx])

    X_public_df, public_feature_report = build_public_features(public_df, cfg, pub_target, pub_task_col, pub_smiles_col)
    # Rebuild task report after filtering, for exact output
    public_task_report = dict(temp_task_report)
    public_task_report["n_public_rows_final"] = int(len(public_df))
    public_task_report["n_tasks_final"] = int(len(pd.unique(public_task_ids)))

    # -------------------------------------------------------------------------
    # Load in-house target data
    # -------------------------------------------------------------------------
    inhouse_path = Path(cfg.inhouse_xlsx)
    inhouse_df, main_sheet, sheet_names = read_excel_target_sheet(inhouse_path, cfg.target, cfg.inhouse_main_sheet)
    inhouse_df.columns = [clean_col(c) for c in inhouse_df.columns]

    inhouse_df = inhouse_df[
        pd.to_numeric(inhouse_df[cfg.target], errors="coerce").notna()
    ].copy().reset_index(drop=True)

    (
        inhouse_df,
        invalid_removed,
        qc_removed,
        population_audit,
        inhouse_colmap,
    ) = prepare_final_r1r4_inhouse_population(
        inhouse_df,
        cfg,
        cfg.target,
    )

    y_inhouse_raw = pd.to_numeric(inhouse_df[cfg.target], errors="coerce").to_numpy(dtype=float)
    y_inhouse_percentile = to_percentile(y_inhouse_raw)

    smiles_map, smiles_report = load_inhouse_smiles_reference(inhouse_path, cfg.inhouse_smiles_sheet)
    X_inhouse_df, inhouse_feature_report = build_inhouse_features(inhouse_df, cfg, inhouse_colmap, smiles_map)

    print(f"[In-house] main sheet={main_sheet}")
    print(f"[In-house] all sheets={sheet_names}")
    print(f"[In-house] N={len(inhouse_df)} | y mean={np.mean(y_inhouse_raw):.4f} | y std={np.std(y_inhouse_raw):.4f}")
    print(f"[In-house] SMILES loaded={len(smiles_map)} | Morgan match rate={inhouse_feature_report['morgan_match_rate']:.3f}")
    print(f"[In-house] detected columns={json.dumps(inhouse_colmap, ensure_ascii=False)}")

    # Align public and in-house features.
    all_cols = sorted(set(X_public_df.columns) | set(X_inhouse_df.columns))
    X_public_df = X_public_df.reindex(columns=all_cols, fill_value=0.0)
    X_inhouse_df = X_inhouse_df.reindex(columns=all_cols, fill_value=0.0)

    scaler = StandardScaler()
    X_public = scaler.fit_transform(X_public_df.to_numpy(dtype=np.float32)).astype(np.float32)
    X_inhouse = scaler.transform(X_inhouse_df.to_numpy(dtype=np.float32)).astype(np.float32)

    taskset = PublicTaskSet(X_public, y_public_percentile, public_task_ids, cfg)

    print(f"[Features] aligned feature count={len(all_cols)}")
    print(f"[Public tasks] usable tasks={len(taskset.keys)}")

    # Save mapping report early.
    mapping_report = {
        "config": asdict(cfg),
        "public_target": pub_target,
        "public_task_col": pub_task_col,
        "public_smiles_col": pub_smiles_col,
        "public_task_report": public_task_report,
        "public_feature_report": public_feature_report,
        "inhouse_main_sheet": main_sheet,
        "inhouse_sheet_names": sheet_names,
        "inhouse_colmap": inhouse_colmap,
        "inhouse_smiles_report": smiles_report,
        "inhouse_feature_report": inhouse_feature_report,
        "aligned_feature_count": len(all_cols),
        "aligned_features_preview": all_cols[:200],
        "important_warning": (
            "This script does not retrain the user's own in-house RF/GB/HGB models. "
            "It only reports public MetaLNP-style models on the in-house target."
        ),
    }
    (outdir / "metalnp_feature_mapping_report.json").write_text(
        json.dumps(mapping_report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    population_audit.to_csv(
        outdir / "inhouse_population_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if len(invalid_removed):
        invalid_removed.to_csv(
            outdir / "invalid_template_removed_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if len(qc_removed):
        qc_removed.to_csv(
            outdir / "qc_removed_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    combined_removed = []
    if len(invalid_removed):
        combined_removed.append(invalid_removed.assign(removal_stage="invalid_or_template"))
    if len(qc_removed):
        combined_removed.append(qc_removed.assign(removal_stage="QC"))
    if combined_removed:
        pd.concat(combined_removed, ignore_index=True).to_csv(
            outdir / "inhouse_removed_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # Train public models
    # -------------------------------------------------------------------------
    wanted = [m.strip().lower() for m in cfg.models.split(",") if m.strip()]
    trained = {}

    for m in wanted:
        if m in {"maml", "fomaml", "metasgd"}:
            print("\n" + "-" * 88)
            print(f"[Train public meta model] {m}")
            trained[m] = train_meta(m, taskset, input_dim=X_public.shape[1], cfg=cfg)
            torch.save(trained[m].state_dict(), outdir / f"public_meta_{m}.pt")
        elif m == "supervised_ann":
            print("\n" + "-" * 88)
            print("[Train public supervised ANN baseline]")
            trained[m] = train_supervised_ann(X_public, y_public_percentile, X_public.shape[1], cfg)
            torch.save(trained[m].state_dict(), outdir / "public_supervised_ann.pt")
        else:
            print(f"[WARN] Unknown model name skipped: {m}")

    # -------------------------------------------------------------------------
    # Evaluation A: zero-shot on all in-house rows
    # -------------------------------------------------------------------------
    rows = []
    pred_rows = []

    for name, model in trained.items():
        pred_zero = predict_zero_shot(model, X_inhouse, name, cfg)
        md = metrics(y_inhouse_percentile, pred_zero)
        md.update({
            "model": name,
            "evaluation": "A_zero_shot_all_inhouse",
            "target_scale": "inhouse_percentile",
            "repeat": 0,
            "split_name": "all_rows",
            "n_support": 0,
            "n_query": int(len(y_inhouse_percentile)),
        })
        rows.append(md)
        for i, (yt, yp) in enumerate(zip(y_inhouse_percentile, pred_zero)):
            pred_rows.append({
                "model": name,
                "evaluation": "A_zero_shot_all_inhouse",
                "target_scale": "inhouse_percentile",
                "repeat": 0,
                "split_name": "all_rows",
                "row_index": int(i),
                "y_true": float(yt),
                "y_pred": float(yp),
            })

    # -------------------------------------------------------------------------
    # Evaluation B: few-shot target adaptation
    # -------------------------------------------------------------------------
    splits = make_target_splits(inhouse_df, inhouse_colmap, cfg)
    print(f"[Target splits] {len(splits)} split(s): {[s[1] for s in splits[:5]]}")

    for rep, split_name, support_idx, query_idx in splits:
        Xs = X_inhouse[support_idx]
        Xq = X_inhouse[query_idx]

        # B1: adapt to raw in-house target. This is the one closest to your original R2 scale.
        ys_raw = y_inhouse_raw[support_idx]
        yq_raw = y_inhouse_raw[query_idx]

        # B2: adapt to percentile in-house target. This is scale-normalized and more MetaLNP-like.
        ys_pct = y_inhouse_percentile[support_idx]
        yq_pct = y_inhouse_percentile[query_idx]

        for name, model in trained.items():
            if name == "supervised_ann":
                pred_raw = finetune_supervised_ann(model, Xs, ys_raw, Xq, cfg)
                pred_pct = finetune_supervised_ann(model, Xs, ys_pct, Xq, cfg)
            else:
                pred_raw = predict_fewshot(model, Xs, ys_raw, Xq, cfg)
                pred_pct = predict_fewshot(model, Xs, ys_pct, Xq, cfg)

            for scale_name, y_true, y_pred in [
                ("raw_inhouse_target", yq_raw, pred_raw),
                ("inhouse_percentile", yq_pct, pred_pct),
            ]:
                md = metrics(y_true, y_pred)
                md.update({
                    "model": name,
                    "evaluation": "B_fewshot_adapt_support_to_query",
                    "target_scale": scale_name,
                    "repeat": int(rep),
                    "split_name": split_name,
                    "n_support": int(len(support_idx)),
                    "n_query": int(len(query_idx)),
                })
                rows.append(md)

                for ii, global_i in enumerate(query_idx):
                    pred_rows.append({
                        "model": name,
                        "evaluation": "B_fewshot_adapt_support_to_query",
                        "target_scale": scale_name,
                        "repeat": int(rep),
                        "split_name": split_name,
                        "row_index": int(global_i),
                        "y_true": float(y_true[ii]),
                        "y_pred": float(y_pred[ii]),
                    })

    metrics_df = pd.DataFrame(rows)
    preds_df = pd.DataFrame(pred_rows)

    summary = (
        metrics_df
        .groupby(["model", "evaluation", "target_scale", "split_name"], as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            n_support_mean=("n_support", "mean"),
            n_query_mean=("n_query", "mean"),
            n_finite_mean=("n_finite", "mean"),
            valid_prediction_rate=("valid_prediction", "mean"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            Pearson_mean=("Pearson", "mean"),
            Pearson_std=("Pearson", "std"),
            Spearman_mean=("Spearman", "mean"),
            Spearman_std=("Spearman", "std"),
            Top20_precision_mean=("Top20_precision", "mean"),
            Top20_precision_std=("Top20_precision", "std"),
        )
        .sort_values(["evaluation", "target_scale", "R2_mean", "Spearman_mean"], ascending=[True, True, False, False])
        .reset_index(drop=True)
    )

    metrics_path = outdir / "metalnp_only_repeat_metrics.csv"
    preds_path = outdir / "metalnp_only_predictions.csv"
    summary_path = outdir / "metalnp_only_model_summary.csv"
    xlsx_path = outdir / "MetaLNP_ONLY_public_to_inhouse_results.xlsx"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="model_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="repeat_metrics", index=False)
        preds_df.to_excel(writer, sheet_name="predictions", index=False)
        population_audit.to_excel(writer, sheet_name="population_audit", index=False)
        inhouse_df.to_excel(writer, sheet_name="inhouse_104_population", index=False)
        invalid_removed.to_excel(writer, sheet_name="invalid_removed", index=False)
        qc_removed.to_excel(writer, sheet_name="qc_removed", index=False)
        pd.DataFrame({"feature": all_cols}).to_excel(writer, sheet_name="aligned_features", index=False)
        pd.DataFrame([asdict(cfg)]).to_excel(writer, sheet_name="config", index=False)

    print("\n" + "=" * 100)
    print("[DONE] MetaLNP-only benchmark finished.")
    print("Main summary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"- {summary_path}")
    print(f"- {metrics_path}")
    print(f"- {preds_path}")
    print(f"- {xlsx_path}")
    print(f"- {outdir / 'inhouse_population_audit.csv'}")
    print(f"- {outdir / 'inhouse_removed_rows.csv'}")
    print(f"- {outdir / 'metalnp_feature_mapping_report.json'}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    raise SystemExit(main())