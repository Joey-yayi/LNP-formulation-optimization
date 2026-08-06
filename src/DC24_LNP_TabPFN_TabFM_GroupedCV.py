# -*- coding: utf-8 -*-
"""
DC24_LNP_TabPFN_TabFM_GroupedCV_Publication_FIXED_v2_0_4.py

Publication-oriented, leakage-safe benchmark of TabPFN and Google TabFM for
normalized DC2.4 mRNA-LNP transfection efficiency.

Design principles
-----------------
* Exact duplicate formulations are kept within the same train/test partition.
* Five-fold formulation-grouped cross-validation is repeated three times by
  default; every sample receives one out-of-fold prediction per repeat.
* Auxiliary molecular descriptors and Morgan fingerprint bits are selected by
  mRMR using only the current outer-training fold.
* HaCaT measurements are never used as the response or as model predictors.
* The normalized DC2.4 target is required by default; the script does not
  silently substitute another target column.
* Split assignments, fold-level predictions, repeat-level metrics, bootstrap
  confidence intervals, pairwise model comparisons, software versions, hashes,
  and a complete run manifest are exported for audit and reproduction.

Recommended repository layout
-----------------------------
    data/processed/lnp_dc24_hacat_modeling_dataset.xlsx
    src/foundation_models/DC24_LNP_TabPFN_TabFM_GroupedCV_Publication.py
    scripts/run_DC24_foundation_benchmark_windows.py
    results/                         # generated; normally ignored by Git

Typical formal run
------------------
    python DC24_LNP_TabPFN_TabFM_GroupedCV_Publication.py \
      --data-path data/processed/lnp_dc24_hacat_modeling_dataset.xlsx \
      --outer-folds 5 --outer-repeats 3 --bootstrap-iterations 2000

Quick validation without fitting remote/foundation models
---------------------------------------------------------
    python DC24_LNP_TabPFN_TabFM_GroupedCV_Publication.py \
      --validate-only --data-path path/to/workbook.xlsx

Privacy and licensing
---------------------
The TabPFN client uploads training features and target values to Prior Labs.
Use --tabpfn-backend local or --no-tabpfn when remote transfer is unsuitable.
TabFM pretrained weights are loaded locally and are subject to the separate
TabFM pretrained-weight license. Do not commit API tokens or model weights.
"""


from __future__ import annotations

import argparse
import hashlib
import fnmatch
import json
import logging
import os
import re
import sys
import time
import warnings
import importlib.util
import importlib.metadata as importlib_metadata
import subprocess
import platform
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Automatic interpreter bootstrap
# -----------------------------------------------------------------------------
def _candidate_tabfm_interpreters() -> List[Path]:
    """Return likely TabFM virtual-environment interpreters without user-specific paths."""
    candidates: List[Path] = []

    def add_root(root: Path) -> None:
        candidates.extend([
            root / ".venv_tabfm" / "Scripts" / "python.exe",
            root / ".venv_tabfm" / "Scripts" / "python",
            root / ".venv_tabfm" / "bin" / "python",
        ])

    starts: List[Path] = [Path.cwd(), Path.home()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    for start_path in starts:
        for root in [start_path, *list(start_path.parents)[:6]]:
            add_root(root)

    # A downloaded script may be in Downloads while the environment remains in
    # a project under Desktop/OneDrive. Search only a shallow, bounded set of
    # common roots so startup remains predictable.
    home = Path.home()
    search_roots = [
        home / "Desktop", home / "桌面", home / "Documents",
        home / "OneDrive", home / "OneDrive - nny",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            for env_dir in root.glob("**/.venv_tabfm"):
                try:
                    relative_depth = len(env_dir.relative_to(root).parts)
                except Exception:
                    relative_depth = 99
                if relative_depth <= 5:
                    add_root(env_dir.parent)
        except (OSError, PermissionError):
            continue

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique

def _clean_child_environment(python_executable: Path) -> Dict[str, str]:
    """
    Build a clean environment for the TabFM interpreter.

    PyCharm can inject PYTHONPATH entries from another interpreter. In the
    user's previous run, Python 3.12 from .venv_tabfm imported NumPy from
    .venv314_clean, which caused the C-extension failure. Clearing the Python
    path variables and using isolated mode prevents that cross-environment mix.
    """
    env = os.environ.copy()

    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "__PYVENV_LAUNCHER__",
    ):
        env.pop(key, None)

    venv_root = python_executable.parent.parent
    env["VIRTUAL_ENV"] = str(venv_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    env["DC24_TABFM_AUTO_REEXEC"] = "1"
    # Preserve the interpreter that launched the script. It often contains
    # RDKit even when the isolated TabFM environment does not. The child uses
    # it only as a subprocess, never by mixing site-packages.
    env["DC24_ORIGINAL_PYTHON"] = os.environ.get(
        "DC24_ORIGINAL_PYTHON", sys.executable
    )

    # Put the correct Scripts directory first for any subprocesses spawned later.
    current_path = env.get("PATH", "")
    env["PATH"] = str(python_executable.parent) + os.pathsep + current_path
    return env


def _interpreter_health_check(python_executable: Path) -> Tuple[bool, str]:
    """Confirm that TabFM and the scientific Python stack import cleanly."""
    check_code = (
        "import sys;"
        "import numpy, pandas, sklearn, tabfm;"
        "print(sys.executable);"
        "print(numpy.__file__);"
        "print(tabfm.__file__)"
    )
    try:
        result = subprocess.run(
            [str(python_executable), "-I", "-c", check_code],
            env=_clean_child_environment(python_executable),
            cwd=str(Path.home()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
        details = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, details.strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _auto_relaunch_with_tabfm_environment() -> None:
    """
    Relaunch with .venv_tabfm when the current interpreter lacks TabFM.

    The child interpreter is started with Python's -I isolated mode and a
    sanitized environment. This prevents PyCharm's old PYTHONPATH from mixing
    .venv314_clean packages into the Python 3.12 TabFM environment.
    """
    if "--no-tabfm" in sys.argv:
        return
    if os.environ.get("DC24_TABFM_AUTO_REEXEC", "") == "1":
        return
    if importlib.util.find_spec("tabfm") is not None:
        return

    current = os.path.normcase(os.path.abspath(sys.executable))
    health_failures: List[str] = []

    for candidate in _candidate_tabfm_interpreters():
        if not candidate.is_file():
            continue

        candidate_norm = os.path.normcase(os.path.abspath(str(candidate)))
        if candidate_norm == current:
            continue

        healthy, details = _interpreter_health_check(candidate)
        if not healthy:
            health_failures.append(f"{candidate}\n{details}")
            continue

        script_path = Path(__file__).resolve()
        child_env = _clean_child_environment(candidate)

        print(
            "\n[Environment] TabFM is not installed in the current interpreter:\n"
            f"  {sys.executable}\n"
            "[Environment] Relaunching with a clean isolated TabFM environment:\n"
            f"  {candidate}\n"
            "[Environment] Old PYTHONPATH/PYTHONHOME values are removed to prevent "
            "cross-environment NumPy imports.\n",
            flush=True,
        )

        completed = subprocess.run(
            [str(candidate), "-I", str(script_path), *sys.argv[1:]],
            env=child_env,
            cwd=str(script_path.parent),
            check=False,
        )
        raise SystemExit(completed.returncode)

    searched = "\n".join(f"  - {path}" for path in _candidate_tabfm_interpreters())
    failure_text = "\n\n".join(health_failures) if health_failures else "(none found)"
    raise RuntimeError(
        "TabFM was requested but no healthy .venv_tabfm interpreter was found.\n"
        f"Current interpreter: {sys.executable}\n"
        "Searched:\n" + searched + "\n\n"
        "Interpreter health-check details:\n" + failure_text + "\n\n"
        "Run the script from .venv_tabfm, repair that environment, or use --no-tabfm."
    )


_auto_relaunch_with_tabfm_environment()

# PyCharm/Windows may expose a GBK console even though paths and tracebacks
# contain characters outside GBK.  Configure replacement rather than allowing
# logging itself to terminate an otherwise recoverable run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold


try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

warnings.filterwarnings("once")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))

SCRIPT_VERSION = "2.0.5-publication-reproducible-safe-tree-merge"
DEFAULT_FILE = "lnp_dc24_hacat_modeling_dataset.xlsx"

EXTERNAL_RDKIT_INFO: Dict[str, str] = {}
_RUN_LOG_HANDLE: Optional[Any] = None

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
    """Locate the training workbook deterministically and duplicate-safely.

    Resolution order
    ----------------
    1. ``--data-path`` or ``DC24_LNP_DATA``.
    2. Exact filename in the script/repository neighbourhood.
    3. Semantic filename matches in bounded Desktop/Documents/OneDrive searches.

    When several semantic matches are found, byte-identical copies are treated
    as harmless duplicates.  The function chooses one canonical copy using a
    deterministic priority rule and prints every duplicate path.  If the files
    have different SHA-256 hashes, execution stops rather than silently using
    an ambiguous dataset.
    """
    expected_name = DEFAULT_FILE
    expected_folded = expected_name.casefold()
    required_tokens = ("dc24", "hacat", "modeling", "dataset")

    def clean(value: Any) -> Path:
        return Path(
            os.path.expandvars(
                os.path.expanduser(str(value).strip().strip('"').strip("'"))
            )
        )

    def is_excel(candidate: Path) -> bool:
        return (
            candidate.is_file()
            and not candidate.name.startswith("~$")
            and candidate.suffix.casefold() in {".xlsx", ".xlsm", ".xls"}
        )

    def semantic_match(candidate: Path) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "", candidate.name.casefold())
        return all(token in normalized for token in required_tokens)

    def file_sha256(candidate: Path) -> str:
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def canonical_priority(candidate: Path, script_dir: Path) -> Tuple[int, int, int, str]:
        """Lower values are preferred; the rule is stable across runs."""
        parts_folded = [part.casefold() for part in candidate.parts]

        # Repository convention is the strongest preference.
        in_data_processed = any(
            parts_folded[index:index + 2] == ["data", "processed"]
            for index in range(max(0, len(parts_folded) - 1))
        )

        # A copied workbook in an old publication/output folder should not beat
        # the canonical project-level or data/processed copy.
        archival_names = {
            "8.02 publish", "publish", "publication", "outputs", "lnp_outputs",
            "downloads", "download", "archive", "backup", "old", "temp", "tmp",
        }
        archival_penalty = int(any(part in archival_names for part in parts_folded[:-1]))

        beside_script = int(candidate.parent.resolve() != script_dir.resolve())
        path_depth = len(candidate.parts)
        return (
            0 if in_data_processed else 1,
            archival_penalty,
            beside_script + path_depth,
            os.path.normcase(str(candidate)),
        )

    def resolve_multiple(matches: Sequence[Path], script_dir: Path) -> Optional[Path]:
        unique = list(dict.fromkeys(path.resolve() for path in matches if is_excel(path)))
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]

        hash_to_paths: Dict[str, List[Path]] = {}
        hash_errors: List[str] = []
        for candidate in unique:
            try:
                digest = file_sha256(candidate)
            except OSError as exc:
                hash_errors.append(f"  - {candidate}: {type(exc).__name__}: {exc}")
                continue
            hash_to_paths.setdefault(digest, []).append(candidate)

        if hash_errors:
            raise FileNotFoundError(
                "Multiple workbook candidates were found, but one or more could "
                "not be hashed safely:\n" + "\n".join(hash_errors)
            )

        if len(hash_to_paths) == 1:
            chosen = min(unique, key=lambda path: canonical_priority(path, script_dir))
            print("[DataPath] Multiple byte-identical workbook copies were found.")
            print(f"[DataPath] Canonical copy selected: {chosen}")
            for duplicate in unique:
                if duplicate != chosen:
                    print(f"[DataPath] Identical duplicate ignored: {duplicate}")
            return chosen

        details: List[str] = []
        for digest, paths in sorted(hash_to_paths.items()):
            details.append(f"SHA256 {digest}")
            details.extend(f"  - {path}" for path in paths)
        raise FileNotFoundError(
            "Multiple possible DC24/HaCaT workbooks with DIFFERENT contents were "
            "found. For publication safety, the script will not guess which one "
            "is correct. Delete/rename the unintended copy or pass --data-path:\n"
            + "\n".join(details)
        )

    explicit_values = [user_path, os.environ.get("DC24_LNP_DATA", "")]
    for explicit in explicit_values:
        if not explicit:
            continue
        supplied = clean(explicit)
        candidates = [supplied] if supplied.is_absolute() else [
            Path.cwd() / supplied,
            Path(__file__).resolve().parent / supplied,
        ]
        for candidate in candidates:
            if is_excel(candidate):
                resolved = candidate.resolve()
                print(f"[DataPath] Using explicit workbook: {resolved}")
                return str(resolved)
        raise FileNotFoundError(
            "An explicit data path was supplied but no workbook exists at:\n"
            + "\n".join(f"  - {candidate}" for candidate in candidates)
        )

    script_dir = Path(__file__).resolve().parent
    home = Path.home()
    nearby_dirs: List[Path] = [Path.cwd(), script_dir]
    for parent in [script_dir, *list(script_dir.parents)[:6]]:
        nearby_dirs.extend([
            parent,
            parent / "data" / "processed",
            parent / "data",
            parent / "datasets",
        ])

    unique_dirs: List[Path] = []
    seen_dirs: set[str] = set()
    for directory in nearby_dirs:
        key = os.path.normcase(os.path.abspath(str(directory)))
        if key not in seen_dirs:
            seen_dirs.add(key)
            unique_dirs.append(directory)

    semantic_hits: List[Path] = []
    nearby_workbooks: List[Path] = []
    exact_hits: List[Path] = []
    for directory in unique_dirs:
        if not directory.is_dir():
            continue
        try:
            workbooks = [path for path in directory.iterdir() if is_excel(path)]
        except (OSError, PermissionError):
            continue
        nearby_workbooks.extend(workbooks)
        exact_hits.extend(path.resolve() for path in workbooks if path.name.casefold() == expected_folded)
        semantic_hits.extend(path.resolve() for path in workbooks if semantic_match(path))

    resolved_exact = resolve_multiple(exact_hits, script_dir)
    if resolved_exact is not None:
        print(f"[DataPath] Found exact workbook: {resolved_exact}")
        return str(resolved_exact)

    # Bounded recursive search for a script downloaded outside the project.
    recursive_roots = [
        home / "Desktop", home / "桌面", home / "Documents",
        home / "OneDrive", home / "OneDrive - nny",
    ]
    for root in recursive_roots:
        if not root.is_dir():
            continue
        try:
            for candidate in root.glob("**/*"):
                try:
                    depth = len(candidate.relative_to(root).parts)
                except Exception:
                    depth = 99
                if depth > 7 or not is_excel(candidate):
                    continue
                if candidate.name.casefold() == expected_folded:
                    exact_hits.append(candidate.resolve())
                if semantic_match(candidate):
                    semantic_hits.append(candidate.resolve())
        except (OSError, PermissionError):
            continue

    resolved_exact = resolve_multiple(exact_hits, script_dir)
    if resolved_exact is not None:
        print(f"[DataPath] Found workbook recursively: {resolved_exact}")
        return str(resolved_exact)

    resolved_semantic = resolve_multiple(semantic_hits, script_dir)
    if resolved_semantic is not None:
        print(f"[DataPath] Found semantic workbook match: {resolved_semantic}")
        return str(resolved_semantic)

    nearby_text = "\n".join(f"  - {path}" for path in nearby_workbooks[:30]) or "  (none)"
    raise FileNotFoundError(
        f"Could not find '{expected_name}'. Put it beside the script, under "
        "data/processed, set DC24_LNP_DATA, or pass --data-path.\n\n"
        f"Python file actually running:\n  {Path(__file__).resolve()}\n\n"
        f"Nearby Excel workbooks:\n{nearby_text}"
    )

def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not HAS_SCIPY or len(y_true) < 3:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan

def safe_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not HAS_SCIPY or len(y_true) < 3:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    keys = [
        "R2", "RMSE", "MAE", "Spearman", "Pearson", "Top20_recall",
        "Calibration_slope", "Calibration_intercept", "Mean_bias",
    ]
    if len(yt) == 0:
        return {key: np.nan for key in keys}

    if len(yt) >= 2 and float(np.std(yt)) > 1e-12:
        slope, intercept = np.polyfit(yt, yp, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "R2": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
        "RMSE": rmse(yt, yp),
        "MAE": float(mean_absolute_error(yt, yp)),
        "Spearman": safe_spearman(yt, yp),
        "Pearson": safe_pearson(yt, yp),
        "Top20_recall": top_fraction_recall(yt, yp, 0.20),
        "Calibration_slope": float(slope) if np.isfinite(slope) else np.nan,
        "Calibration_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
        "Mean_bias": float(np.mean(yp - yt)),
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
    """Attach the response without silently substituting a different endpoint."""
    out = df.copy()

    if target_mode == "normalized":
        source = find_column(original_df, COLUMN_ALIASES["normalized_target"])
        if source is None:
            raise ValueError(
                "The normalized DC2.4 target column was not found. Expected one of: "
                + ", ".join(COLUMN_ALIASES["normalized_target"])
                + ". The publication pipeline intentionally does not fall back to a "
                  "different response column."
            )
        if "hacat" in str(source).casefold():
            raise ValueError(
                f"Refusing to use HaCaT-labelled column '{source}' as the DC2.4 response."
            )
        values = pd.to_numeric(original_df[source], errors="coerce")
        out["TE"] = values.values
        label = str(source)
    elif target_mode == "log10_raw":
        if not raw_target_column:
            raise ValueError("--raw-target-column is required with --target-mode log10_raw.")
        source = find_column(original_df, [raw_target_column])
        if source is None:
            raise ValueError(f"Raw target column '{raw_target_column}' was not found.")
        if "hacat" in str(source).casefold():
            raise ValueError(
                f"Refusing to use HaCaT-labelled column '{source}' as the DC2.4 response."
            )
        raw = pd.to_numeric(original_df[source], errors="coerce")
        out["TE"] = np.log10(raw.clip(lower=1e-12))
        label = f"log10({source})"
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    out = out.dropna(subset=["TE"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No finite response values remained after target parsing.")
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

    try:
        ordered = ([sheet_hint] if sheet_hint in xls.sheet_names else []) + list(xls.sheet_names)
        seen: set[str] = set()
        for sheet in ordered:
            if sheet in seen:
                continue
            seen.add(sheet)
            try:
                table = pd.read_excel(xls, sheet_name=sheet)
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
    finally:
        xls.close()
    return smiles_map

def build_morgan_map(smiles_map: Dict[str, str], n_bits: int = 128, radius: int = 2) -> Dict[str, np.ndarray]:
    """Generate Morgan fingerprints in the active interpreter when RDKit is available."""
    try:
        from rdkit import Chem, DataStructs
        try:
            from rdkit.Chem import rdFingerprintGenerator
            use_new = True
        except Exception:
            from rdkit.Chem import AllChem
            use_new = False
    except Exception as exc:
        print(
            "[Morgan] RDKit import failed in the active interpreter "
            f"({sys.executable}): {type(exc).__name__}: {exc}"
        )
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
    print(
        f"[Morgan] Generated {n_bits}-bit fingerprints for {len(result)} "
        "ionizable lipids with RDKit in the active interpreter."
    )
    return result


def _candidate_rdkit_interpreters() -> List[Path]:
    """Find isolated Python interpreters that may contain RDKit.

    The TabFM environment is intentionally isolated. RDKit may remain installed
    only in the original ``.venv``. Rather than importing packages across
    environments, this function locates that Python executable and uses it only
    as a subprocess to calculate deterministic fingerprint bit strings.
    """
    candidates: List[Path] = []

    original_python = os.environ.get("DC24_ORIGINAL_PYTHON", "").strip()
    if original_python:
        candidates.append(Path(original_python))
    candidates.append(Path(sys.executable))

    roots: List[Path] = []
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        env_root = Path(virtual_env)
        roots.extend([env_root.parent, env_root.parent.parent])

    current_python = Path(sys.executable).resolve()
    if len(current_python.parents) >= 3:
        roots.append(current_python.parents[2])

    try:
        script_dir = Path(__file__).resolve().parent
        roots.extend([script_dir, *list(script_dir.parents)[:6]])
    except Exception:
        pass
    roots.extend([Path.cwd(), Path.home() / "Desktop", Path.home() / "桌面"])

    env_names = [".venv", ".venv_clean", ".venv314_clean", ".venv_tabfm"]
    for root in roots:
        for env_name in env_names:
            candidates.extend([
                root / env_name / "Scripts" / "python.exe",
                root / env_name / "Scripts" / "python",
                root / env_name / "bin" / "python",
            ])

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = os.path.normcase(os.path.abspath(str(candidate)))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            unique.append(candidate)
    return unique


def build_morgan_map_with_external_rdkit(
    smiles_map: Dict[str, str],
    n_bits: int = 128,
    radius: int = 2,
    auto_repair: bool = True,
) -> Tuple[Dict[str, np.ndarray], str]:
    """Generate Morgan bits in an isolated RDKit subprocess.

    The active TabFM interpreter is tested first, followed by nearby project
    interpreters.  When RDKit is installed but its compiled DLLs are broken, the
    script performs one conservative wheel-only reinstall in the active
    environment (``pip --force-reinstall --no-deps rdkit``) and retries.  No
    training rows or target values are sent to the helper process.
    """
    payload = {
        "smiles_map": {
            lipid: str(smiles)
            for lipid, smiles in smiles_map.items()
            if lipid in IONIZABLE_DESCRIPTORS and str(smiles).strip()
        },
        "n_bits": int(n_bits),
        "radius": int(radius),
    }
    helper_code = r"""
import json, sys
payload = json.loads(sys.stdin.read())
from rdkit import Chem, rdBase
try:
    from rdkit.Chem import rdFingerprintGenerator
    use_new = True
except Exception:
    from rdkit.Chem import AllChem
    use_new = False
n_bits = int(payload["n_bits"])
radius = int(payload["radius"])
result = {}
for lipid, smiles in payload["smiles_map"].items():
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        continue
    if use_new:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    result[lipid] = [1 if ch == "1" else 0 for ch in fp.ToBitString()]
print(json.dumps({"rdkit_version": rdBase.rdkitVersion, "vectors": result}, separators=(",", ":")))
"""

    def clean_env_for(interpreter: Path) -> Dict[str, str]:
        env = os.environ.copy()
        for key in (
            "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP",
            "__PYVENV_LAUNCHER__",
        ):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
        env["VIRTUAL_ENV"] = str(interpreter.parent.parent)
        env["PATH"] = str(interpreter.parent) + os.pathsep + env.get("PATH", "")
        return env

    def try_interpreter(interpreter: Path) -> Tuple[Dict[str, np.ndarray], str, str]:
        try:
            completed = subprocess.run(
                [str(interpreter), "-I", "-c", helper_code],
                input=json.dumps(payload, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="backslashreplace",
                timeout=120,
                check=False,
                env=clean_env_for(interpreter),
                cwd=str(Path.home()),
            )
        except Exception as exc:
            return {}, "", f"{type(exc).__name__}: {exc}"

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            # Keep the diagnostic compact and encoding-safe.
            return {}, "", detail[-900:]
        try:
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            decoded = json.loads(lines[-1])
            vectors = decoded.get("vectors", {})
            mapped = {
                str(lipid): np.asarray(vector, dtype=float)
                for lipid, vector in vectors.items()
                if len(vector) == n_bits
            }
            if not mapped:
                return {}, "", "RDKit returned no valid vectors"
            rdkit_version = str(decoded.get("rdkit_version", "unknown"))
            EXTERNAL_RDKIT_INFO.clear()
            EXTERNAL_RDKIT_INFO.update({
                "python_executable": str(interpreter),
                "rdkit_version": rdkit_version,
                "mode": "isolated_subprocess_bridge",
            })
            return mapped, f"external_rdkit_{rdkit_version}", ""
        except Exception as exc:
            return {}, "", f"output parse failed: {type(exc).__name__}: {exc}"

    active = Path(sys.executable).resolve()
    candidates: List[Path] = [active]
    candidates.extend(_candidate_rdkit_interpreters())
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = os.path.normcase(os.path.abspath(str(candidate)))
        except Exception:
            continue
        if key not in seen and candidate.is_file():
            seen.add(key)
            unique.append(candidate)

    failures: List[str] = []
    for interpreter in unique:
        mapped, source, error = try_interpreter(interpreter)
        if mapped:
            print(
                f"[MorganBridge] Generated {n_bits}-bit fingerprints for "
                f"{len(mapped)} ionizable lipids via:\n  {interpreter}"
            )
            return mapped, source
        failures.append(f"{interpreter}: {error}")

    # A broken Windows RDKit DLL is commonly repaired by reinstalling the
    # platform wheel without touching NumPy/Torch dependencies.  This is tried
    # once and only in the isolated TabFM environment.
    repair_key = "DC24_RDKIT_REPAIR_ATTEMPTED"
    if auto_repair and os.environ.get(repair_key, "") != "1":
        os.environ[repair_key] = "1"
        print(
            "[MorganRepair] RDKit is missing or its DLLs are broken. "
            "Attempting one wheel-only repair in the active TabFM environment..."
        )
        repair_env = clean_env_for(active)
        repair_env[repair_key] = "1"
        try:
            repaired = subprocess.run(
                [
                    str(active), "-m", "pip", "install", "--upgrade",
                    "--force-reinstall", "--no-cache-dir", "--no-deps", "rdkit",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="backslashreplace",
                timeout=900,
                check=False,
                env=repair_env,
                cwd=str(Path.home()),
            )
            repair_output = (repaired.stdout or "").strip()
            if repaired.returncode == 0:
                print("[MorganRepair] RDKit wheel reinstall completed; retrying fingerprint generation.")
                mapped, source, error = try_interpreter(active)
                if mapped:
                    EXTERNAL_RDKIT_INFO["mode"] = "self_repaired_isolated_subprocess"
                    print(
                        f"[MorganBridge] Generated {n_bits}-bit fingerprints for "
                        f"{len(mapped)} ionizable lipids after RDKit repair."
                    )
                    return mapped, source + "_self_repaired"
                failures.append(f"post-repair {active}: {error}")
            else:
                failures.append(
                    "RDKit wheel repair failed: " + repair_output[-1200:]
                )
        except Exception as exc:
            failures.append(f"RDKit wheel repair exception: {type(exc).__name__}: {exc}")

    if failures:
        print("[MorganBridge] No RDKit route succeeded. Compact diagnostics:")
        for failure in failures[:8]:
            safe_failure = str(failure).encode("ascii", "backslashreplace").decode("ascii")
            print(f"  - {safe_failure}")
    return {}, "unavailable"

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
    """Load precomputed Morgan/ECFP bit columns from the workbook."""
    try:
        xls = pd.ExcelFile(workbook_path)
    except Exception:
        return {}, 128, "none"

    try:
        ordered = ([sheet_hint] if sheet_hint in xls.sheet_names else []) + list(xls.sheet_names)
        seen: set[str] = set()
        pattern = re.compile(r"^(fp|bit|morgan|ecfp)[_\- ]?(\d+)$", re.I)
        for sheet in ordered:
            if sheet in seen:
                continue
            seen.add(sheet)
            try:
                table = pd.read_excel(xls, sheet_name=sheet)
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
    finally:
        xls.close()
    return {}, 128, "none"

def prepare_morgan_features(
    workbook_path: str,
    smiles_map: Dict[str, str],
    requested_bits: int = 128,
    radius: int = 2,
    auto_repair: bool = True,
) -> Tuple[Dict[str, np.ndarray], int, str]:
    """Prepare Morgan features without mixing virtual environments.

    Order of preference:
    1. Precomputed fingerprint columns in the workbook.
    2. RDKit in the active interpreter.
    3. RDKit in a separate isolated project interpreter, usually ``.venv``.
    """
    excel_map, excel_bits, excel_source = load_precomputed_morgan_from_excel(workbook_path)
    if excel_map:
        return excel_map, excel_bits, excel_source

    rdkit_map = build_morgan_map(smiles_map, n_bits=requested_bits, radius=radius)
    if rdkit_map:
        try:
            import rdkit
            version = getattr(rdkit, "__version__", "unknown")
        except Exception:
            version = "unknown"
        return rdkit_map, requested_bits, f"rdkit_smiles_{version}"

    bridge_map, bridge_source = build_morgan_map_with_external_rdkit(
        smiles_map,
        n_bits=requested_bits,
        radius=radius,
        auto_repair=auto_repair,
    )
    if bridge_map:
        return bridge_map, requested_bits, bridge_source

    print(
        "[Morgan] No precomputed fingerprint columns were found, RDKit was not "
        "available in the TabFM interpreter, and no separate RDKit-enabled "
        "project interpreter could generate the fingerprints."
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
        "Mean_prediction_SD": float(oof["prediction_sd_across_repeats"].mean()),
    }

# -----------------------------------------------------------------------------
# Optional tabular foundation-model benchmarks: TabPFN and TabFM
# -----------------------------------------------------------------------------
def _is_transient_remote_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    tokens = (
        "ssl", "eof", "server disconnected", "connection", "connecterror",
        "getaddrinfo", "timed out", "timeout", "network", "httpx",
        "temporarily unavailable", "remote protocol", "readerror", "502", "503",
    )
    return any(token in message for token in tokens)


def _is_auth_or_license_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    tokens = (
        "unauthorized", "authentication", "access token", "api key", "forbidden",
        "401", "403", "license", "gated model", "accept the license",
    )
    return any(token in message for token in tokens)


def initialize_tabpfn_factory(
    backend_preference: str = "auto",
) -> Tuple[Any, str, str]:
    """Return a zero-argument TabPFNRegressor factory.

    Client access follows the user's previous working approach: an access token
    may be provided through PRIORLABS_API_KEY or TABPFN_TOKEN.  No credential is
    stored in this source file or written to output files.
    """
    errors: List[str] = []
    token = (
        os.environ.get("PRIORLABS_API_KEY", "")
        or os.environ.get("TABPFN_TOKEN", "")
    ).strip()
    os.environ.setdefault("TABPFN_NO_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    if backend_preference in {"auto", "client"}:
        try:
            import tabpfn_client
            from tabpfn_client import TabPFNRegressor as ClientRegressor

            if token:
                setter = getattr(tabpfn_client, "set_access_token", None)
                if callable(setter):
                    setter(token)
                os.environ["PRIORLABS_API_KEY"] = token
                os.environ["TABPFN_TOKEN"] = token

            def client_factory() -> Any:
                return ClientRegressor()

            note = (
                "TabPFN client backend. Training features and target values are "
                "sent to Prior Labs' hosted service."
            )
            return client_factory, "client", note
        except Exception as exc:
            errors.append(f"client: {type(exc).__name__}: {exc}")
            if backend_preference == "client":
                raise RuntimeError("TabPFN client could not be initialized: " + errors[-1]) from exc

    if backend_preference in {"auto", "local"}:
        try:
            from tabpfn import TabPFNRegressor as LocalRegressor

            def local_factory() -> Any:
                return LocalRegressor()

            note = (
                "TabPFN local backend. Model weights may be downloaded on first "
                "use and the Prior Labs license may require one-time acceptance."
            )
            return local_factory, "local", note
        except Exception as exc:
            errors.append(f"local: {type(exc).__name__}: {exc}")
            if backend_preference == "local":
                raise RuntimeError("TabPFN local backend could not be initialized: " + errors[-1]) from exc

    raise RuntimeError(
        "TabPFN is unavailable. Install 'tabpfn-client' for hosted inference or "
        "'tabpfn' for local inference. Details: " + " | ".join(errors)
    )


def initialize_tabfm_factory(
    backend: str = "pytorch",
) -> Tuple[Any, str, str]:
    """Load TabFM v1.0.0 once and return fresh sklearn-compatible wrappers."""
    if backend == "pytorch":
        try:
            from tabfm import TabFMRegressor
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        except Exception as exc:
            raise RuntimeError(
                "TabFM PyTorch backend could not be imported in the active "
                f"interpreter ({sys.executable}). Original error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    elif backend == "jax":
        try:
            from tabfm import TabFMRegressor
            from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
        except Exception as exc:
            raise RuntimeError(
                "TabFM JAX backend is unavailable. Install the official "
                "google-research/tabfm package with its JAX extra."
            ) from exc
    else:
        raise ValueError(f"Unsupported TabFM backend: {backend}")

    print(f"[TabFM] Loading v1.0.0 {backend} regression weights...")
    pretrained_model = tabfm_v1_0_0.load(model_type="regression")

    def factory() -> Any:
        # A new wrapper is created for each fold so its encoders/scalers cannot
        # carry information from another outer split.  The frozen pretrained
        # weights are shared to avoid re-downloading/reloading them 15 times.
        return TabFMRegressor(model=pretrained_model)

    note = (
        f"TabFM v1.0.0 {backend} backend. Pretrained weights are loaded from "
        "Hugging Face on first use and are governed by the TabFM non-commercial license."
    )
    return factory, f"tabfm_v1.0.0_{backend}", note


def _foundation_fit_predict(
    factory: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_name: str,
    retries: int,
    retry_sleep_sec: int,
) -> np.ndarray:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            estimator = factory()
            estimator.fit(X_train, y_train.to_numpy(dtype=float))
            prediction = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
            if len(prediction) != len(X_test):
                raise RuntimeError(
                    f"{model_name} returned {len(prediction)} predictions for "
                    f"{len(X_test)} test rows."
                )
            return prediction
        except Exception as exc:
            last_error = exc
            if _is_auth_or_license_error(exc) or not _is_transient_remote_error(exc):
                raise
            if attempt < max(1, retries):
                wait = retry_sleep_sec * attempt
                print(
                    f"[{model_name}] Transient remote error on attempt "
                    f"{attempt}/{retries}; retrying in {wait}s: {str(exc)[:180]}"
                )
                time.sleep(wait)
    assert last_error is not None
    raise last_error


def nested_cv_foundation_model(
    model_name: str,
    factory: Any,
    backend_label: str,
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    y: pd.Series,
    sample_ids: Sequence[str],
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: CVConfig,
    formulation_groups: Sequence[int],
    formulation_signatures: Sequence[str],
    retries: int = 3,
    retry_sleep_sec: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a foundation model and retain every repeat-specific OOF prediction."""
    prediction_lists: List[List[float]] = [[] for _ in range(len(y))]
    fold_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    split_prediction_rows: List[Dict[str, Any]] = []
    folds_per_repeat = len(outer_splits) // max(1, config.outer_repeats)

    for global_fold_id, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        repeat_id = (global_fold_id - 1) // folds_per_repeat + 1
        fold_id = (global_fold_id - 1) % folds_per_repeat + 1
        selected_aux = mrmr_select_auxiliary(
            X_aux.iloc[train_idx], y.iloc[train_idx], config.auxiliary_top_k
        )
        X_train = combine_selected_features(X_core, X_aux, train_idx, selected_aux)
        X_test = combine_selected_features(X_core, X_aux, test_idx, selected_aux)

        prediction = _foundation_fit_predict(
            factory=factory,
            X_train=X_train,
            y_train=y.iloc[train_idx],
            X_test=X_test,
            model_name=model_name,
            retries=retries,
            retry_sleep_sec=retry_sleep_sec,
        )
        for row_index, value in zip(test_idx, prediction):
            row_index = int(row_index)
            prediction_lists[row_index].append(float(value))
            split_prediction_rows.append({
                "model": model_name,
                "backend": backend_label,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "global_fold_id": global_fold_id,
                "sample_index": row_index,
                "candidate_id": str(sample_ids[row_index]),
                "formulation_group": int(formulation_groups[row_index]),
                "formulation_signature": str(formulation_signatures[row_index]),
                "actual": float(y.iloc[row_index]),
                "predicted": float(value),
                "residual": float(y.iloc[row_index] - value),
            })

        test_metrics = metric_dict(y.iloc[test_idx].values, prediction)
        fold_rows.append({
            "model": model_name,
            "backend": backend_label,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "global_fold_id": global_fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "n_core_features": X_core.shape[1],
            "n_aux_features": len(selected_aux),
        })
        for rank, feature in enumerate(selected_aux, start=1):
            feature_rows.append({
                "model": model_name,
                "backend": backend_label,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "global_fold_id": global_fold_id,
                "aux_rank": rank,
                "feature": feature,
            })
        print(
            f"[FoundationCV] {model_name:<12} repeat {repeat_id:02d} fold {fold_id:02d} "
            f"({global_fold_id:02d}/{len(outer_splits)}) R2={test_metrics['R2']:.3f} "
            f"RMSE={test_metrics['RMSE']:.3f} Spearman={test_metrics['Spearman']:.3f}"
        )

    oof_rows: List[Dict[str, Any]] = []
    for index, values in enumerate(prediction_lists):
        if len(values) != config.outer_repeats:
            raise RuntimeError(
                f"{model_name}: sample {index} received {len(values)} predictions; "
                f"expected {config.outer_repeats}."
            )
        oof_rows.append({
            "sample_index": index,
            "candidate_id": str(sample_ids[index]),
            "formulation_group": int(formulation_groups[index]),
            "formulation_signature": str(formulation_signatures[index]),
            "actual": float(y.iloc[index]),
            "predicted": float(np.mean(values)),
            "prediction_sd_across_repeats": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_predictions": len(values),
            "model": model_name,
            "backend": backend_label,
        })
    return (
        pd.DataFrame(oof_rows),
        pd.DataFrame(split_prediction_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(feature_rows),
    )

def run_foundation_benchmarks(
    args: argparse.Namespace,
    X_core: pd.DataFrame,
    X_aux: pd.DataFrame,
    y: pd.Series,
    sample_ids: Sequence[str],
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: CVConfig,
    output_dir: str,
    formulation_groups: Sequence[int],
    formulation_signatures: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run requested foundation models and record complete or failed status."""
    oof_frames: List[pd.DataFrame] = []
    split_prediction_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    feature_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    requests: List[Tuple[str, Any]] = []
    if not args.no_tabpfn:
        requests.append(("TabPFN", lambda: initialize_tabpfn_factory(args.tabpfn_backend)))
    if not args.no_tabfm:
        requests.append(("TabFM", lambda: initialize_tabfm_factory(args.tabfm_backend)))

    for model_name, initializer in requests:
        started = time.time()
        try:
            factory, backend_label, note = initializer()
            print(f"\n[FoundationModel] {model_name} backend={backend_label}")
            print(f"[FoundationModel] {note}")
            oof, split_predictions, folds, features = nested_cv_foundation_model(
                model_name=model_name,
                factory=factory,
                backend_label=backend_label,
                X_core=X_core,
                X_aux=X_aux,
                y=y,
                sample_ids=sample_ids,
                outer_splits=outer_splits,
                config=config,
                formulation_groups=formulation_groups,
                formulation_signatures=formulation_signatures,
                retries=args.foundation_retries,
                retry_sleep_sec=args.foundation_retry_sleep,
            )
            summary_rows.append(summarize_oof(oof, folds))
            oof_frames.append(oof)
            split_prediction_frames.append(split_predictions)
            fold_frames.append(folds)
            feature_frames.append(features)
            status_rows.append({
                "model": model_name,
                "status": "completed",
                "backend": backend_label,
                "note": note,
                "runtime_minutes": (time.time() - started) / 60.0,
                "error": "",
            })
            save_parity_plot(
                oof["actual"], oof["predicted"],
                os.path.join(output_dir, f"{model_name.lower()}_formulation_grouped_cv_parity.png"),
                f"{model_name}: repeated formulation-grouped CV",
                metric_dict(oof["actual"], oof["predicted"]),
            )
            save_residual_plot(
                oof["actual"], oof["predicted"],
                os.path.join(output_dir, f"{model_name.lower()}_formulation_grouped_cv_residuals.png"),
                f"{model_name}: repeated formulation-grouped CV residuals",
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[FoundationModel] {model_name} failed: {message}")
            status_rows.append({
                "model": model_name,
                "status": "failed_or_unavailable",
                "backend": "",
                "note": "",
                "runtime_minutes": (time.time() - started) / 60.0,
                "error": message,
            })

    return (
        pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame(),
        pd.concat(split_prediction_frames, ignore_index=True) if split_prediction_frames else pd.DataFrame(),
        pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame(),
        pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(),
        pd.DataFrame(summary_rows),
        pd.DataFrame(status_rows),
    )

# -----------------------------------------------------------------------------
# Cumulative round learning-curve evaluation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Leave-one-round-out evaluation
# -----------------------------------------------------------------------------
def normalize_round_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"
    text = str(value).strip()
    match = re.search(r"R\s*(\d+)", text, re.I)
    return f"R{match.group(1)}" if match else text


# -----------------------------------------------------------------------------
# Final deployment model
# -----------------------------------------------------------------------------
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
    actual_array, predicted_array = actual_array[valid], predicted_array[valid]
    if len(actual_array) == 0:
        return
    metrics = metrics or metric_dict(actual_array, predicted_array)
    lower = min(float(np.min(actual_array)), float(np.min(predicted_array)))
    upper = max(float(np.max(actual_array)), float(np.max(predicted_array)))
    padding = max((upper - lower) * 0.05, 1e-6)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(actual_array, predicted_array, alpha=0.8)
    ax.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        linestyle="--",
    )
    ax.set_xlabel("Experimental target")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)
    text = (
        f"R² = {metrics.get('R2', np.nan):.3f}\n"
        f"RMSE = {metrics.get('RMSE', np.nan):.3f}\n"
        f"MAE = {metrics.get('MAE', np.nan):.3f}\n"
        f"Spearman = {metrics.get('Spearman', np.nan):.3f}\n"
        f"Calibration slope = {metrics.get('Calibration_slope', np.nan):.3f}"
    )
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top")
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def save_residual_plot(
    actual: Sequence[float],
    predicted: Sequence[float],
    output_path: str,
    title: str,
) -> None:
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
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(predicted_array[valid], residual[valid], alpha=0.8)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Residual (experimental - predicted)")
    ax.set_title(title)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# External prospective validation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def save_excel(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

class TeeStream:
    """Mirror text to several streams while remaining TextIO-compatible.

    Some third-party packages (notably tabpfn-client, Hugging Face progress
    utilities, absl logging and tqdm) expect ``sys.stdout``/``sys.stderr`` to
    provide methods such as ``isatty()``, ``fileno()`` and ``close()``.  The
    previous lightweight tee implemented only ``write`` and ``flush``; this
    caused TabPFN import to fail with ``AttributeError: ... isatty`` and caused
    an additional exception during interpreter shutdown when absl called
    ``close``.  This proxy deliberately does not close the underlying console
    or run.log handle; ownership remains with ``main``/the finalizer.
    """

    def __init__(self, *streams: Any):
        self.streams = tuple(stream for stream in streams if stream is not None)

    def _primary_stream(self) -> Any:
        for stream in self.streams:
            if not bool(getattr(stream, "closed", False)):
                return stream
        return sys.__stdout__

    @property
    def encoding(self) -> str:
        return getattr(self._primary_stream(), "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._primary_stream(), "errors", None) or "backslashreplace"

    @property
    def closed(self) -> bool:
        return all(bool(getattr(stream, "closed", False)) for stream in self.streams)

    @staticmethod
    def _safe_for_stream(stream: Any, data: str) -> str:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            data.encode(encoding, errors="strict")
            return data
        except (UnicodeEncodeError, LookupError):
            try:
                return data.encode(encoding, errors="backslashreplace").decode(
                    encoding, errors="replace"
                )
            except Exception:
                return data.encode("ascii", errors="backslashreplace").decode("ascii")

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        for stream in self.streams:
            if bool(getattr(stream, "closed", False)):
                continue
            try:
                stream.write(data)
            except UnicodeEncodeError:
                try:
                    stream.write(self._safe_for_stream(stream, data))
                except Exception:
                    pass
            except Exception:
                # A logging/progress handler may write during interpreter
                # shutdown after one secondary stream has already closed.
                # Continue writing to the remaining live stream(s).
                continue
            try:
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            if bool(getattr(stream, "closed", False)):
                continue
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for stream in self.streams:
            try:
                if stream.isatty():
                    return True
            except Exception:
                continue
        return False

    def fileno(self) -> int:
        primary = self._primary_stream()
        fileno = getattr(primary, "fileno", None)
        if not callable(fileno):
            raise OSError("TeeStream has no file descriptor")
        return int(fileno())

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        # Do not close sys.__stdout__, sys.__stderr__ or run.log here.  External
        # logging frameworks call close() on their stream wrappers at shutdown,
        # while the script's finalizer owns the actual run.log file handle.
        self.flush()

    def __enter__(self) -> "TeeStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.flush()

    def __getattr__(self, name: str) -> Any:
        # Preserve compatibility with libraries that inspect less common
        # TextIO attributes (for example ``buffer`` or ``name``).
        return getattr(self._primary_stream(), name)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def git_commit_hash(start_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(start_dir),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=10, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "not a git checkout"
    except Exception:
        return "git unavailable"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_split_assignments(
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: CVConfig,
    sample_ids: Sequence[str],
    actual: Sequence[float],
    groups: Sequence[int],
    signatures: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    folds_per_repeat = len(outer_splits) // max(1, config.outer_repeats)
    all_indices = np.arange(len(sample_ids))
    for global_fold_id, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        repeat_id = (global_fold_id - 1) // folds_per_repeat + 1
        fold_id = (global_fold_id - 1) % folds_per_repeat + 1
        test_set = set(int(value) for value in test_idx)
        for sample_index in all_indices:
            sample_index = int(sample_index)
            rows.append({
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "global_fold_id": global_fold_id,
                "sample_index": sample_index,
                "candidate_id": str(sample_ids[sample_index]),
                "formulation_group": int(groups[sample_index]),
                "formulation_signature": str(signatures[sample_index]),
                "role": "test" if sample_index in test_set else "train",
                "actual": float(actual[sample_index]),
            })
    return pd.DataFrame(rows)


def summarize_repeat_predictions(split_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if split_predictions.empty:
        return pd.DataFrame()
    for (model, repeat_id), frame in split_predictions.groupby(["model", "repeat_id"], sort=True):
        metrics = metric_dict(frame["actual"], frame["predicted"])
        rows.append({
            "model": model,
            "repeat_id": int(repeat_id),
            "n": len(frame),
            **metrics,
        })
    return pd.DataFrame(rows)


def enrich_external_oof(
    oof: pd.DataFrame,
    reference: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Attach current formulation metadata to a prior tree-ensemble OOF file.

    Older tree-model result files do not always contain ``sample_index``.
    The cleaned/current reference table also historically lacked that explicit
    column because its DataFrame index already represented sample order.  This
    function creates a stable sample index, prefers candidate-ID matching when
    possible, validates coverage, and skips an incompatible optional tree OOF
    file instead of aborting the completed TabPFN/TabFM benchmark.
    """
    if oof.empty:
        return pd.DataFrame()

    frame = oof.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if "actual" not in frame.columns or "predicted" not in frame.columns:
        warnings.warn(
            "The optional tree OOF file lacks 'actual' and/or 'predicted'; "
            "tree-model integration was skipped.",
            RuntimeWarning,
        )
        return pd.DataFrame()

    frame["model"] = model_name

    metadata = reference.copy().reset_index(drop=True)
    if "sample_index" not in metadata.columns:
        metadata.insert(0, "sample_index", np.arange(len(metadata), dtype=int))

    required_metadata = [
        "sample_index", "candidate_id", "formulation_group",
        "formulation_signature",
    ]
    missing_metadata = [
        column for column in required_metadata if column not in metadata.columns
    ]
    if missing_metadata:
        warnings.warn(
            "Current training metadata is missing columns "
            f"{missing_metadata}; optional tree OOF integration was skipped.",
            RuntimeWarning,
        )
        return pd.DataFrame()

    metadata = metadata[required_metadata].copy()
    metadata["candidate_id"] = metadata["candidate_id"].astype(str)
    metadata["sample_index"] = pd.to_numeric(
        metadata["sample_index"], errors="coerce"
    )

    # Candidate IDs are safer than positional indices when an old result file
    # was produced before/after a QC filtering change.
    can_use_candidate = (
        "candidate_id" in frame.columns
        and frame["candidate_id"].notna().any()
        and not metadata["candidate_id"].duplicated().any()
    )

    if can_use_candidate:
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        # Always replace historical positional/group metadata with metadata
        # from the exact cleaned dataset used in this run.
        frame = frame.drop(
            columns=[
                "sample_index", "formulation_group",
                "formulation_signature",
            ],
            errors="ignore",
        )
        merged = frame.merge(
            metadata,
            on="candidate_id",
            how="left",
            validate="many_to_one",
        )
        merge_key = "candidate_id"
    elif "sample_index" in frame.columns:
        frame["sample_index"] = pd.to_numeric(
            frame["sample_index"], errors="coerce"
        )
        frame = frame.drop(
            columns=[
                "candidate_id", "formulation_group",
                "formulation_signature",
            ],
            errors="ignore",
        )
        merged = frame.merge(
            metadata,
            on="sample_index",
            how="left",
            validate="many_to_one",
        )
        merge_key = "sample_index"
    else:
        warnings.warn(
            "The optional tree OOF file contains neither a usable candidate_id "
            "nor sample_index; tree-model integration was skipped.",
            RuntimeWarning,
        )
        return pd.DataFrame()

    unmatched = merged["formulation_group"].isna()
    if unmatched.any():
        examples = merged.loc[unmatched, merge_key].astype(str).head(8).tolist()
        warnings.warn(
            f"The optional tree OOF file matched only "
            f"{len(merged) - int(unmatched.sum())}/{len(merged)} rows to the "
            "current cleaned dataset. To avoid an invalid model comparison, "
            f"tree-model integration was skipped. Unmatched examples: {examples}",
            RuntimeWarning,
        )
        return pd.DataFrame()

    merged["sample_index"] = pd.to_numeric(
        merged["sample_index"], errors="raise"
    ).astype(int)
    merged["formulation_group"] = pd.to_numeric(
        merged["formulation_group"], errors="raise"
    ).astype(int)
    return merged


def cluster_bootstrap_metrics(
    oof: pd.DataFrame,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    """Cluster bootstrap by exact formulation group."""
    if oof.empty or iterations <= 0 or "formulation_group" not in oof.columns:
        return {}, pd.DataFrame()
    groups = np.asarray(sorted(pd.unique(oof["formulation_group"])))
    group_indices = {
        group: np.where(oof["formulation_group"].to_numpy() == group)[0]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    metrics_to_keep = [
        "R2", "RMSE", "MAE", "Spearman", "Pearson", "Top20_recall",
        "Calibration_slope", "Calibration_intercept", "Mean_bias",
    ]
    rows: List[Dict[str, float]] = []
    for bootstrap_id in range(1, iterations + 1):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sample = oof.iloc[sampled_indices]
        values = metric_dict(sample["actual"], sample["predicted"])
        rows.append({"bootstrap_id": bootstrap_id, **values})
    distribution = pd.DataFrame(rows)
    alpha = (1.0 - confidence) / 2.0
    intervals: Dict[str, Tuple[float, float]] = {}
    for metric in metrics_to_keep:
        finite = pd.to_numeric(distribution[metric], errors="coerce").dropna()
        if len(finite) >= max(100, iterations // 10):
            intervals[metric] = (
                float(finite.quantile(alpha)),
                float(finite.quantile(1.0 - alpha)),
            )
    return intervals, distribution


def add_bootstrap_intervals(
    summary: pd.DataFrame,
    oof_all: pd.DataFrame,
    iterations: int,
    seed: int,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    distribution_frames: List[pd.DataFrame] = []
    for model_index, model_name in enumerate(summary["model"].astype(str)):
        oof = oof_all[oof_all["model"].astype(str) == model_name].copy()
        intervals, distribution = cluster_bootstrap_metrics(
            oof, iterations=iterations, seed=seed + model_index * 1009,
        )
        for metric, (low, high) in intervals.items():
            summary.loc[summary["model"].astype(str) == model_name, f"{metric}_CI_low"] = low
            summary.loc[summary["model"].astype(str) == model_name, f"{metric}_CI_high"] = high
        if not distribution.empty:
            distribution.insert(0, "model", model_name)
            distribution_frames.append(distribution)
    distributions = pd.concat(distribution_frames, ignore_index=True) if distribution_frames else pd.DataFrame()
    if not distributions.empty:
        distributions.to_csv(
            os.path.join(output_dir, "cluster_bootstrap_metric_distributions.csv"),
            index=False, encoding="utf-8-sig",
        )
    return summary, distributions


def paired_cluster_bootstrap_comparisons(
    oof_all: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Paired cluster-bootstrap differences for models evaluated on identical samples."""
    if oof_all.empty or iterations <= 0:
        return pd.DataFrame()
    models = sorted(pd.unique(oof_all["model"].astype(str)))
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            a = oof_all[oof_all["model"].astype(str) == model_a][
                ["sample_index", "actual", "predicted", "formulation_group"]
            ].rename(columns={"predicted": "predicted_a"})
            b = oof_all[oof_all["model"].astype(str) == model_b][
                ["sample_index", "predicted"]
            ].rename(columns={"predicted": "predicted_b"})
            paired = a.merge(b, on="sample_index", how="inner")
            if len(paired) < 10:
                continue
            groups = np.asarray(sorted(pd.unique(paired["formulation_group"])))
            group_indices = {
                group: np.where(paired["formulation_group"].to_numpy() == group)[0]
                for group in groups
            }
            metric_names = ["R2", "RMSE", "MAE", "Spearman", "Top20_recall"]
            distributions = {metric: [] for metric in metric_names}
            for _ in range(iterations):
                sampled_groups = rng.choice(groups, size=len(groups), replace=True)
                indices = np.concatenate([group_indices[group] for group in sampled_groups])
                sample = paired.iloc[indices]
                ma = metric_dict(sample["actual"], sample["predicted_a"])
                mb = metric_dict(sample["actual"], sample["predicted_b"])
                for metric in metric_names:
                    distributions[metric].append(ma[metric] - mb[metric])
            point_a = metric_dict(paired["actual"], paired["predicted_a"])
            point_b = metric_dict(paired["actual"], paired["predicted_b"])
            for metric in metric_names:
                values = pd.Series(distributions[metric], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
                if values.empty:
                    continue
                low, high = values.quantile([0.025, 0.975]).tolist()
                rows.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "difference_a_minus_b": float(point_a[metric] - point_b[metric]),
                    "CI_low": float(low),
                    "CI_high": float(high),
                    "bootstrap_iterations": iterations,
                    "interval_excludes_zero": bool(low > 0 or high < 0),
                })
    return pd.DataFrame(rows)


def build_environment_table(data_path: str) -> pd.DataFrame:
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "openpyxl", "joblib", "rdkit", "torch", "tabfm",
        "tabpfn-client", "tabpfn",
    ]
    rows = [
        {"component": "timestamp_utc", "version_or_value": datetime.now(timezone.utc).isoformat()},
        {"component": "python", "version_or_value": sys.version.replace("\n", " ")},
        {"component": "python_executable", "version_or_value": sys.executable},
        {"component": "platform", "version_or_value": platform.platform()},
        {"component": "machine", "version_or_value": platform.machine()},
        {"component": "hostname", "version_or_value": socket.gethostname()},
        {"component": "dataset_sha256", "version_or_value": sha256_file(data_path)},
        {"component": "script_sha256", "version_or_value": sha256_file(str(Path(__file__).resolve()))},
        {"component": "git_commit", "version_or_value": git_commit_hash(Path(__file__).resolve().parent)},
        {"component": "command", "version_or_value": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])},
    ]
    rows.extend({"component": package, "version_or_value": package_version(package)} for package in packages)
    if EXTERNAL_RDKIT_INFO:
        rows.extend(
            {"component": f"external_rdkit_{key}", "version_or_value": value}
            for key, value in EXTERNAL_RDKIT_INFO.items()
        )
    return pd.DataFrame(rows)


def build_column_audit(
    original: pd.DataFrame,
    source_columns: Dict[str, str],
    target_label: str,
    core_columns: Sequence[str],
    auxiliary_columns: Sequence[str],
) -> pd.DataFrame:
    feature_sources = set(source_columns.values())
    rows: List[Dict[str, Any]] = []
    for column in original.columns:
        name = str(column)
        lowered = name.casefold()
        if name == target_label:
            role = "response"
        elif name in feature_sources:
            role = "formulation_input_source"
        elif "hacat" in lowered:
            role = "excluded_HaCaT_column"
        else:
            role = "not_used"
        rows.append({"workbook_column": name, "role": role})
    rows.extend({"workbook_column": column, "role": "derived_core_feature"} for column in core_columns)
    rows.extend({"workbook_column": column, "role": "derived_auxiliary_feature"} for column in auxiliary_columns)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Foundation-model-only workflow
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe grouped-CV comparison of TabPFN and TabFM for "
            "normalized DC2.4 mRNA-LNP transfection efficiency."
        )
    )
    parser.add_argument(
        "--data-path",
        default="",
        help=(
            "Training Excel workbook. When omitted, the script searches the "
            "script directory, parent directories, the project directory, "
            "Desktop/OneDrive, and data/processed."
        ),
    )
    parser.add_argument("--sheet-name", default="0", help="Sheet name or zero-based index.")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument(
        "--tree-summary-path",
        default="",
        help=(
            "Optional model_summary CSV from the publication-safe tree-model run. "
            "When omitted, the script searches nearby lnp_outputs folders."
        ),
    )
    parser.add_argument(
        "--tree-oof-path",
        default="",
        help=(
            "Optional nested_cv_primary_ensemble_oof CSV. It is copied into the "
            "combined workbook for paper-figure preparation."
        ),
    )
    parser.add_argument(
        "--cumulative-summary-path",
        default="",
        help=(
            "Optional cumulative_round_primary_ensemble_summary CSV. When omitted, "
            "the script searches nearby lnp_outputs folders."
        ),
    )
    parser.add_argument("--target-mode", choices=["normalized", "log10_raw"], default="normalized")
    parser.add_argument(
        "--raw-target-column",
        default="DC_Cell_Transfection_Efficiency",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        help="Number of formulation-grouped outer folds.",
    )
    parser.add_argument(
        "--outer-repeats",
        type=int,
        default=3,
        help=(
            "Number of repeated formulation-grouped CV partitions. Default 3 is "
            "the formal manuscript setting; use --outer-repeats 1 for a quick test."
        ),
    )
    parser.add_argument(
        "--aux-top-k",
        type=int,
        default=8,
        help="Auxiliary molecular/Morgan features selected inside each training fold.",
    )
    parser.add_argument("--group-round-decimals", type=int, default=4)
    parser.add_argument("--pdi-max", type=float, default=0.5)
    parser.add_argument("--size-min", type=float, default=30.0)
    parser.add_argument("--size-max", type=float, default=300.0)
    parser.add_argument(
        "--allow-missing-qc",
        action="store_true",
        help="Keep rows with missing size/PDI; default is to exclude them.",
    )
    parser.add_argument("--no-tabpfn", action="store_true", help="Skip TabPFN.")
    parser.add_argument("--no-tabfm", action="store_true", help="Skip TabFM.")
    parser.add_argument(
        "--tabpfn-backend",
        choices=["auto", "client", "local"],
        default="auto",
        help="auto prefers tabpfn-client and falls back to local tabpfn.",
    )
    parser.add_argument(
        "--tabfm-backend",
        choices=["pytorch", "jax"],
        default="pytorch",
    )
    parser.add_argument("--foundation-retries", type=int, default=3)
    parser.add_argument("--foundation-retry-sleep", type=int, default=8)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=2000,
        help="Exact-formulation cluster-bootstrap iterations for 95%% confidence intervals.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Load, clean, audit and featurize data, then stop before fitting models.",
    )
    parser.add_argument(
        "--allow-model-failure", action="store_true",
        help="Permit a requested foundation model to fail without making the run unsuccessful.",
    )
    parser.add_argument(
        "--allow-missing-morgan", action="store_true",
        help=("Deprecated compatibility flag. Missing Morgan fingerprints now "
              "fall back to descriptor-only features by default."),
    )
    parser.add_argument(
        "--require-morgan", action="store_true",
        help="Stop the run if Morgan fingerprints remain unavailable after automatic repair.",
    )
    parser.add_argument(
        "--no-auto-repair-rdkit", action="store_true",
        help="Do not attempt the one-time wheel-only RDKit repair.",
    )
    return parser.parse_args()


def parse_sheet_argument(value: Any) -> Any:
    text_value = str(value)
    return int(text_value) if text_value.isdigit() else text_value


def save_foundation_comparison_plot(summary: pd.DataFrame, path: str) -> None:
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    plot_df = summary.copy().reset_index(drop=True)
    metrics = ["R2", "Spearman"]
    x = np.arange(len(plot_df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for index, metric in enumerate(metrics):
        values = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (index - 0.5) * width, values, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"].astype(str))
    ax.set_ylabel("Metric value")
    ax.set_title("Tabular foundation-model comparison")
    ax.axhline(0.0, linewidth=1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def _discover_result_file(
    data_path: str,
    explicit_path: str,
    filename_patterns: Sequence[str],
) -> str:
    """Return an explicit or nearby result file, preferring the newest match."""
    if explicit_path:
        path = os.path.abspath(os.path.expanduser(explicit_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Result file not found: {path}")
        return path

    roots = []
    data_dir = os.path.dirname(os.path.abspath(data_path))
    roots.extend([
        data_dir,
        os.path.join(data_dir, "lnp_outputs"),
        os.path.dirname(data_dir),
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ])
    matches: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for current_root, _, files in os.walk(root):
            # Limit recursive search depth to keep startup fast.
            rel = os.path.relpath(current_root, root)
            if rel != "." and rel.count(os.sep) > 3:
                continue
            for filename in files:
                lower = filename.lower()
                if any(fnmatch.fnmatch(lower, pattern.lower()) for pattern in filename_patterns):
                    matches.append(os.path.join(current_root, filename))
    if not matches:
        return ""
    return max(matches, key=os.path.getmtime)


def _load_optional_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _prepare_tree_summary(tree_summary: pd.DataFrame) -> pd.DataFrame:
    if tree_summary.empty:
        return tree_summary
    required = {"model", "R2", "RMSE", "MAE", "Spearman"}
    missing = required.difference(tree_summary.columns)
    if missing:
        raise ValueError(f"Tree summary is missing columns: {sorted(missing)}")
    out = tree_summary.copy()
    out["model_family"] = "Conventional ML"
    out["validation_label"] = "Repeated grouped nested CV"
    return out


def _prepare_foundation_summary(foundation_summary: pd.DataFrame) -> pd.DataFrame:
    out = foundation_summary.copy()
    if out.empty:
        return out
    out["model_family"] = "Tabular foundation model"
    out["validation_label"] = "Repeated formulation-grouped CV"
    return out


def save_paper_model_comparison_plots(summary: pd.DataFrame, output_dir: str) -> None:
    """Create publication panels with bootstrap confidence intervals when available."""
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    preferred = ["PrimaryTreeEnsemble", "TabFM", "TabPFN"]
    subset = summary[summary["model"].isin(preferred)].copy()
    if subset.empty:
        subset = summary.copy()
    ordering = {name: index for index, name in enumerate(preferred)}
    subset["_order"] = subset["model"].map(ordering).fillna(len(ordering))
    subset = subset.sort_values(["_order", "model"]).reset_index(drop=True)
    labels = subset["model"].replace({"PrimaryTreeEnsemble": "Tree ensemble"}).tolist()
    x = np.arange(len(subset))

    def errors(metric: str, values: np.ndarray) -> Optional[np.ndarray]:
        low_col, high_col = f"{metric}_CI_low", f"{metric}_CI_high"
        if low_col not in subset.columns or high_col not in subset.columns:
            return None
        low = pd.to_numeric(subset[low_col], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(subset[high_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(low) & np.isfinite(high)):
            return None
        return np.vstack([values - low, high - values])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    metrics = ["R2", "Spearman", "Top20_recall"]
    width = 0.24
    for index, metric in enumerate(metrics):
        if metric not in subset.columns:
            continue
        values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x + (index - 1) * width, values, width,
            yerr=errors(metric, values), capsize=3,
            label=metric.replace("Top20_recall", "Top-20% recall"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("Predictive accuracy and candidate-ranking performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "paper_model_accuracy_comparison.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "paper_model_accuracy_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    metrics = ["RMSE", "MAE"]
    width = 0.30
    for index, metric in enumerate(metrics):
        if metric not in subset.columns:
            continue
        values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x + (index - 0.5) * width, values, width,
            yerr=errors(metric, values), capsize=3, label=metric,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction error (normalized DC2.4 units)")
    ax.set_title("Prediction-error comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "paper_model_error_comparison.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "paper_model_error_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    args = parse_args()
    start_time = time.time()
    seed_everything(RANDOM_STATE)

    if args.no_tabpfn and args.no_tabfm and not args.validate_only:
        raise ValueError("Both TabPFN and TabFM were disabled; at least one model is required.")
    if args.outer_folds < 2 or args.outer_repeats < 1:
        raise ValueError("--outer-folds must be >=2 and --outer-repeats must be >=1.")
    if args.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations must be non-negative.")

    config = CVConfig(
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        inner_folds=2,
        tune_iter=1,
        auxiliary_top_k=args.aux_top_k,
        split_mode="grouped",
        group_round_decimals=args.group_round_decimals,
        random_state=RANDOM_STATE,
    )
    qc = QCConfig(
        args.pdi_max,
        args.size_min,
        args.size_max,
        require_complete=not args.allow_missing_qc,
    )

    data_path = find_existing_data_path(args.data_path or None)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(data_path),
        "lnp_outputs",
        "tabpfn_tabfm_publication_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
    )
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    global _RUN_LOG_HANDLE
    log_handle = open(os.path.join(output_dir, "run.log"), "w", encoding="utf-8", buffering=1)
    _RUN_LOG_HANDLE = log_handle
    sys.stdout = TeeStream(sys.__stdout__, log_handle)
    sys.stderr = TeeStream(sys.__stderr__, log_handle)

    print(f"[Version] {SCRIPT_VERSION}")
    print(f"[Data] {data_path}")
    print(f"[Output] {output_dir}")
    print(f"[Interpreter] {sys.executable}")
    print(f"[Python version] {sys.version.split()[0]}")
    print(f"[NumPy] version={np.__version__} | path={Path(np.__file__).resolve()}")
    if os.environ.get("DC24_TABFM_AUTO_REEXEC", "") == "1":
        print("[Environment] Clean isolated TabFM relaunch is active.")
    print(
        f"[Benchmark] TabPFN={not args.no_tabpfn} | TabFM={not args.no_tabfm} | "
        f"grouped outer CV={config.outer_folds} folds x {config.outer_repeats} repeat(s)"
    )
    print(
        f"[Bootstrap] exact-formulation cluster bootstrap={args.bootstrap_iterations} "
        f"iterations | seed={args.bootstrap_seed}"
    )

    original = pd.read_excel(data_path, sheet_name=parse_sheet_argument(args.sheet_name))
    standardized, source_columns = standardize_formulation_columns(original)
    standardized, target_label = attach_training_target(
        standardized, original, args.target_mode, args.raw_target_column,
    )
    standardized, invalid_removed = drop_invalid_rows(standardized)
    standardized, qc_removed = apply_qc_filter(standardized, qc)
    standardized = fill_missing_round_labels(standardized)

    if len(standardized) < 20:
        raise ValueError(f"Only {len(standardized)} training rows remained after cleaning/QC.")

    standardized = standardized.reset_index(drop=True)
    y = standardized["TE"].astype(float).reset_index(drop=True)
    sample_ids = standardized["candidate_id"].astype(str).tolist()
    formulation_groups, formulation_signatures = make_formulation_groups(
        standardized, config.group_round_decimals,
    )
    standardized["formulation_signature"] = formulation_signatures.values
    standardized["formulation_group"] = formulation_groups

    print(
        f"[Target] {target_label} | N={len(y)} | mean={y.mean():.4f} | "
        f"SD={y.std(ddof=1):.4f} | min={y.min():.4f} | max={y.max():.4f}"
    )
    if args.target_mode == "normalized" and (y.min() < -0.05 or y.max() > 1.05):
        warnings.warn(
            "The normalized target contains values outside approximately [0, 1]. "
            "Verify the normalization procedure before manuscript use.",
            RuntimeWarning,
        )

    smiles_map = load_smiles_map(data_path)
    morgan_map, morgan_n_bits, morgan_source = prepare_morgan_features(
        data_path, smiles_map, requested_bits=128, radius=2,
        auto_repair=not args.no_auto_repair_rdkit,
    )
    if not morgan_map:
        if args.require_morgan:
            raise RuntimeError(
                "Morgan fingerprints remain unavailable after all recovery attempts. "
                "Install a working RDKit wheel or add precomputed fp_0...fp_N columns."
            )
        morgan_source = "unavailable_descriptor_only"
        warnings.warn(
            "Morgan fingerprints are unavailable. The run will continue with the "
            "35 core formulation features plus 16 molecular descriptors. This is "
            "a valid descriptor-only sensitivity run, but it must not be presented "
            "as the Morgan-enabled primary analysis.",
            RuntimeWarning,
        )
        print(
            "[MorganFallback] Continuing safely without Morgan bits. "
            "Expected feature dimensions: core=35, auxiliary=16, total=51."
        )
    X_core, X_aux = build_feature_blocks(standardized, morgan_map, n_bits=morgan_n_bits)
    print(
        f"[Features] core={X_core.shape[1]} | auxiliary={X_aux.shape[1]} | "
        f"Morgan source={morgan_source} | bits={morgan_n_bits}"
    )

    outer_splits = make_outer_splits(len(y), config, groups=formulation_groups)
    split_assignments = build_split_assignments(
        outer_splits, config, sample_ids, y.to_numpy(), formulation_groups,
        formulation_signatures.astype(str).to_numpy(),
    )
    split_assignments.to_csv(
        os.path.join(output_dir, "split_assignments.csv"),
        index=False, encoding="utf-8-sig",
    )
    column_audit = build_column_audit(
        original, source_columns, target_label, X_core.columns, X_aux.columns,
    )
    column_audit.to_csv(
        os.path.join(output_dir, "column_audit.csv"),
        index=False, encoding="utf-8-sig",
    )
    environment_versions = build_environment_table(data_path)
    environment_versions.to_csv(
        os.path.join(output_dir, "environment_versions.csv"),
        index=False, encoding="utf-8-sig",
    )
    configuration = {
        "script_version": SCRIPT_VERSION,
        "arguments": vars(args),
        "resolved_data_path": os.path.abspath(data_path),
        "resolved_output_dir": output_dir,
        "random_state": RANDOM_STATE,
    }
    with open(os.path.join(output_dir, "configuration.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(configuration), handle, ensure_ascii=False, indent=2, allow_nan=False)

    if args.validate_only:
        validation_path = os.path.join(output_dir, "validation_only_audit.xlsx")
        save_excel(validation_path, {
            "training_data_used": standardized,
            "invalid_removed": invalid_removed,
            "qc_removed": qc_removed,
            "split_assignments": split_assignments,
            "column_audit": column_audit,
            "environment_versions": environment_versions,
        })
        print("[ValidateOnly] Data loading, QC, features, groups and splits validated successfully.")
        print(f"[ValidateOnly] Audit workbook: {validation_path}")
        return

    print(
        f"[FoundationCV] {len(outer_splits)} outer fits per requested model; "
        "no task-specific inner hyperparameter search."
    )
    (
        foundation_oof,
        foundation_split_predictions,
        foundation_folds,
        foundation_features,
        foundation_summary,
        foundation_status,
    ) = run_foundation_benchmarks(
        args=args,
        X_core=X_core,
        X_aux=X_aux,
        y=y,
        sample_ids=sample_ids,
        outer_splits=outer_splits,
        config=config,
        output_dir=output_dir,
        formulation_groups=formulation_groups,
        formulation_signatures=formulation_signatures.astype(str).to_numpy(),
    )

    foundation_status.to_csv(
        os.path.join(output_dir, "foundation_model_status.csv"),
        index=False, encoding="utf-8-sig",
    )
    requested_models = {
        model for model, requested in {
            "TabPFN": not args.no_tabpfn,
            "TabFM": not args.no_tabfm,
        }.items() if requested
    }
    completed_models = set(
        foundation_status.loc[foundation_status["status"] == "completed", "model"].astype(str)
    )
    failed_requested = sorted(requested_models - completed_models)
    if failed_requested:
        # Preserve every successfully completed fit before enforcing the formal
        # all-requested-models requirement.  In the previous version, a TabPFN
        # initialization error occurred only after all 15 TabFM folds had
        # completed, but the strict exception prevented those costly results
        # from being written to the normal tabular outputs.
        checkpoint_tables = {
            "foundation_status": foundation_status,
            "foundation_summary": foundation_summary,
            "OOF_mean_predictions": foundation_oof,
            "OOF_by_repeat": foundation_split_predictions,
            "fold_metrics": foundation_folds,
            "fold_selected_aux": foundation_features,
        }
        checkpoint_path = os.path.join(
            output_dir, "foundation_partial_checkpoint_before_failure.xlsx"
        )
        save_excel(checkpoint_path, checkpoint_tables)
        for filename, frame in {
            "foundation_partial_summary.csv": foundation_summary,
            "foundation_partial_oof_predictions.csv": foundation_oof,
            "foundation_partial_predictions_by_repeat.csv": foundation_split_predictions,
            "foundation_partial_fold_metrics.csv": foundation_folds,
        }.items():
            if not frame.empty:
                frame.to_csv(
                    os.path.join(output_dir, filename),
                    index=False,
                    encoding="utf-8-sig",
                )
        print(
            "[FoundationCheckpoint] Requested model failure detected; all "
            f"completed-model results were preserved at: {checkpoint_path}"
        )

    if failed_requested and not args.allow_model_failure:
        raise RuntimeError(
            "Requested model(s) failed: " + ", ".join(failed_requested)
            + ". Review foundation_model_status.csv, the partial checkpoint, "
              "and run.log. Use --allow-model-failure only for exploratory "
              "partial runs."
        )
    if foundation_summary.empty:
        raise RuntimeError("No requested foundation model completed.")

    repeat_metrics = summarize_repeat_predictions(foundation_split_predictions)
    foundation_summary = foundation_summary.sort_values(
        ["R2", "Spearman"], ascending=False,
    ).reset_index(drop=True)

    # Save the complete foundation-model results before touching any optional
    # historical tree-model files. Optional comparison inputs must never erase
    # a successful and potentially expensive TabPFN/TabFM run.
    completed_checkpoint_path = os.path.join(
        output_dir, "foundation_completed_checkpoint.xlsx"
    )
    save_excel(completed_checkpoint_path, {
        "foundation_summary": foundation_summary,
        "repeat_metrics": repeat_metrics,
        "fold_metrics": foundation_folds,
        "OOF_mean_predictions": foundation_oof,
        "OOF_by_repeat": foundation_split_predictions,
        "fold_selected_aux": foundation_features,
        "foundation_status": foundation_status,
    })
    for checkpoint_filename, checkpoint_frame in {
        "tabpfn_tabfm_summary.csv": foundation_summary,
        "tabpfn_tabfm_oof_predictions.csv": foundation_oof,
        "foundation_predictions_by_repeat.csv": foundation_split_predictions,
        "foundation_fold_metrics.csv": foundation_folds,
        "foundation_repeat_metrics.csv": repeat_metrics,
        "fold_selected_auxiliary_features.csv": foundation_features,
    }.items():
        if not checkpoint_frame.empty:
            checkpoint_frame.to_csv(
                os.path.join(output_dir, checkpoint_filename),
                index=False,
                encoding="utf-8-sig",
            )
    print(
        "[FoundationCheckpoint] Completed TabPFN/TabFM outputs saved before "
        f"optional tree-result integration: {completed_checkpoint_path}"
    )

    tree_summary_path = _discover_result_file(
        data_path, args.tree_summary_path,
        ["model_summary.csv", "model_summary(*).csv", "model_summary*.csv"],
    )
    tree_oof_path = _discover_result_file(
        data_path, args.tree_oof_path,
        ["nested_cv_primary_ensemble_oof.csv", "nested_cv_primary_ensemble_oof*.csv"],
    )
    cumulative_summary_path = _discover_result_file(
        data_path, args.cumulative_summary_path,
        ["cumulative_round_primary_ensemble_summary.csv", "cumulative_round_primary_ensemble_summary*.csv"],
    )
    tree_integration_error = ""
    try:
        tree_summary = _prepare_tree_summary(
            _load_optional_csv(tree_summary_path)
        )
        raw_tree_oof = _load_optional_csv(tree_oof_path)
        tree_oof = enrich_external_oof(
            raw_tree_oof, standardized, "PrimaryTreeEnsemble"
        )
        cumulative_summary = _load_optional_csv(cumulative_summary_path)
    except Exception as exc:
        tree_integration_error = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "Optional historical tree-result integration failed and was "
            "skipped. The completed TabPFN/TabFM benchmark remains valid and "
            f"saved. Details: {tree_integration_error}",
            RuntimeWarning,
        )
        tree_summary = pd.DataFrame()
        raw_tree_oof = pd.DataFrame()
        tree_oof = pd.DataFrame()
        cumulative_summary = pd.DataFrame()

    combined_model_summary = pd.concat(
        [_prepare_tree_summary(tree_summary), _prepare_foundation_summary(foundation_summary)],
        ignore_index=True, sort=False,
    )
    combined_oof = pd.concat(
        [frame for frame in [tree_oof, foundation_oof] if not frame.empty],
        ignore_index=True, sort=False,
    )
    combined_model_summary, bootstrap_distributions = add_bootstrap_intervals(
        combined_model_summary,
        combined_oof,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        output_dir=output_dir,
    )
    pairwise_comparisons = paired_cluster_bootstrap_comparisons(
        combined_oof,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 500_000,
    )

    # Propagate confidence intervals back to the foundation-only table.
    interval_columns = [
        column for column in combined_model_summary.columns
        if column.endswith("_CI_low") or column.endswith("_CI_high")
    ]
    foundation_summary = foundation_summary.drop(columns=interval_columns, errors="ignore").merge(
        combined_model_summary[["model", *interval_columns]], on="model", how="left",
    )

    if not tree_summary.empty:
        print(f"[TreeResults] Loaded: {tree_summary_path}")
    else:
        print("[TreeResults] No prior tree-model summary found; foundation-only outputs will be saved.")
    if raw_tree_oof.empty:
        print("[TreeResults] No tree ensemble OOF file found; tree bootstrap intervals were not computed.")

    print("\n[TabPFN / TabFM grouped-CV comparison]")
    print(foundation_summary.to_string(index=False))
    if not pairwise_comparisons.empty:
        print("\n[Paired cluster-bootstrap model comparisons]")
        print(pairwise_comparisons.to_string(index=False))

    save_paper_model_comparison_plots(combined_model_summary, output_dir)
    save_foundation_comparison_plot(
        foundation_summary,
        os.path.join(output_dir, "tabpfn_tabfm_comparison.png"),
    )

    feature_frequency = pd.DataFrame()
    if not foundation_features.empty:
        feature_frequency = (
            foundation_features.groupby(["model", "feature"], as_index=False)
            .agg(
                selection_count=("global_fold_id", "count"),
                selection_fraction=("global_fold_id", lambda values: len(values) / len(outer_splits)),
                mean_rank=("aux_rank", "mean"),
            )
            .sort_values(
                ["model", "selection_count", "mean_rank"],
                ascending=[True, False, True],
            )
        )

    runtime_minutes = (time.time() - start_time) / 60.0
    run_info = pd.DataFrame([
        {"item": "script_version", "value": SCRIPT_VERSION},
        {"item": "data_path", "value": data_path},
        {"item": "dataset_sha256", "value": sha256_file(data_path)},
        {"item": "script_sha256", "value": sha256_file(str(Path(__file__).resolve()))},
        {"item": "target_mode", "value": args.target_mode},
        {"item": "target_label", "value": target_label},
        {"item": "n_training_after_cleaning_QC", "value": len(standardized)},
        {"item": "outer_folds", "value": config.outer_folds},
        {"item": "outer_repeats", "value": config.outer_repeats},
        {"item": "split_mode", "value": "exact-formulation grouped"},
        {"item": "n_exact_formulation_groups", "value": len(np.unique(formulation_groups))},
        {"item": "auxiliary_top_k", "value": config.auxiliary_top_k},
        {"item": "core_feature_count", "value": X_core.shape[1]},
        {"item": "auxiliary_feature_count", "value": X_aux.shape[1]},
        {"item": "morgan_source", "value": morgan_source},
        {"item": "morgan_n_bits", "value": morgan_n_bits},
        {"item": "bootstrap_iterations", "value": args.bootstrap_iterations},
        {"item": "bootstrap_seed", "value": args.bootstrap_seed},
        {"item": "tabpfn_requested", "value": not args.no_tabpfn},
        {"item": "tabfm_requested", "value": not args.no_tabfm},
        {"item": "tabpfn_backend_preference", "value": args.tabpfn_backend},
        {"item": "tabfm_backend", "value": args.tabfm_backend},
        {"item": "tree_integration_error", "value": tree_integration_error},
        {"item": "runtime_minutes", "value": runtime_minutes},
    ])

    excel_path = os.path.join(output_dir, "tabpfn_tabfm_publication_results.xlsx")
    save_excel(excel_path, {
        "combined_model_summary": combined_model_summary,
        "foundation_summary": foundation_summary,
        "repeat_metrics": repeat_metrics,
        "fold_metrics": foundation_folds,
        "OOF_mean_predictions": foundation_oof,
        "OOF_by_repeat": foundation_split_predictions,
        "pairwise_bootstrap": pairwise_comparisons,
        "fold_selected_aux": foundation_features,
        "aux_selection_frequency": feature_frequency,
        "split_assignments": split_assignments,
        "tree_model_summary": tree_summary,
        "tree_primary_oof": tree_oof,
        "cumulative_rounds": cumulative_summary,
        "foundation_status": foundation_status,
        "training_data_used": standardized,
        "invalid_removed": invalid_removed,
        "qc_removed": qc_removed,
        "column_audit": column_audit,
        "environment_versions": environment_versions,
        "run_info": run_info,
    })

    csv_outputs = {
        "tabpfn_tabfm_summary.csv": foundation_summary,
        "paper_combined_model_summary.csv": combined_model_summary,
        "tabpfn_tabfm_oof_predictions.csv": foundation_oof,
        "foundation_predictions_by_repeat.csv": foundation_split_predictions,
        "foundation_fold_metrics.csv": foundation_folds,
        "foundation_repeat_metrics.csv": repeat_metrics,
        "fold_selected_auxiliary_features.csv": foundation_features,
        "auxiliary_feature_selection_frequency.csv": feature_frequency,
        "paired_cluster_bootstrap_comparisons.csv": pairwise_comparisons,
    }
    for filename, frame in csv_outputs.items():
        if not frame.empty:
            frame.to_csv(os.path.join(output_dir, filename), index=False, encoding="utf-8-sig")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": os.path.abspath(data_path),
        "dataset_sha256": sha256_file(data_path),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(str(Path(__file__).resolve())),
        "git_commit": git_commit_hash(Path(__file__).resolve().parent),
        "arguments": vars(args),
        "target_mode": args.target_mode,
        "target_label": target_label,
        "n_training": int(len(standardized)),
        "outer_folds": int(config.outer_folds),
        "outer_repeats": int(config.outer_repeats),
        "split_mode": "exact-formulation grouped",
        "n_exact_formulation_groups": int(len(np.unique(formulation_groups))),
        "morgan_source": morgan_source,
        "morgan_n_bits": int(morgan_n_bits),
        "external_rdkit_bridge": dict(EXTERNAL_RDKIT_INFO),
        "foundation_summary": foundation_summary.to_dict(orient="records"),
        "foundation_status": foundation_status.to_dict(orient="records"),
        "pairwise_bootstrap_comparisons": pairwise_comparisons.to_dict(orient="records"),
        "tree_summary_path": tree_summary_path,
        "tree_oof_path": tree_oof_path,
        "cumulative_summary_path": cumulative_summary_path,
        "tree_integration_error": tree_integration_error,
        "combined_model_summary": combined_model_summary.to_dict(orient="records"),
        "runtime_minutes": runtime_minutes,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n" + "=" * 88)
    print("[Done] Publication-oriented TabPFN / TabFM grouped-CV benchmark completed.")
    print(f"Results workbook: {excel_path}")
    print(f"Run manifest: {os.path.join(output_dir, 'run_manifest.json')}")
    print(f"Runtime: {runtime_minutes:.1f} min")
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
    finally:
        if _RUN_LOG_HANDLE is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            try:
                if not _RUN_LOG_HANDLE.closed:
                    _RUN_LOG_HANDLE.close()
            except Exception:
                pass
