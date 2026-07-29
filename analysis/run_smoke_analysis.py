#!/usr/bin/env python3
"""Portable smoke-run extraction of the certified analysis notebook.

Generated from ``certified_source_notebook.ipynb`` (SHA-256
``f41a9b1542f4a544eea2ff2085cc096743eb6f98d166a6e2dc0af4e96590ba13``).
Only runtime-path handling, dependency bootstrapping, and publication-mode
selection were changed. This file always uses the notebook's reduced-data smoke
profile and cannot create an authoritative publication completion sentinel. Use
the exact notebook for a complete publication rerun.
"""

import os
import re
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

warnings.resetwarnings()
warnings.simplefilter("default")

try:
    import google.colab  # type: ignore  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def report_runtime():
    if IN_COLAB:
        print("Google Colab detected. Mount storage separately and set PLOS_DATA_DIR/PLOS_OUTPUT_DIR.")
    else:
        print("Local runtime detected. Using environment variables or repository-relative paths.")


report_runtime()

data_override = os.environ.get("PLOS_DATA_DIR") or os.environ.get("DATA_DIR")
output_override = os.environ.get("PLOS_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR")

if data_override:
    DATA_DIR = Path(data_override).expanduser()
else:
    local_candidates = [Path.cwd() / "data", Path.cwd()]
    DATA_DIR = next(
        (candidate for candidate in local_candidates
         if all((candidate / name).is_file() for name in ("train.txt", "dev.txt", "test.txt"))),
        local_candidates[0],
    )

if output_override:
    OUTPUT_ROOT = Path(output_override).expanduser()
else:
    OUTPUT_ROOT = Path.cwd() / "outputs"

train_path = DATA_DIR / "train.txt"
dev_path = DATA_DIR / "dev.txt"
test_path = DATA_DIR / "test.txt"


def check_required_files(paths):
    missing = []
    print("Checking locked dataset files...")
    for path in paths:
        exists = path.is_file()
        print(f"  {'OK' if exists else 'MISSING'}: {path}")
        if not exists:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing PubMed RCT split file(s). Set PLOS_DATA_DIR to the folder "
            f"containing train.txt, dev.txt, and test.txt: {missing}"
        )
    return True


FILES_OK = check_required_files([train_path, dev_path, test_path])
print("DATA_DIR =", DATA_DIR)
print("OUTPUT_ROOT =", OUTPUT_ROOT)

# ============================================================
# PUBLICATION LOCK CONFIGURATION AND FAIL-CLOSED INPUT PREFLIGHT
# ============================================================
import hashlib
import importlib.util
import importlib.metadata
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


# ============================================================
# Built-in runtime provenance and Arial configuration
# ============================================================
# Do not hard-code this notebook's own SHA-256 here: adding that value would
# change the file and therefore change the byte-level hash. The certified hash
# is read from the launch environment or a private Colab secret instead.

_SOURCE_HASH_KEY = "PLOS_SOURCE_NOTEBOOK_SHA256"
_FONT_PATH_KEY = "PLOS_FONT_PATH"


def _read_private_colab_setting(name):
    if not IN_COLAB:
        return ""
    try:
        from google.colab import userdata

        return str(userdata.get(name) or "").strip()
    except Exception:
        return ""


_RUNTIME_CONFIG_SCHEMA = "plos_colab_runtime_config_v2"
_RUNTIME_CONFIG_RELEASE = "PLOS_R1_REVIEWER_ALIGNED_G4_FINAL_EXPORT_COMPLETE_CLASSICAL_PARAMS_20260729"


def _runtime_config_incompatible(candidate, origin, reason):
    message = (
        f"Runtime configuration {candidate} is incompatible: {reason}. "
        f"Use schema {_RUNTIME_CONFIG_SCHEMA!r} for release "
        f"{_RUNTIME_CONFIG_RELEASE!r}."
    )
    if origin == "explicit":
        raise RuntimeError(message)
    warnings.warn(
        message + " The auto-discovered file was ignored.",
        RuntimeWarning,
        stacklevel=2,
    )


def _load_runtime_config_file():
    candidates = []
    explicit = os.environ.get("PLOS_RUNTIME_CONFIG_PATH", "").strip()
    if explicit:
        candidates.append(("explicit", Path(explicit).expanduser()))
    if "DATA_DIR" in globals():
        candidates.append((
            "data directory",
            Path(DATA_DIR) / "PLOS_COLAB_RUNTIME_CONFIG.json",
        ))
    candidates.append((
        "working directory",
        Path.cwd() / "PLOS_COLAB_RUNTIME_CONFIG.json",
    ))
    seen = set()
    for origin, candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _runtime_config_incompatible(
                candidate, origin, f"unreadable JSON ({exc})"
            )
            continue
        if not isinstance(payload, dict):
            _runtime_config_incompatible(
                candidate, origin, "top-level value is not a JSON object"
            )
            continue
        if payload.get("schema_version") != _RUNTIME_CONFIG_SCHEMA:
            _runtime_config_incompatible(
                candidate,
                origin,
                f"schema is {payload.get('schema_version')!r}",
            )
            continue
        if payload.get("release_id") != _RUNTIME_CONFIG_RELEASE:
            _runtime_config_incompatible(
                candidate,
                origin,
                f"release is {payload.get('release_id')!r}",
            )
            continue
        return candidate.resolve(), payload
    return None, {}


def _configure_plos_runtime():
    _config_path, _config = _load_runtime_config_file()
    for _key in ("PLOS_REQUIRE_G4",):
        if not os.environ.get(_key, "").strip() and _key in _config:
            os.environ[_key] = str(_config[_key])

    _hash_sources = []
    _environment_hash = os.environ.get(_SOURCE_HASH_KEY, "").strip().lower()
    if _environment_hash:
        _hash_sources.append(("launch environment", _environment_hash))
    _secret_hash = _read_private_colab_setting(_SOURCE_HASH_KEY).lower()
    if _secret_hash:
        _hash_sources.append(("private Colab secret", _secret_hash))
    _config_hash = str(_config.get(_SOURCE_HASH_KEY, "")).strip().lower()
    if _config_hash:
        _hash_sources.append(("runtime configuration file", _config_hash))
    distinct_hashes = {value for _, value in _hash_sources}
    if len(distinct_hashes) > 1:
        raise RuntimeError(f"Conflicting source SHA-256 values: {_hash_sources}")
    if _hash_sources:
        os.environ[_SOURCE_HASH_KEY] = _hash_sources[0][1]

    _font_candidates = []
    _environment_font = os.environ.get(_FONT_PATH_KEY, "").strip()
    if _environment_font:
        _font_candidates.append(("launch environment", _environment_font))
    _secret_font = _read_private_colab_setting(_FONT_PATH_KEY)
    if _secret_font:
        _font_candidates.append(("private Colab secret", _secret_font))
    _config_font = str(_config.get(_FONT_PATH_KEY, "")).strip()
    if _config_font:
        _font_candidates.append(("runtime configuration file", _config_font))
    # A licensed font must be supplied explicitly; no local or cloud path is embedded.
    # This keeps the portable source free of machine-specific paths.

    _selected_font = None
    for _font_origin, _raw_font_path in _font_candidates:
        _font_candidate = Path(_raw_font_path).expanduser()
        try:
            _font_exists = _font_candidate.is_file()
        except OSError:
            _font_exists = False
        if _font_exists:
            _selected_font = (_font_origin, _font_candidate.resolve())
            break

    if _selected_font is None:
        os.environ.pop(_FONT_PATH_KEY, None)
    else:
        os.environ[_FONT_PATH_KEY] = str(_selected_font[1])

    _status = {
        "runtime_config_path": str(_config_path) if _config_path else None,
        "runtime_config_release_id": _config.get("release_id"),
        "source_hash_supplied": bool(os.environ.get(_SOURCE_HASH_KEY, "").strip()),
        "arial_configured": _selected_font is not None,
        "arial_origin": _selected_font[0] if _selected_font else None,
        "arial_path": str(_selected_font[1]) if _selected_font else None,
    }
    print("PLOS runtime config =", _status["runtime_config_path"] or "environment/secrets only")
    print("PLOS source SHA configured =", _status["source_hash_supplied"])
    print("PLOS Arial font =", _status["arial_path"] or "not yet available")
    return _status


PLOS_BOOTSTRAP_STATUS = _configure_plos_runtime()

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


# The portable Python extraction is deliberately smoke-only. The exact, hash-bound
# notebook in certified_source_notebook.ipynb remains the publication runner.
SMOKE_RUN = True
PUBLICATION_RUN = False
ALLOW_UNVERIFIED_DATA = _env_flag("PLOS_ALLOW_UNVERIFIED_DATA", False)
if PUBLICATION_RUN and ALLOW_UNVERIFIED_DATA:
    raise RuntimeError("PLOS_ALLOW_UNVERIFIED_DATA is forbidden during a publication run.")

SEED = 42
REFERENCE_SEED = 42
CLASSICAL_SEARCH_N_JOBS = max(1, min(8, (os.cpu_count() or 2) // 2))
CLASSICAL_INNER_ESTIMATOR_N_JOBS = 1
PLOS_REQUIRE_G4 = _env_flag("PLOS_REQUIRE_G4", True)
SOURCE_NOTEBOOK_RELEASE_ID = "PLOS_R1_REVIEWER_ALIGNED_G4_FINAL_EXPORT_COMPLETE_CLASSICAL_PARAMS_20260729"
SOURCE_NOTEBOOK_SHA256 = os.environ.get("PLOS_SOURCE_NOTEBOOK_SHA256", "").strip().lower()
if not SOURCE_NOTEBOOK_SHA256 and IN_COLAB:
    try:
        from google.colab import userdata
        SOURCE_NOTEBOOK_SHA256 = str(userdata.get("PLOS_SOURCE_NOTEBOOK_SHA256") or "").strip().lower()
    except Exception:
        SOURCE_NOTEBOOK_SHA256 = ""
if SOURCE_NOTEBOOK_SHA256 and not re.fullmatch(r"[0-9a-f]{64}", SOURCE_NOTEBOOK_SHA256):
    raise ValueError("PLOS_SOURCE_NOTEBOOK_SHA256 must be exactly 64 hexadecimal characters.")
if PUBLICATION_RUN and not SOURCE_NOTEBOOK_SHA256:
    raise RuntimeError(
        "Publication mode requires PLOS_SOURCE_NOTEBOOK_SHA256 from the certified release "
        "record. Set it in the launch environment or as a private Colab secret, restart, "
        "and Run all. This is a provenance value, not a Hugging Face token."
    )

if PUBLICATION_RUN:
    DEBUG_SMALL_MODE = False
    TRAIN_FRAC = 1.00
    DEV_FRAC = 1.00
    TEST_FRAC = 1.00
    SEQ_TRAIN_ABSTRACT_FRAC = 1.00
    SEQ_DEV_ABSTRACT_FRAC = 1.00
    MAX_TRANSFORMER_TRAIN = None
    MAX_TRANSFORMER_DEV = None
    MAX_TRANSFORMER_TEST = None
    N_FOLDS_CV = 5
    N_INNER_FOLDS_CV = 4
    TRANSFORMER_EPOCHS_FINAL = 2
    TRANSFORMER_BATCH_SIZE = 64
    TRANSFORMER_EVAL_BATCH_SIZE = 128
    TRANSFORMER_GRADIENT_ACCUMULATION_STEPS = 1
    TRANSFORMER_USE_FP16 = False
    TRANSFORMER_USE_BF16 = True
    TRANSFORMER_USE_TF32 = True
    TRANSFORMER_DATALOADER_WORKERS = max(2, min(8, (os.cpu_count() or 2) // 2))
    BILSTM_EPOCHS_FINAL = 5
    BILSTM_BATCH_SIZE = 64
    BILSTM_DATALOADER_WORKERS = max(2, min(8, (os.cpu_count() or 2) // 2))
else:
    # Reduced-data GPU integration mode. It never creates a publication sentinel.
    DEBUG_SMALL_MODE = True
    TRAIN_FRAC = 0.01
    DEV_FRAC = 0.05
    TEST_FRAC = 0.05
    SEQ_TRAIN_ABSTRACT_FRAC = TRAIN_FRAC
    SEQ_DEV_ABSTRACT_FRAC = DEV_FRAC
    MAX_TRANSFORMER_TRAIN = 3000
    MAX_TRANSFORMER_DEV = 1500
    MAX_TRANSFORMER_TEST = 1500
    N_FOLDS_CV = 3
    N_INNER_FOLDS_CV = 2
    TRANSFORMER_EPOCHS_FINAL = 1
    TRANSFORMER_BATCH_SIZE = 32
    TRANSFORMER_EVAL_BATCH_SIZE = 64
    TRANSFORMER_GRADIENT_ACCUMULATION_STEPS = 1
    TRANSFORMER_USE_FP16 = False
    TRANSFORMER_USE_BF16 = True
    TRANSFORMER_USE_TF32 = True
    TRANSFORMER_DATALOADER_WORKERS = max(2, min(4, (os.cpu_count() or 2) // 2))
    BILSTM_EPOCHS_FINAL = 2
    BILSTM_BATCH_SIZE = 32
    BILSTM_DATALOADER_WORKERS = max(2, min(4, (os.cpu_count() or 2) // 2))

RUN_CLASSICAL_CV = True
RUN_CLASSICAL_GRIDSEARCH = True
RUN_CRF = True
RUN_BILSTM_STYLE = True
RUN_TRANSFORMERS = True
CV_SHUFFLE = True
CV_RANDOM_STATE = SEED
REPORT_FOLD_CONFUSIONS = True

requested_run_id = os.environ.get("PLOS_RUN_ID", "").strip()
if requested_run_id:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", requested_run_id):
        raise ValueError("PLOS_RUN_ID may contain only letters, numbers, dot, underscore, and hyphen.")
    RUN_ID = requested_run_id
else:
    mode = "publication" if PUBLICATION_RUN else "smoke"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RUN_ID = f"PLOS_R1_{mode}_{stamp}"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = OUTPUT_ROOT / RUN_ID
# A unique directory prevents stale outputs from satisfying the final gate.
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
LOCAL_RUNTIME_CACHE_ROOT = OUTPUT_DIR / "_local_runtime_cache"
LOCAL_RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=False)
TRANSFORMER_SCRATCH_ROOT = LOCAL_RUNTIME_CACHE_ROOT / "transformer_scratch"
TRANSFORMER_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

EXPECTED_PUBLICATION_DATA = {
    "train": {"rows": 180040, "abstracts": 15000, "bytes": 30809461,
              "sha256": "2b1be3797844d060873d97e445aa4bf05fafbd050d526528364e233c17a7ced0"},
    "dev": {"rows": 30212, "abstracts": 2500, "bytes": 5166306,
            "sha256": "21019fb5fde4ed4fd67a358ae2a74aadf3b85540aa42ef2fb41d7a86d7aeda82"},
    "test": {"rows": 30135, "abstracts": 2500, "bytes": 5128560,
             "sha256": "b5dea844fecf520933f67f5dcd216c6cf1250f92946170cd93e55302fbb07d50"},
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PUBLICATION_FILE_PREFLIGHT = []
for split_name, path in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
    expected = EXPECTED_PUBLICATION_DATA[split_name]
    observed = {
        "split": split_name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    observed["verified"] = (
        observed["bytes"] == expected["bytes"]
        and observed["sha256"] == expected["sha256"]
    )
    PUBLICATION_FILE_PREFLIGHT.append(observed)

if not all(row["verified"] for row in PUBLICATION_FILE_PREFLIGHT):
    message = "Locked PubMed RCT byte-size/SHA-256 preflight failed: " + json.dumps(PUBLICATION_FILE_PREFLIGHT)
    if ALLOW_UNVERIFIED_DATA and not PUBLICATION_RUN:
        warnings.warn(message + " (nonpublication override active)")
    else:
        raise RuntimeError(message)

DEPENDENCY_BOOTSTRAP_LOG = []
try:
    installed_crf_version = importlib.metadata.version("sklearn-crfsuite")
except importlib.metadata.PackageNotFoundError:
    installed_crf_version = None
if RUN_CRF and installed_crf_version != "0.5.0":
    raise RuntimeError(
        "The smoke runner requires sklearn-crfsuite==0.5.0. Install the analysis extra "
        "before execution; the script never mutates its environment."
    )

REQUIRED_RUNTIME_MODULES = {
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "scipy": "scipy", "matplotlib": "matplotlib", "torch": "torch",
    "transformers": "transformers", "tokenizers": "tokenizers",
    "datasets": "datasets", "accelerate": "accelerate",
    "huggingface_hub": "huggingface-hub", "safetensors": "safetensors",
    "psutil": "psutil", "joblib": "joblib",
    "PIL": "Pillow", "packaging": "packaging",
}
missing_modules = [
    package for module, package in REQUIRED_RUNTIME_MODULES.items()
    if importlib.util.find_spec(module) is None
]
if missing_modules:
    raise RuntimeError(
        "Missing publication dependencies: " + ", ".join(missing_modules)
        + ". Install them in the clean GPU runtime, restart it, and Run all again."
    )

from packaging.specifiers import SpecifierSet
from packaging.version import Version

PUBLICATION_VERSION_CONSTRAINTS = {
    "numpy": ">=1.26,<3", "pandas": ">=2,<3", "scikit-learn": ">=1.5,<2",
    "scipy": ">=1.11,<2", "matplotlib": ">=3.8,<4", "torch": ">=2.7,<3",
    "transformers": ">=4.46,<6", "tokenizers": ">=0.20,<0.24",
    "datasets": ">=2.19,<5", "accelerate": ">=1.1,<2",
    "huggingface-hub": ">=0.26,<2", "safetensors": ">=0.4.3,<1",
    "psutil": ">=5.9,<8", "sklearn-crfsuite": "==0.5.0",
    "Pillow": ">=10,<13",
}
incompatible_versions = []
for package, constraint in PUBLICATION_VERSION_CONSTRAINTS.items():
    observed_version = importlib.metadata.version(package)
    if Version(observed_version) not in SpecifierSet(constraint):
        incompatible_versions.append(f"{package}=={observed_version} not in {constraint}")
if incompatible_versions:
    raise RuntimeError(
        "Unsupported publication dependency versions: " + "; ".join(incompatible_versions)
    )
if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
    raise RuntimeError(
        f"Publication runtime requires Python >=3.10,<3.15; found {sys.version.split()[0]}"
    )

from matplotlib import font_manager as _runtime_font_manager

_runtime_font_setting = os.environ.get("PLOS_FONT_PATH", "").strip()
if _runtime_font_setting:
    _runtime_font_file = Path(_runtime_font_setting).expanduser()
    if not _runtime_font_file.is_file():
        raise FileNotFoundError(
            "PLOS_FONT_PATH does not identify an existing font file: "
            f"{_runtime_font_file}"
        )
    try:
        _runtime_font_manager.fontManager.addfont(str(_runtime_font_file))
    except Exception as exc:
        raise RuntimeError(
            f"The configured Arial file is unreadable: {_runtime_font_file}"
        ) from exc

try:
    _runtime_arial_file = _runtime_font_manager.findfont(
        "Arial", fallback_to_default=False
    )
except ValueError as exc:
    raise RuntimeError(
        "Arial is unavailable. Supply a legitimately licensed font file through "
        "the PLOS_FONT_PATH environment variable; the repository does not redistribute it."
    ) from exc

_runtime_arial_family = _runtime_font_manager.FontProperties(
    fname=_runtime_arial_file
).get_name()
if _runtime_arial_family.casefold() != "arial":
    raise RuntimeError(
        "The resolved publication font is not Arial: "
        f"family={_runtime_arial_family!r}, file={_runtime_arial_file!r}"
    )
PLOS_EARLY_FONT_PREFLIGHT = {
    "family": _runtime_arial_family,
    "path": str(_runtime_arial_file),
}
print("Early Arial preflight passed:", PLOS_EARLY_FONT_PREFLIGHT)


required_cublas_workspace = ":4096:8"
existing_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if existing_cublas_workspace not in {None, required_cublas_workspace}:
    raise RuntimeError(
        f"CUBLAS_WORKSPACE_CONFIG={existing_cublas_workspace!r} conflicts with "
        f"the locked value {required_cublas_workspace!r}. Restart with the locked value."
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_cublas_workspace


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(TRANSFORMER_USE_TF32)
    torch.backends.cudnn.allow_tf32 = bool(TRANSFORMER_USE_TF32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)


set_seed(SEED)
import torch

if RUN_TRANSFORMERS and not torch.cuda.is_available():
    raise RuntimeError("Publication execution requires the requested Colab G4 CUDA GPU.")
if RUN_TRANSFORMERS and torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"The locked protocol requires exactly one visible CUDA GPU; found {torch.cuda.device_count()}."
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_properties = torch.cuda.get_device_properties(0)
gpu_vram_gb = gpu_properties.total_memory / (1024 ** 3)
gpu_capability = tuple(torch.cuda.get_device_capability(0))
gpu_arch_list = list(torch.cuda.get_arch_list())
gpu_bf16_supported = bool(torch.cuda.is_bf16_supported())
cuda_build_version = Version(str(torch.version.cuda or "0"))
gpu_is_g4 = bool(re.search(r"RTX\s+PRO\s+6000.*Blackwell", gpu_name, flags=re.IGNORECASE))

PUBLICATION_GPU_PREFLIGHT = {
    "cuda_available": True,
    "device_name": gpu_name,
    "device_vram_gb": gpu_vram_gb,
    "compute_capability": list(gpu_capability),
    "torch_cuda_build": str(torch.version.cuda),
    "torch_arch_list": gpu_arch_list,
    "bf16_supported": gpu_bf16_supported,
    "requested_colab_gpu_type": "G4",
    "g4_name_match": gpu_is_g4,
}
if PUBLICATION_RUN and not PLOS_REQUIRE_G4:
    raise RuntimeError("Publication mode cannot disable the locked G4 hardware requirement.")
if PUBLICATION_RUN:
    g4_failures = []
    if not gpu_is_g4:
        g4_failures.append(f"device name is {gpu_name!r}, not RTX PRO 6000 Blackwell")
    if gpu_vram_gb < 80.0:
        g4_failures.append(f"VRAM is {gpu_vram_gb:.1f} GB; full G4 profile requires at least 80 GB")
    if gpu_capability != (12, 0):
        g4_failures.append(f"compute capability is {gpu_capability}, expected (12, 0)")
    if cuda_build_version < Version("12.8"):
        g4_failures.append(f"PyTorch CUDA build is {torch.version.cuda}; G4 requires >=12.8")
    if "sm_120" not in gpu_arch_list:
        g4_failures.append("PyTorch arch list does not contain sm_120")
    if not gpu_bf16_supported:
        g4_failures.append("BF16 is not supported")
    if g4_failures:
        raise RuntimeError("G4 publication preflight failed: " + "; ".join(g4_failures))
if TRANSFORMER_USE_BF16 and not gpu_bf16_supported:
    raise RuntimeError("The configured BF16 profile is unsupported by the assigned GPU.")

print("PUBLICATION_RUN =", PUBLICATION_RUN)
print("SMOKE_RUN =", SMOKE_RUN)
print("RUN_ID =", RUN_ID)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("REFERENCE_SEED =", REFERENCE_SEED)
print("DATA PREFLIGHT =", json.dumps(PUBLICATION_FILE_PREFLIGHT, indent=2))
print("GPU PREFLIGHT =", json.dumps(PUBLICATION_GPU_PREFLIGHT, indent=2))

# ============================================================
# PUBLICATION RUN METADATA AND ENVIRONMENT SNAPSHOT
# ============================================================
import importlib.metadata
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import psutil


def safe_get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def safe_command(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


nvidia_query = safe_command([
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
])
package_names = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "transformers",
    "datasets", "accelerate", "tokenizers", "huggingface-hub", "safetensors",
    "sklearn-crfsuite", "torch", "psutil", "joblib", "Pillow", "packaging",
]
ENVIRONMENT_SNAPSHOT = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256 if SOURCE_NOTEBOOK_SHA256 else None,
    "python": sys.version.replace("\n", " "),
    "implementation": platform.python_implementation(),
    "os_platform": platform.platform(),
    "kernel_release": platform.release(),
    "machine": platform.machine(),
    "processor": platform.processor() or platform.uname().processor,
    "hostname": socket.gethostname(),
    "logical_cpu_count": psutil.cpu_count(logical=True),
    "physical_cpu_count": psutil.cpu_count(logical=False),
    "total_ram_gb": psutil.virtual_memory().total / (1024 ** 3),
    "nvidia_smi": nvidia_query,
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "publication_run": PUBLICATION_RUN,
    "smoke_run": SMOKE_RUN,
    "dependency_bootstrap_log": DEPENDENCY_BOOTSTRAP_LOG,
    "publication_version_constraints": PUBLICATION_VERSION_CONSTRAINTS,
    "publication_file_preflight": PUBLICATION_FILE_PREFLIGHT,
    "publication_gpu_preflight": PUBLICATION_GPU_PREFLIGHT,
    "seed": SEED,
    "reference_seed": REFERENCE_SEED,
    "train_frac": TRAIN_FRAC,
    "dev_frac": DEV_FRAC,
    "test_frac": TEST_FRAC,
    "n_outer_folds": N_FOLDS_CV,
    "n_inner_folds": N_INNER_FOLDS_CV,
    "classical_search_n_jobs": CLASSICAL_SEARCH_N_JOBS,
    "classical_inner_estimator_n_jobs": CLASSICAL_INNER_ESTIMATOR_N_JOBS,
    "requested_colab_gpu_type": "G4",
    "require_g4": PLOS_REQUIRE_G4,
    "transformer_epochs_final": TRANSFORMER_EPOCHS_FINAL,
    "transformer_batch_size": TRANSFORMER_BATCH_SIZE,
    "transformer_eval_batch_size": TRANSFORMER_EVAL_BATCH_SIZE,
    "transformer_bf16": TRANSFORMER_USE_BF16,
    "transformer_fp16": TRANSFORMER_USE_FP16,
    "transformer_tf32": TRANSFORMER_USE_TF32,
    "transformer_dataloader_workers": TRANSFORMER_DATALOADER_WORKERS,
    "bilstm_epochs_final": BILSTM_EPOCHS_FINAL,
    "bilstm_batch_size": BILSTM_BATCH_SIZE,
    "bilstm_dataloader_workers": BILSTM_DATALOADER_WORKERS,
    "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
    "colab_backend_version": os.environ.get("COLAB_BACKEND_VERSION"),
}
ENVIRONMENT_SNAPSHOT.update({
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_vram_gb": (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available() else None
    ),
})
for package in package_names:
    ENVIRONMENT_SNAPSHOT[f"package__{package}"] = safe_get_package_version(package)

env_json_path = OUTPUT_DIR / "environment_snapshot.json"
env_json_path.write_text(
    json.dumps(ENVIRONMENT_SNAPSHOT, indent=2, default=str), encoding="utf-8"
)
requirements_path = OUTPUT_DIR / "requirements_snapshot.txt"
freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
requirements_path.write_text(freeze_output, encoding="utf-8")

display(pd.DataFrame([ENVIRONMENT_SNAPSHOT]).T.rename(columns={0: "value"}))
print("Saved environment snapshot:", env_json_path)
print("Saved exact package snapshot:", requirements_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("error", category=ConvergenceWarning)
print("Core packages loaded; scikit-learn convergence warnings are fatal.")

# ============================================================
# REPRODUCIBILITY INVARIANTS
# ============================================================
EXPERIMENT_INVARIANTS = {
    "research_focus": "standard performance, calibration, and automatic prediction sensitivity/consistency",
    "dataset": "fixed PubMed RCT sentence-classification benchmark; previously evaluated in the original submission",
    "seed": SEED,
    "reference_seed": REFERENCE_SEED,
    "final_training_policy": "training split only; development is not merged into final training",
    "classical_selection_policy": "training-only 5x4 stratified grouped nested CV; four-fold full-training refit search",
    "neural_selection_policy": "development Macro-F1 with fixed model-ID tie-break; no classwise or item-level development analysis",
    "test_policy": "revised systems and comparisons are hash-locked before corrected-run test prediction; prior-submission access is disclosed",
    "uncertainty_scope": "cluster intervals condition on fixed OOF or seed-42 predictions; seed SD separately describes training variability",
    "perturbation_policy": "post-lock prediction-change analysis; no semantic-validity or correctness claim",
}

set_seed(SEED)
ENVIRONMENT_SNAPSHOT["experiment_invariants"] = EXPERIMENT_INVARIANTS
ENVIRONMENT_SNAPSHOT["snapshot_refreshed_after_installs"] = datetime.now(timezone.utc).isoformat(
    timespec="seconds"
)
requirements_path.write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    encoding="utf-8",
)
env_json_path.write_text(
    json.dumps(ENVIRONMENT_SNAPSHOT, indent=2, default=str), encoding="utf-8"
)
print(json.dumps(EXPERIMENT_INVARIANTS, indent=2))

# ============================================================
# COMPUTATIONAL EFFICIENCY INSTRUMENTATION
# ============================================================
# Purpose:
#   Provide reusable helpers for measuring training time, inference time,
#   memory footprint, model size, and parameter counts.
#
# Interpretation:
#   These are implementation-specific measurements under this notebook environment.
#   They should be reported as practical deployment indicators, not universal
#   properties of the algorithms.
# ============================================================

from pathlib import Path
import os
import gc
import sys
import time
import json
import math
import pickle
import tempfile
import threading
import platform
import subprocess
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import joblib
except Exception:
    joblib = None

# ------------------------------------------------------------
# Output directory (defined by the locked runtime setup)
# ------------------------------------------------------------

if "OUTPUT_DIR" not in globals():
    raise RuntimeError("Run the locked runtime/data setup before efficiency instrumentation.")
OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Efficiency settings
# ------------------------------------------------------------

EFFICIENCY_RANDOM_STATE = SEED if "SEED" in globals() else 42

# For publication, keep None to use the complete fixed public benchmark test split.
# For a quick smoke test, change to e.g. 2000.
EFFICIENCY_BENCHMARK_SAMPLE_N = None

# The locked final evaluation performs one warm-up and five raw inference repeats.
EFFICIENCY_INFERENCE_REPEATS = 5
EFFICIENCY_DO_WARMUP = True
EFFICIENCY_WARMUP_N = 128

if "efficiency_records" not in globals():
    efficiency_records = []


# ------------------------------------------------------------
# Hardware and package metadata
# ------------------------------------------------------------

def safe_package_version(package_name):
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def get_hardware_snapshot():
    snapshot = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "psutil_available": psutil is not None,
    }

    if psutil is not None:
        try:
            snapshot["total_ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            snapshot["total_ram_gb"] = np.nan

    if torch is not None:
        snapshot["torch_version"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
            snapshot["gpu_count"] = torch.cuda.device_count()
        else:
            snapshot["gpu_name"] = None
            snapshot["gpu_count"] = 0
    else:
        snapshot["torch_version"] = None
        snapshot["cuda_available"] = False
        snapshot["cuda_version"] = None
        snapshot["gpu_name"] = None
        snapshot["gpu_count"] = 0

    for pkg in [
        "numpy", "pandas", "scikit-learn", "scipy", "torch",
        "transformers", "datasets", "sklearn-crfsuite", "joblib", "psutil"
    ]:
        snapshot[pkg] = safe_package_version(pkg)

    return snapshot


compute_hardware_snapshot = get_hardware_snapshot()

compute_hardware_snapshot_path = OUTPUT_DIR / "compute_hardware_snapshot.json"
with open(compute_hardware_snapshot_path, "w", encoding="utf-8") as f:
    json.dump(compute_hardware_snapshot, f, indent=2, default=str)

print("Compute hardware snapshot saved to:")
print(compute_hardware_snapshot_path)


# ------------------------------------------------------------
# Memory monitor
# ------------------------------------------------------------

class PeakCPUMemoryMonitor:
    """Sample resident memory of the current Python process during an operation."""

    def __init__(self, interval_seconds=0.05):
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None
        self.start_rss_gb = np.nan
        self.peak_rss_gb = np.nan
        self.end_rss_gb = np.nan

    def _get_rss_gb(self):
        if psutil is None:
            return np.nan
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    def _sample_loop(self):
        peak = self.start_rss_gb
        while not self._stop_event.is_set():
            current = self._get_rss_gb()
            if not np.isnan(current):
                peak = current if np.isnan(peak) else max(peak, current)
            time.sleep(self.interval_seconds)
        self.end_rss_gb = self._get_rss_gb()
        self.peak_rss_gb = peak

    def start(self):
        self.start_rss_gb = self._get_rss_gb()
        self.peak_rss_gb = self.start_rss_gb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.end_rss_gb = self._get_rss_gb()
        return {
            "cpu_start_rss_gb": self.start_rss_gb,
            "cpu_peak_rss_gb": self.peak_rss_gb,
            "cpu_end_rss_gb": self.end_rss_gb,
            "cpu_peak_increment_gb": (
                self.peak_rss_gb - self.start_rss_gb
                if not np.isnan(self.peak_rss_gb) and not np.isnan(self.start_rss_gb)
                else np.nan
            ),
        }


def reset_gpu_peak_memory():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_allocated_gb():
    if torch is None or not torch.cuda.is_available():
        return np.nan
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def get_gpu_peak_reserved_gb():
    if torch is None or not torch.cuda.is_available():
        return np.nan
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


def get_device_name():
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


# ------------------------------------------------------------
# Timing wrapper
# ------------------------------------------------------------

def time_efficiency_stage(model, stage, callable_fn, n_examples=None, notes=""):
    """
    Time one training or inference operation and append a row to efficiency_records.

    Parameters
    ----------
    model : str
        Model identifier.
    stage : str
        e.g. 'final_training', 'final_test_inference', or 'final_training_plus_prediction'.
    callable_fn : callable
        Zero-argument function that performs the operation.
    n_examples : int or None
        Number of training or prediction examples.
    notes : str
        Protocol details, such as epochs, batch size, input column, or model family.
    """
    gc.collect()
    reset_gpu_peak_memory()

    memory_monitor = PeakCPUMemoryMonitor()
    memory_monitor.start()

    start = time.perf_counter()
    try:
        result = callable_fn()
    finally:
        elapsed_seconds = time.perf_counter() - start
        cpu_stats = memory_monitor.stop()

    row = {
        "model": str(model),
        "stage": str(stage),
        "n_examples": int(n_examples) if n_examples is not None and not pd.isna(n_examples) else np.nan,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
        "seconds_per_1000_examples": (
            elapsed_seconds / n_examples * 1000
            if n_examples is not None and n_examples > 0
            else np.nan
        ),
        "examples_per_second": (
            n_examples / elapsed_seconds
            if n_examples is not None and n_examples > 0 and elapsed_seconds > 0
            else np.nan
        ),
        "cpu_start_rss_gb": cpu_stats["cpu_start_rss_gb"],
        "cpu_peak_rss_gb": cpu_stats["cpu_peak_rss_gb"],
        "cpu_end_rss_gb": cpu_stats["cpu_end_rss_gb"],
        "cpu_peak_increment_gb": cpu_stats["cpu_peak_increment_gb"],
        "gpu_peak_allocated_gb": get_gpu_peak_allocated_gb(),
        "gpu_peak_reserved_gb": get_gpu_peak_reserved_gb(),
        "hardware": get_device_name(),
        "notes": str(notes),
    }

    efficiency_records.append(row)

    print(
        f"[efficiency] {model} | {stage}: "
        f"{elapsed_seconds:.2f}s, "
        f"{row['seconds_per_1000_examples'] if not pd.isna(row['seconds_per_1000_examples']) else np.nan:.4f}s/1000 examples"
    )

    return result


# ------------------------------------------------------------
# Model size and parameter helpers
# ------------------------------------------------------------

def infer_model_family_for_efficiency(model_name, model_info=None):
    name = str(model_name).lower()

    if isinstance(model_info, dict):
        family = str(model_info.get("family", "")).lower()
        if family:
            return family

    if "biomedbert" in name or "scibert" in name or name.startswith("s9_") or "bert" in name:
        return "transformer"
    if "bilstm" in name:
        return "sequence_bilstm"
    if "crf" in name:
        return "sequence_crf"
    if "position" in name:
        return "position_aware_classical"
    if "svm" in name:
        return "classical_linear_svm"
    if "logreg" in name or "logistic" in name:
        return "classical_logistic_regression"
    if "mnb" in name or "naive" in name:
        return "classical_naive_bayes"
    return "unknown"


def extract_core_model_object_for_efficiency(model_info):
    if isinstance(model_info, dict):
        for key in ["model", "estimator", "pipeline", "clf"]:
            if key in model_info and model_info[key] is not None:
                return model_info[key]
        if "trainer" in model_info and model_info["trainer"] is not None:
            trainer_obj = model_info["trainer"]
            if hasattr(trainer_obj, "model"):
                return trainer_obj.model
            return trainer_obj
    return model_info


def count_torch_parameters_for_efficiency(model):
    if torch is None or not hasattr(model, "parameters"):
        return {
            "n_parameters_total": np.nan,
            "n_parameters_trainable": np.nan,
            "torch_parameter_size_mb": np.nan,
        }

    total = 0
    trainable = 0
    bytes_total = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        bytes_total += n * p.element_size()

    return {
        "n_parameters_total": int(total),
        "n_parameters_trainable": int(trainable),
        "torch_parameter_size_mb": bytes_total / (1024 ** 2),
    }


def count_sklearn_predictive_parameters_for_efficiency(model):
    """Count fitted predictive coefficient/intercept elements for classical models."""
    if model is None:
        return np.nan

    # Pipelines contain preprocessing state, but their trainable predictive
    # parameters belong to the final classifier step.
    if hasattr(model, "named_steps") and getattr(model, "named_steps", None):
        final_step = list(model.named_steps.values())[-1]
        return count_sklearn_predictive_parameters_for_efficiency(final_step)

    # PositionAwareTextClassifier stores its fitted logistic-regression head here.
    if hasattr(model, "clf_"):
        return count_sklearn_predictive_parameters_for_efficiency(model.clf_)

    # Preserve support for fitted multiclass wrappers such as OneVsOneClassifier.
    if hasattr(model, "estimators_"):
        component_counts = [
            count_sklearn_predictive_parameters_for_efficiency(estimator)
            for estimator in model.estimators_
        ]
        component_counts = [
            int(value) for value in component_counts
            if value is not None and np.isfinite(value) and int(value) > 0
        ]
        return int(sum(component_counts)) if component_counts else np.nan

    learned_elements = 0
    observed_array = False
    for attribute in ("coef_", "intercept_"):
        if not hasattr(model, attribute):
            continue
        value = getattr(model, attribute)
        try:
            n_elements = int(value.size)
        except Exception:
            n_elements = int(np.asarray(value).size)
        if n_elements > 0:
            learned_elements += n_elements
            observed_array = True
    return int(learned_elements) if observed_array else np.nan


def estimate_serialized_size_mb_for_efficiency(obj):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "model_size_estimate.pkl"
            if joblib is not None:
                joblib.dump(obj, tmp_path)
            else:
                with open(tmp_path, "wb") as f:
                    pickle.dump(obj, f)
            return tmp_path.stat().st_size / (1024 ** 2)
    except Exception:
        return np.nan


def get_model_introspection_for_efficiency(model_name, model_info):
    family = infer_model_family_for_efficiency(model_name, model_info)
    core_model = extract_core_model_object_for_efficiency(model_info)

    torch_stats = count_torch_parameters_for_efficiency(core_model)

    if not np.isnan(torch_stats["torch_parameter_size_mb"]):
        n_parameters_total = torch_stats["n_parameters_total"]
        n_parameters_trainable = torch_stats["n_parameters_trainable"]
        parameter_count_method = "torch_parameter_elements"
        estimated_size_mb = torch_stats["torch_parameter_size_mb"]
        size_method = "torch_parameter_bytes"
    else:
        classical_parameter_count = count_sklearn_predictive_parameters_for_efficiency(core_model)
        n_parameters_total = classical_parameter_count
        n_parameters_trainable = classical_parameter_count
        parameter_count_method = "fitted_predictor_coef_and_intercept_elements"
        estimated_size_mb = estimate_serialized_size_mb_for_efficiency(core_model)
        size_method = "serialized_joblib_or_pickle"

    return {
        "model": str(model_name),
        "model_family": family,
        "n_parameters_total": n_parameters_total,
        "n_parameters_trainable": n_parameters_trainable,
        "parameter_count_method": parameter_count_method,
        "estimated_model_size_mb": estimated_size_mb,
        "model_size_method": size_method,
    }

print("Computational efficiency helpers loaded.")

def load_pubmed_rct(path, split_name):
    """Load either the locked 3-column files or canonical header/2-column files."""
    rows = []
    current_pubmed_id = None
    expected_labels = {"background", "objective", "methods", "results", "conclusions"}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle):
            line = raw_line.rstrip("\r\n")
            if line.startswith("###"):
                current_pubmed_id = line[3:].strip()
                if not current_pubmed_id:
                    raise ValueError(f"{path}:{line_num + 1}: empty abstract identifier")
                continue
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            if len(parts) == 3:
                pubmed_id, label, text = parts
            elif len(parts) == 2 and current_pubmed_id is not None:
                label, text = parts
                pubmed_id = current_pubmed_id
            else:
                raise ValueError(
                    f"{path}:{line_num + 1}: expected PubMedID<TAB>label<TAB>text "
                    "or a canonical ###PubMedID header followed by label<TAB>text"
                )
            label = label.strip().lower()
            text = text.strip()
            if label not in expected_labels or not str(pubmed_id).strip() or not text:
                raise ValueError(f"{path}:{line_num + 1}: invalid identifier, label, or empty text")
            rows.append({
                "pubmed_id": str(pubmed_id).strip(),
                "label": label,
                "text": text,
                "split": split_name,
                # Preserve physical line order so item IDs match the frozen generator.
                "file_order": line_num,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No PubMed RCT sentences were parsed from {path}")
    if frame.duplicated(["pubmed_id", "file_order"]).any():
        raise ValueError(f"Duplicate PubMedID/file-order keys in {path}")
    return frame


train_df = load_pubmed_rct(train_path, "train")
dev_df = load_pubmed_rct(dev_path, "dev")
test_df = load_pubmed_rct(test_path, "test")

def _audit_raw_input_order(frame):
    within_abstract_failures = 0
    for _, group in frame.groupby("pubmed_id", sort=False):
        if not group["file_order"].is_monotonic_increasing:
            within_abstract_failures += 1
    abstract_blocks = int(frame["pubmed_id"].ne(frame["pubmed_id"].shift()).sum())
    repeated_abstract_blocks = max(0, abstract_blocks - int(frame["pubmed_id"].nunique()))
    return within_abstract_failures + repeated_abstract_blocks

RAW_INPUT_ORDER_AUDIT = {
    "train": _audit_raw_input_order(train_df),
    "development": _audit_raw_input_order(dev_df),
    "test": _audit_raw_input_order(test_df),
}

PUBLICATION_DATASET_PREFLIGHT = []
for split_name, frame in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
    expected = EXPECTED_PUBLICATION_DATA[split_name]
    observed = {
        "split": split_name,
        "rows": int(len(frame)),
        "abstracts": int(frame["pubmed_id"].nunique()),
    }
    observed["verified"] = (
        observed["rows"] == expected["rows"] and
        observed["abstracts"] == expected["abstracts"]
    )
    PUBLICATION_DATASET_PREFLIGHT.append(observed)

abstract_id_sets = {
    "train": set(train_df["pubmed_id"]),
    "dev": set(dev_df["pubmed_id"]),
    "test": set(test_df["pubmed_id"]),
}
PUBLICATION_SPLIT_OVERLAP = {
    "train_dev": len(abstract_id_sets["train"] & abstract_id_sets["dev"]),
    "train_test": len(abstract_id_sets["train"] & abstract_id_sets["test"]),
    "dev_test": len(abstract_id_sets["dev"] & abstract_id_sets["test"]),
}

if PUBLICATION_RUN:
    verified = all(row["verified"] for row in PUBLICATION_DATASET_PREFLIGHT)
    no_overlap = sum(PUBLICATION_SPLIT_OVERLAP.values()) == 0
    if not (verified and no_overlap):
        message = (
            "Locked PubMed RCT row/abstract/overlap preflight failed: "
            f"{PUBLICATION_DATASET_PREFLIGHT}; overlap={PUBLICATION_SPLIT_OVERLAP}"
        )
        if ALLOW_UNVERIFIED_DATA:
            warnings.warn(message + " (explicit PLOS_ALLOW_UNVERIFIED_DATA override active)")
        else:
            raise RuntimeError(message)

print("Loaded train/dev/test:", train_df.shape, dev_df.shape, test_df.shape)
print("Dataset preflight:", PUBLICATION_DATASET_PREFLIGHT)
print("Abstract-ID overlap:", PUBLICATION_SPLIT_OVERLAP)


def normalize_numbers(text: str) -> str:
    return re.sub(r"\d", "0", text)

def add_sentence_position_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["pubmed_id", "file_order"]).reset_index(drop=True)
    df["sentence_index_in_abstract"] = df.groupby("pubmed_id").cumcount()
    df["abstract_length"] = df.groupby("pubmed_id")["pubmed_id"].transform("size")
    df["relative_position"] = np.where(
        df["abstract_length"] > 1,
        df["sentence_index_in_abstract"] / (df["abstract_length"] - 1),
        0.0
    )
    df["is_first_sentence"] = (df["sentence_index_in_abstract"] == 0).astype(int)
    df["is_last_sentence"] = (df["sentence_index_in_abstract"] == (df["abstract_length"] - 1)).astype(int)
    df["text_num_norm"] = df["text"].astype(str).apply(normalize_numbers)
    return df

train_df = add_sentence_position_features(train_df)
dev_df   = add_sentence_position_features(dev_df)
test_df  = add_sentence_position_features(test_df)

train_df.head()

LABEL_ORDER = ["background", "objective", "methods", "results", "conclusions"]


def label_distribution_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = (
        df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        .rename_axis("label").reset_index(name=f"{name}_count")
    )
    out[f"{name}_pct"] = (out[f"{name}_count"] / max(len(df), 1) * 100).round(2)
    return out


# Pre-lock inspection is restricted to training and development outcomes.
label_dist = label_distribution_table(train_df, "train")
label_dist = label_dist.merge(label_distribution_table(dev_df, "development"), on="label", how="outer")
label_dist

def abstract_stats(df: pd.DataFrame, split_name: str):
    abs_lengths = df.groupby("pubmed_id").size()
    print(f"\n{split_name.upper()} split")
    print(f"Sentences: {len(df):,}")
    print(f"Abstracts: {df['pubmed_id'].nunique():,}")
    print(f"Average sentences/abstract: {abs_lengths.mean():.2f}")
    print(f"Median sentences/abstract: {abs_lengths.median():.2f}")
    print(f"Min/Max sentences/abstract: {abs_lengths.min()} / {abs_lengths.max()}")


# Test-label summaries are intentionally deferred until after final-system locking.
abstract_stats(train_df, "train")
abstract_stats(dev_df, "development")


for label in LABEL_ORDER:
    subset = train_df[train_df["label"] == label]
    print("\n" + "="*80)
    print(f"Label: {label}")
    for ex in subset["text"].head(3).tolist():
        print("-", ex)


label_encoder = LabelEncoder()
label_encoder.fit(LABEL_ORDER)

train_df["label_id"] = label_encoder.transform(train_df["label"])
dev_df["label_id"]   = label_encoder.transform(dev_df["label"])

ID2LABEL = dict(enumerate(label_encoder.classes_))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
print("Label mapping:", LABEL2ID)

def sample_debug_subset_by_abstract(df, frac, random_state=42, max_rows=None):
    """Select complete abstracts for a nonpublication integration run."""
    if frac >= 1.0 and max_rows is None:
        return df.copy().sort_values(["pubmed_id", "file_order"]).reset_index(drop=True)
    abstract_ids = pd.Series(df["pubmed_id"].astype(str).unique())
    n_abstracts = max(1, int(round(len(abstract_ids) * min(frac, 1.0))))
    sampled_ids = abstract_ids.sample(n=n_abstracts, random_state=random_state).tolist()
    rng = np.random.default_rng(random_state)
    rng.shuffle(sampled_ids)
    if max_rows is not None:
        lengths = df.groupby(df["pubmed_id"].astype(str)).size().to_dict()
        kept, running = [], 0
        for abstract_id in sampled_ids:
            length = int(lengths[abstract_id])
            if kept and running + length > max_rows:
                continue
            kept.append(abstract_id)
            running += length
            if running >= max_rows:
                break
        sampled_ids = kept
    out = df[df["pubmed_id"].astype(str).isin(sampled_ids)].copy()
    return out.sort_values(["pubmed_id", "file_order"]).reset_index(drop=True)


def build_working_splits(train_frame, dev_frame, test_frame, debug_small_mode=True):
    if debug_small_mode:
        train_work = sample_debug_subset_by_abstract(
            train_frame, TRAIN_FRAC, SEED, MAX_TRANSFORMER_TRAIN
        )
        dev_work = sample_debug_subset_by_abstract(
            dev_frame, DEV_FRAC, SEED + 1, MAX_TRANSFORMER_DEV
        )
        test_work = sample_debug_subset_by_abstract(
            test_frame, TEST_FRAC, SEED + 2, MAX_TRANSFORMER_TEST
        )
    else:
        train_work = train_frame.copy()
        dev_work = dev_frame.copy()
        test_work = test_frame.copy()
    return {
        "train_sent": train_work.reset_index(drop=True),
        "dev_sent": dev_work.reset_index(drop=True),
        "test_sent": test_work.reset_index(drop=True),
        "train_seq": train_work.reset_index(drop=True),
        "dev_seq": dev_work.reset_index(drop=True),
        "test_seq": test_work.reset_index(drop=True),
    }


splits = build_working_splits(train_df, dev_df, test_df, DEBUG_SMALL_MODE)
for split_key, frame in splits.items():
    print(split_key, frame.shape, "abstracts=", frame["pubmed_id"].nunique())


def compute_metrics(y_true, y_pred, labels=LABEL_ORDER) -> Dict:
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    for label in labels:
        metrics[f"{label}_precision"] = report[label]["precision"]
        metrics[f"{label}_recall"] = report[label]["recall"]
        metrics[f"{label}_f1"] = report[label]["f1-score"]
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.show()

results_store = []

def record_result(model_name: str, split_name: str, y_true, y_pred, meta: Optional[Dict] = None):
    row = {"model_name": model_name, "split_name": split_name}
    row.update(compute_metrics(y_true, y_pred))
    if meta is not None:
        row.update(meta)
    results_store.append(row)
    return row

def summarize_results_table(results_store):
    if len(results_store) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(results_store)
    # Development evidence is restricted to prespecified aggregate selection metrics.
    # Classwise development results are neither displayed nor exported.
    cols = ["model_name", "split_name", "accuracy", "macro_f1", "weighted_f1"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["split_name", "macro_f1"], ascending=[True, False]).reset_index(drop=True)


def train_eval_text_model(model_name, pipeline, train_df, dev_df, text_col="text"):
    X_train = train_df[text_col]
    y_train = train_df["label"]
    X_dev = dev_df[text_col]
    y_dev = dev_df["label"]

    pipeline.fit(X_train, y_train)
    pred_dev = pipeline.predict(X_dev)

    row = record_result(
        model_name=model_name,
        split_name="dev",
        y_true=y_dev.tolist(),
        y_pred=pred_dev.tolist(),
        meta={"uses_sequence": False, "text_column": text_col}
    )
    print(f"Finished: {model_name}")
    print(pd.Series({
        "accuracy": row["accuracy"],
        "macro_f1": row["macro_f1"],
        "background_f1": row["background_f1"],
        "objective_f1": row["objective_f1"]
    }))
    return pipeline, pred_dev

def train_eval_feature_model(model_name, X_train, y_train, X_dev, y_dev, estimator):
    estimator.fit(X_train, y_train)
    pred_dev = estimator.predict(X_dev)
    row = record_result(
        model_name=model_name,
        split_name="dev",
        y_true=y_dev.tolist(),
        y_pred=pred_dev.tolist(),
        meta={"uses_sequence": False}
    )
    print(f"Finished: {model_name}")
    print(pd.Series({
        "accuracy": row["accuracy"],
        "macro_f1": row["macro_f1"],
        "background_f1": row["background_f1"],
        "objective_f1": row["objective_f1"]
    }))
    return estimator, pred_dev

# ============================================================
# TRAINING-ONLY 5 x 4 ABSTRACT-GROUPED NESTED CV
# ============================================================
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from joblib import Memory, parallel_backend

CLASSICAL_PIPELINE_MEMORY = Memory(
    location=str(LOCAL_RUNTIME_CACHE_ROOT / "sklearn_pipeline_cache"), verbose=0
)

cv_results_store = []
cv_fold_predictions = {}


class PositionAwareTextClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        text_col="text",
        pos_cols=("sentence_index_in_abstract", "relative_position", "is_first_sentence", "is_last_sentence"),
        ngram_range=(1, 2),
        min_df=1,
        C=1.0,
        class_weight=None,
        max_iter=2000,
        tol=1e-4,
        random_state=42,
    ):
        self.text_col = text_col
        self.pos_cols = pos_cols
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _transform(self, X):
        X_text = self.vectorizer_.transform(X[self.text_col])
        X_pos = self.scaler_.transform(X[list(self.pos_cols)])
        return hstack([X_text, csr_matrix(X_pos)])

    def fit(self, X, y):
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )
        X_text = self.vectorizer_.fit_transform(X[self.text_col])
        self.scaler_ = StandardScaler()
        X_pos = self.scaler_.fit_transform(X[list(self.pos_cols)])
        X_all = hstack([X_text, csr_matrix(X_pos)])
        self.clf_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            tol=self.tol,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.clf_.fit(X_all, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self._transform(X))

    def decision_function(self, X):
        return self.clf_.decision_function(self._transform(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._transform(X))


def _text_column_transformer(text_col, min_df=1):
    return ColumnTransformer(
        [("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=min_df, lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b", norm="l2",
            use_idf=True, smooth_idf=True, sublinear_tf=False,
        ), text_col)],
        remainder="drop",
    )


def _logistic_estimator():
    return LogisticRegression(
        max_iter=2000, tol=1e-4, solver="lbfgs", random_state=SEED
    )


def _linear_svc_estimator():
    return LinearSVC(
        C=1.0, penalty="l2", loss="squared_hinge", dual="auto",
        tol=1e-4, max_iter=5000, multi_class="ovr", random_state=SEED,
    )


def _cached_text_pipeline(steps):
    return Pipeline(steps, memory=CLASSICAL_PIPELINE_MEMORY)


def get_classical_model_specs():
    return {
        "S0_TFIDF_LogReg": {
            "estimator": _cached_text_pipeline([
                ("text_preprocessing", _text_column_transformer("text")),
                ("clf", _logistic_estimator()),
            ]),
            "param_grid": {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
            "text_col": "text", "family": "classical_text",
        },
        "S1_TFIDF_LinearSVM": {
            "estimator": _cached_text_pipeline([
                ("text_preprocessing", _text_column_transformer("text")),
                ("clf", _linear_svc_estimator()),
            ]),
            "param_grid": {"clf__C": [0.5, 1.0, 2.0]},
            "text_col": "text", "family": "classical_text",
        },
        "S1b_TFIDF_LinearSVM_OVO": {
            "estimator": _cached_text_pipeline([
                ("text_preprocessing", _text_column_transformer("text")),
                ("clf", OneVsOneClassifier(_linear_svc_estimator(), n_jobs=CLASSICAL_INNER_ESTIMATOR_N_JOBS)),
            ]),
            "param_grid": {"clf__estimator__C": [0.5, 1.0, 2.0]},
            "text_col": "text", "family": "classical_text_sensitivity",
        },
        "S2_TFIDF_MNB": {
            "estimator": _cached_text_pipeline([
                ("text_preprocessing", _text_column_transformer("text")),
                ("clf", MultinomialNB()),
            ]),
            "param_grid": {"clf__alpha": [0.1, 0.5, 1.0]},
            "text_col": "text", "family": "classical_text",
        },
        "S5_NumberNormalized_LogReg": {
            "estimator": _cached_text_pipeline([
                ("text_preprocessing", _text_column_transformer("text_num_norm")),
                ("clf", _logistic_estimator()),
            ]),
            "param_grid": {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
            "text_col": "text_num_norm", "family": "classical_text",
        },
        "S6_PositionAware_LogReg": {
            "estimator": PositionAwareTextClassifier(text_col="text", random_state=SEED),
            "param_grid": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            "text_col": "text", "family": "classical_position",
        },
    }


CLASSICAL_MODEL_SPECS = get_classical_model_specs()
OUTER_GROUP_CV = StratifiedGroupKFold(
    n_splits=N_FOLDS_CV, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE if CV_SHUFFLE else None
)
INNER_GROUP_CV = StratifiedGroupKFold(
    n_splits=N_INNER_FOLDS_CV, shuffle=CV_SHUFFLE, random_state=(CV_RANDOM_STATE + 1) if CV_SHUFFLE else None
)


def grouped_grid_search(estimator, param_grid, X, y, groups, cv_splitter=INNER_GROUP_CV):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv_splitter,
        n_jobs=CLASSICAL_SEARCH_N_JOBS,
        pre_dispatch=CLASSICAL_SEARCH_N_JOBS,
        refit=True,
        verbose=0,
        error_score="raise",
        return_train_score=False,
    )
    with parallel_backend("loky", inner_max_num_threads=1):
        search.fit(X, y, groups=groups)
    return search


def nested_grouped_cv(model_name, spec, X_df, y, groups):
    fold_rows, fold_predictions, fold_params = [], [], []
    for fold, (train_idx, valid_idx) in enumerate(
        OUTER_GROUP_CV.split(X_df, y, groups=groups), start=1
    ):
        X_train = X_df.iloc[train_idx].reset_index(drop=True)
        X_valid = X_df.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)
        g_train = groups.iloc[train_idx].reset_index(drop=True)
        g_valid = groups.iloc[valid_idx].reset_index(drop=True)
        if set(g_train).intersection(set(g_valid)):
            raise AssertionError(f"Abstract leakage in outer fold {fold}")
        search = grouped_grid_search(
            clone(spec["estimator"]), spec["param_grid"], X_train, y_train, g_train
        )
        pred = search.best_estimator_.predict(X_valid)
        row = {
            "model_name": model_name, "fold": fold,
            "n_abstracts": int(g_valid.nunique()),
            "n_sentences": int(len(valid_idx)),
            "inner_selection_metric": "f1_macro",
        }
        row.update(compute_metrics(y_valid.tolist(), pred.tolist()))
        fold_rows.append(row)
        fold_params.append({
            "model_name": model_name, "fold": fold,
            "inner_best_macro_f1": float(search.best_score_),
            "inner_best_params_json": json.dumps(search.best_params_, sort_keys=True),
        })
        fold_predictions.append(pd.DataFrame({
            "model_name": model_name, "fold": fold,
            "pubmed_id": g_valid.astype(str), "row_index": valid_idx,
            "true": y_valid.astype(str), "pred": np.asarray(pred).astype(str),
        }))
        del search
    return (
        pd.DataFrame(fold_rows),
        pd.concat(fold_predictions, ignore_index=True),
        pd.DataFrame(fold_params),
    )

if any(row.get("model_name") in CLASSICAL_MODEL_SPECS for row in results_store):
    raise RuntimeError("Classical development rows already exist; restart the kernel before rerunning selection.")

classical_cv_results = {}
classical_best_estimators = {}
classical_best_params = {}
classical_nested_params = {}
classical_dev_sanity_rows = []
classical_dev_predictions = {}
classical_full_search_rows = []

classical_train_df = splits["train_sent"].copy().reset_index(drop=True)
classical_dev_df = splits["dev_sent"].copy().reset_index(drop=True)
y_classical_train = classical_train_df["label"].copy().reset_index(drop=True)
classical_groups = classical_train_df["pubmed_id"].astype(str).reset_index(drop=True)

for model_name, spec in CLASSICAL_MODEL_SPECS.items():
    print("=" * 90)
    print("Grouped nested CV:", model_name)
    fold_df, pred_df, nested_param_df = nested_grouped_cv(
        model_name, spec, classical_train_df, y_classical_train, classical_groups
    )
    classical_cv_results[model_name] = fold_df
    cv_fold_predictions[model_name] = pred_df
    classical_nested_params[model_name] = nested_param_df

    full_search = grouped_grid_search(
        clone(spec["estimator"]), spec["param_grid"],
        classical_train_df, y_classical_train, classical_groups,
    )
    classical_best_estimators[model_name] = clone(full_search.best_estimator_)
    classical_best_params[model_name] = full_search.best_params_
    classical_full_search_rows.append({
        "model_name": model_name,
        "selection_split": "training_only",
        "cv": f"{N_INNER_FOLDS_CV}-fold StratifiedGroupKFold",
            "cv_role": "full-training hyperparameter search; not an unbiased performance estimate",
        "selection_metric": "f1_macro",
        "best_cv_macro_f1": float(full_search.best_score_),
        "best_params_json": json.dumps(full_search.best_params_, sort_keys=True),
        "param_grid_json": json.dumps(spec["param_grid"], sort_keys=True),
        "n_jobs": CLASSICAL_SEARCH_N_JOBS,
    })

    # GridSearchCV refit already fitted the selected estimator on all training rows.
    dev_estimator = full_search.best_estimator_
    dev_pred = dev_estimator.predict(classical_dev_df)
    classical_dev_sanity_rows.append(record_result(
        model_name=model_name,
        split_name="development_diagnostic_after_training_only_selection",
        y_true=classical_dev_df["label"].tolist(),
        y_pred=np.asarray(dev_pred).tolist(),
        meta={"selection_basis": "training_only_grouped_nested_cv"},
    ))
    classical_dev_predictions[model_name] = np.asarray(dev_pred)
    print("Full-training grouped-search parameters:", full_search.best_params_)
    print(f"Outer grouped CV macro-F1: {fold_df['macro_f1'].mean():.4f} +/- {fold_df['macro_f1'].std(ddof=1):.4f}")
    del full_search

cv_results_store = []
for model_name, fold_df in classical_cv_results.items():
    row = {"model_name": model_name, "n_outer_folds": int(len(fold_df))}
    excluded = {"model_name", "fold", "n_abstracts", "n_sentences", "inner_selection_metric"}
    for metric in [column for column in fold_df.columns if column not in excluded]:
        if pd.api.types.is_numeric_dtype(fold_df[metric]):
            row[f"{metric}_mean"] = fold_df[metric].mean()
            row[f"{metric}_std"] = fold_df[metric].std(ddof=1)
    cv_results_store.append(row)

classical_cv_summary = (
    pd.DataFrame(cv_results_store)
    .sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
)
classical_full_search_best_params_df = pd.DataFrame(classical_full_search_rows)
display(classical_cv_summary)
display(classical_full_search_best_params_df)

# ============================================================
# DURABLE GROUPED-CV CHECKPOINT
# ============================================================
# Persist the complete training-only nested-CV evidence immediately after it
# is produced. The final publication gate still revalidates the in-memory
# objects; these files are restart-safe evidence, not an assertion bypass.
ANALYSIS_STAGE_DIR = Path(OUTPUT_DIR) / "plos_revision_analysis_checkpoints"
ANALYSIS_STAGE_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_stage_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"Failed to stage non-empty CSV: {temporary}")
    os.replace(temporary, path)
    return path


def _atomic_stage_json(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"Failed to stage non-empty JSON: {temporary}")
    os.replace(temporary, path)
    return path


def _stage_artifact_record(path, row_count=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Staged artifact is missing or empty: {path}")
    return {
        "file": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
        "row_count": None if row_count is None else int(row_count),
    }


expected_classical_models = set(CLASSICAL_MODEL_SPECS)
if (
    set(classical_cv_results) != expected_classical_models
    or set(cv_fold_predictions) != expected_classical_models
    or set(classical_nested_params) != expected_classical_models
):
    raise RuntimeError("Grouped-CV checkpoint coverage differs from the model specification.")
if set(classical_cv_summary["model_name"]) != expected_classical_models:
    raise RuntimeError("Grouped-CV summary omits one or more classical candidates.")
if set(classical_full_search_best_params_df["model_name"]) != expected_classical_models:
    raise RuntimeError("Full-training grouped searches omit one or more candidates.")

grouped_cv_stage_artifacts = []
grouped_cv_stage_artifacts.append(_stage_artifact_record(
    _atomic_stage_csv(
        classical_cv_summary,
        ANALYSIS_STAGE_DIR / "classical_nested_cv_summary.csv",
    ),
    len(classical_cv_summary),
))
grouped_cv_stage_artifacts.append(_stage_artifact_record(
    _atomic_stage_csv(
        classical_full_search_best_params_df,
        ANALYSIS_STAGE_DIR / "full_training_grouped_search_best_params.csv",
    ),
    len(classical_full_search_best_params_df),
))

for model_name in sorted(expected_classical_models):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    fold_frame = classical_cv_results[model_name].copy()
    oof_frame = cv_fold_predictions[model_name].copy()
    parameter_frame = classical_nested_params[model_name].copy()
    if (
        len(fold_frame) != N_FOLDS_CV
        or set(fold_frame["fold"]) != set(range(1, N_FOLDS_CV + 1))
    ):
        raise RuntimeError(f"Incomplete outer grouped-CV folds for {model_name}.")
    if (
        len(oof_frame) != len(classical_train_df)
        or oof_frame["row_index"].duplicated().any()
        or set(oof_frame["row_index"]) != set(range(len(classical_train_df)))
    ):
        raise RuntimeError(f"Incomplete one-to-one OOF coverage for {model_name}.")
    if (
        oof_frame["pubmed_id"].nunique() != classical_groups.nunique()
        or not set(oof_frame["true"]).issubset(set(LABEL_ORDER))
        or not set(oof_frame["pred"]).issubset(set(LABEL_ORDER))
    ):
        raise RuntimeError(f"Invalid grouped-CV labels or abstract coverage for {model_name}.")
    if (
        len(parameter_frame) != N_FOLDS_CV
        or set(parameter_frame["fold"]) != set(range(1, N_FOLDS_CV + 1))
    ):
        raise RuntimeError(f"Incomplete inner-search parameter record for {model_name}.")

    for filename, frame in [
        (f"nested_grouped_cv_fold_metrics__{safe}.csv", fold_frame),
        (f"nested_grouped_cv_oof_predictions__{safe}.csv", oof_frame),
        (f"nested_grouped_cv_inner_best_params__{safe}.csv", parameter_frame),
    ]:
        grouped_cv_stage_artifacts.append(_stage_artifact_record(
            _atomic_stage_csv(frame, ANALYSIS_STAGE_DIR / filename),
            len(frame),
        ))

grouped_cv_stage_contract = {
    "contract_version": "plos_analysis_stage_v1",
    "stage": "grouped_nested_cv_complete",
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "n_models": len(expected_classical_models),
    "n_outer_folds": N_FOLDS_CV,
    "n_inner_folds": N_INNER_FOLDS_CV,
    "n_training_sentences": len(classical_train_df),
    "n_training_abstracts": int(classical_groups.nunique()),
    "artifacts": grouped_cv_stage_artifacts,
}
_atomic_stage_json(
    grouped_cv_stage_contract,
    ANALYSIS_STAGE_DIR / "GROUPED_CV_STAGE_COMPLETE.json",
)
print(
    "Durable grouped-CV checkpoint:",
    ANALYSIS_STAGE_DIR / "GROUPED_CV_STAGE_COMPLETE.json",
)


if len(classical_cv_summary) > 0:
    display(
        classical_cv_summary[[
            "model_name",
            "macro_f1_mean", "macro_f1_std",
            "accuracy_mean", "accuracy_std",
            "background_f1_mean", "objective_f1_mean"
        ]]
    )

    plt.figure(figsize=(10, 5))
    plt.bar(classical_cv_summary["model_name"], classical_cv_summary["macro_f1_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean CV macro-F1")
    plt.title("Classical models: nested abstract-grouped outer-fold Macro-F1")
    plt.tight_layout()
    plt.show()

# ============================================================
# PRESPECIFIED FIXED-HYPERPARAMETER GROUPED OOF ABLATIONS
# ============================================================
from sklearn.metrics import confusion_matrix

ABLATION_BOOTSTRAP_N = 10000
ABLATION_BOOTSTRAP_BATCH = 256
FIXED_ABLATION_C = 1.0
FIXED_ABLATION_WEIGHT = None
FIXED_ABLATION_SOURCE = "prespecified before model-selection outputs; not chosen from search results"


def _macro_f1_from_confusion_batch(confusions):
    confusions = np.asarray(confusions, dtype=float)
    tp = np.diagonal(confusions, axis1=-2, axis2=-1)
    fp = confusions.sum(axis=-2) - tp
    fn = confusions.sum(axis=-1) - tp
    denominator = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    return per_class.mean(axis=-1)


def _preaggregate_abstract_confusions(frame):
    matrices_a, matrices_b = [], []
    for _, group in frame.groupby("pubmed_id", sort=False):
        matrices_a.append(confusion_matrix(group["true"], group["pred_a"], labels=LABEL_ORDER))
        matrices_b.append(confusion_matrix(group["true"], group["pred_b"], labels=LABEL_ORDER))
    return np.asarray(matrices_a, dtype=np.int32), np.asarray(matrices_b, dtype=np.int32)


def abstract_cluster_bootstrap_delta(frame, n_bootstrap=10000, seed=42):
    cm_a, cm_b = _preaggregate_abstract_confusions(frame)
    n_abstracts = len(cm_a)
    probability = np.full(n_abstracts, 1.0 / n_abstracts)
    flat_a = cm_a.reshape(n_abstracts, -1)
    flat_b = cm_b.reshape(n_abstracts, -1)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, ABLATION_BOOTSTRAP_BATCH):
        stop = min(start + ABLATION_BOOTSTRAP_BATCH, n_bootstrap)
        weights = rng.multinomial(n_abstracts, probability, size=stop - start)
        aggregate_a = (weights @ flat_a).reshape(-1, len(LABEL_ORDER), len(LABEL_ORDER))
        aggregate_b = (weights @ flat_b).reshape(-1, len(LABEL_ORDER), len(LABEL_ORDER))
        deltas[start:stop] = (
            _macro_f1_from_confusion_batch(aggregate_b)
            - _macro_f1_from_confusion_batch(aggregate_a)
        )
    return deltas


def _fixed_grouped_oof(model_name, estimator):
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(
        OUTER_GROUP_CV.split(classical_train_df, y_classical_train, groups=classical_groups), start=1
    ):
        fitted = clone(estimator)
        fitted.fit(classical_train_df.iloc[train_idx], y_classical_train.iloc[train_idx])
        prediction = fitted.predict(classical_train_df.iloc[valid_idx])
        rows.append(pd.DataFrame({
            "model_name": model_name, "fold": fold, "row_index": valid_idx,
            "pubmed_id": classical_groups.iloc[valid_idx].astype(str).to_numpy(),
            "true": y_classical_train.iloc[valid_idx].astype(str).to_numpy(),
            "pred": np.asarray(prediction).astype(str),
        }))
        del fitted
    output = pd.concat(rows, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if len(output) != len(classical_train_df) or output["row_index"].duplicated().any():
        raise AssertionError("Fixed grouped OOF predictions do not cover every row exactly once.")
    return output


def _text_logreg(text_col, C, class_weight):
    return _cached_text_pipeline([
        ("text_preprocessing", _text_column_transformer(text_col)),
        ("clf", LogisticRegression(
            C=C, class_weight=class_weight, max_iter=2000, tol=1e-4,
            solver="lbfgs", random_state=SEED,
        )),
    ])


fixed_ablation_predictions = {
    "lexical": _fixed_grouped_oof(
        "fixed_lexical", _text_logreg("text", FIXED_ABLATION_C, FIXED_ABLATION_WEIGHT)
    ),
    "number_normalised": _fixed_grouped_oof(
        "fixed_number_normalised",
        _text_logreg("text_num_norm", FIXED_ABLATION_C, FIXED_ABLATION_WEIGHT),
    ),
    "position": _fixed_grouped_oof(
        "fixed_position",
        PositionAwareTextClassifier(
            C=FIXED_ABLATION_C, class_weight=FIXED_ABLATION_WEIGHT,
            random_state=SEED,
        ),
    ),
}


def _align_fixed(a, b):
    keys = ["row_index", "pubmed_id", "true"]
    return a[keys + ["pred"]].rename(columns={"pred": "pred_a"}).merge(
        b[keys + ["pred"]].rename(columns={"pred": "pred_b"}),
        on=keys, validate="one_to_one",
    )


ablation_rows = []
grouped_ablation_distribution_parts = []
for offset, (name_b, feature_change) in enumerate([
    ("number_normalised", "digit normalisation"),
    ("position", "position features"),
]):
    aligned = _align_fixed(fixed_ablation_predictions["lexical"], fixed_ablation_predictions[name_b])
    bootstrap_seed = SEED + offset
    distribution = abstract_cluster_bootstrap_delta(
        aligned, n_bootstrap=ABLATION_BOOTSTRAP_N, seed=bootstrap_seed
    )
    grouped_ablation_distribution_parts.append(pd.DataFrame({
        "feature_change": feature_change,
        "replicate": np.arange(len(distribution)),
        "delta_macro_f1_changed_minus_lexical": distribution,
        "bootstrap_seed": bootstrap_seed,
    }))
    ablation_rows.append({
        "feature_change": feature_change,
        "fixed_C": FIXED_ABLATION_C,
        "fixed_class_weight": str(FIXED_ABLATION_WEIGHT),
        "fixed_hyperparameter_source": FIXED_ABLATION_SOURCE,
        "delta_macro_f1_changed_minus_lexical": (
            f1_score(aligned["true"], aligned["pred_b"], average="macro", zero_division=0)
            - f1_score(aligned["true"], aligned["pred_a"], average="macro", zero_division=0)
        ),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_proportion_delta_gt_0": float(np.mean(distribution > 0)),
        "ci_method": "95% percentile complete-abstract cluster bootstrap",
        "analysis_role": "prespecified controlled ablation",
        "n_abstracts": int(aligned["pubmed_id"].nunique()),
        "n_cluster_bootstrap": ABLATION_BOOTSTRAP_N,
        "bootstrap_seed": bootstrap_seed,
        "reference_seed": REFERENCE_SEED,
    })

grouped_ablation_results = pd.DataFrame(ablation_rows)
grouped_ablation_bootstrap_distributions_df = pd.concat(
    grouped_ablation_distribution_parts, ignore_index=True
)
display(grouped_ablation_results)

# Reuse the two unweighted OOF cells already fitted above; only the two
# balanced cells require additional fits. The estimands and fold assignments are unchanged.
position_weight_factorial = {
    "position_0__weight_none": fixed_ablation_predictions["lexical"],
    "position_1__weight_none": fixed_ablation_predictions["position"],
}
for use_position in [False, True]:
    class_weight = "balanced"
    name = f"position_{int(use_position)}__weight_balanced"
    estimator = (
        PositionAwareTextClassifier(
            C=FIXED_ABLATION_C, class_weight=class_weight, random_state=SEED
        )
        if use_position else _text_logreg("text", FIXED_ABLATION_C, class_weight)
    )
    position_weight_factorial[name] = _fixed_grouped_oof(name, estimator)

FACTORIAL_CELL_ORDER = [
    "position_0__weight_none",
    "position_1__weight_none",
    "position_0__weight_balanced",
    "position_1__weight_balanced",
]
FACTORIAL_PRED_COLUMNS = {
    "position_0__weight_none": "pred_lexical_unweighted",
    "position_1__weight_none": "pred_position_unweighted",
    "position_0__weight_balanced": "pred_lexical_balanced",
    "position_1__weight_balanced": "pred_position_balanced",
}


def _align_factorial_predictions(prediction_frames):
    keys = ["row_index", "pubmed_id", "true"]
    aligned = None
    for cell_name in FACTORIAL_CELL_ORDER:
        part = prediction_frames[cell_name][keys + ["pred"]].rename(
            columns={"pred": FACTORIAL_PRED_COLUMNS[cell_name]}
        )
        aligned = part if aligned is None else aligned.merge(
            part, on=keys, validate="one_to_one"
        )
    if len(aligned) != len(classical_train_df) or not aligned["row_index"].is_unique:
        raise AssertionError("Factorial OOF predictions are not complete and one-to-one.")
    return aligned.sort_values("row_index").reset_index(drop=True)


factorial_aligned_df = _align_factorial_predictions(position_weight_factorial)
factorial_cell_rows = []
for cell_name in FACTORIAL_CELL_ORDER:
    pred_col = FACTORIAL_PRED_COLUMNS[cell_name]
    use_position = int("position_1" in cell_name)
    class_weight = "balanced" if "balanced" in cell_name else "none"
    factorial_cell_rows.append({
        "cell": cell_name,
        "position_features": bool(use_position),
        "class_weight": class_weight,
        "macro_f1": f1_score(
            factorial_aligned_df["true"], factorial_aligned_df[pred_col],
            labels=LABEL_ORDER, average="macro", zero_division=0,
        ),
        "fixed_C": FIXED_ABLATION_C,
        "n_sentences": int(len(factorial_aligned_df)),
        "n_abstracts": int(factorial_aligned_df["pubmed_id"].nunique()),
        "reference_seed": REFERENCE_SEED,
    })
position_weight_factorial_cells_df = pd.DataFrame(factorial_cell_rows)


def _factorial_cluster_bootstrap(frame, n_bootstrap=10000, seed=62):
    score_by_cell = {
        cell_name: np.empty(n_bootstrap, dtype=float)
        for cell_name in FACTORIAL_CELL_ORDER
    }
    matrices = {cell_name: [] for cell_name in FACTORIAL_CELL_ORDER}
    for _, group in frame.groupby("pubmed_id", sort=False):
        for cell_name in FACTORIAL_CELL_ORDER:
            matrices[cell_name].append(confusion_matrix(
                group["true"], group[FACTORIAL_PRED_COLUMNS[cell_name]],
                labels=LABEL_ORDER,
            ))
    flat = {
        name: np.asarray(values, dtype=np.int32).reshape(len(values), -1)
        for name, values in matrices.items()
    }
    n_abstracts = len(next(iter(flat.values())))
    probability = np.full(n_abstracts, 1.0 / n_abstracts)
    rng = np.random.default_rng(seed)
    for start in range(0, n_bootstrap, ABLATION_BOOTSTRAP_BATCH):
        stop = min(start + ABLATION_BOOTSTRAP_BATCH, n_bootstrap)
        weights = rng.multinomial(n_abstracts, probability, size=stop - start)
        for cell_name in FACTORIAL_CELL_ORDER:
            aggregated = (weights @ flat[cell_name]).reshape(
                -1, len(LABEL_ORDER), len(LABEL_ORDER)
            )
            score_by_cell[cell_name][start:stop] = _macro_f1_from_confusion_batch(aggregated)
    return score_by_cell


FACTORIAL_BOOTSTRAP_SEED = SEED + 20
factorial_bootstrap_scores = _factorial_cluster_bootstrap(
    factorial_aligned_df,
    n_bootstrap=ABLATION_BOOTSTRAP_N,
    seed=FACTORIAL_BOOTSTRAP_SEED,
)

ln = factorial_bootstrap_scores["position_0__weight_none"]
pn = factorial_bootstrap_scores["position_1__weight_none"]
lb = factorial_bootstrap_scores["position_0__weight_balanced"]
pb = factorial_bootstrap_scores["position_1__weight_balanced"]
factorial_contrast_distributions = {
    "position_effect_unweighted": pn - ln,
    "position_effect_balanced": pb - lb,
    "weight_effect_lexical": lb - ln,
    "weight_effect_position": pb - pn,
    "position_main_effect_average": 0.5 * ((pn - ln) + (pb - lb)),
    "weight_main_effect_average": 0.5 * ((lb - ln) + (pb - pn)),
    "position_by_weight_interaction": (pb - lb) - (pn - ln),
}

observed_cell_scores = position_weight_factorial_cells_df.set_index("cell")["macro_f1"]
ln_o = observed_cell_scores["position_0__weight_none"]
pn_o = observed_cell_scores["position_1__weight_none"]
lb_o = observed_cell_scores["position_0__weight_balanced"]
pb_o = observed_cell_scores["position_1__weight_balanced"]
factorial_observed_contrasts = {
    "position_effect_unweighted": pn_o - ln_o,
    "position_effect_balanced": pb_o - lb_o,
    "weight_effect_lexical": lb_o - ln_o,
    "weight_effect_position": pb_o - pn_o,
    "position_main_effect_average": 0.5 * ((pn_o - ln_o) + (pb_o - lb_o)),
    "weight_main_effect_average": 0.5 * ((lb_o - ln_o) + (pb_o - pn_o)),
    "position_by_weight_interaction": (pb_o - lb_o) - (pn_o - ln_o),
}

factorial_contrast_rows = []
factorial_distribution_parts = []
for contrast, distribution in factorial_contrast_distributions.items():
    factorial_contrast_rows.append({
        "contrast": contrast,
        "delta_macro_f1": factorial_observed_contrasts[contrast],
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_proportion_delta_gt_0": float(np.mean(distribution > 0)),
        "ci_method": "95% percentile complete-abstract cluster bootstrap",
        "fixed_C": FIXED_ABLATION_C,
        "n_sentences": int(len(factorial_aligned_df)),
        "n_abstracts": int(factorial_aligned_df["pubmed_id"].nunique()),
        "n_cluster_bootstrap": ABLATION_BOOTSTRAP_N,
        "bootstrap_seed": FACTORIAL_BOOTSTRAP_SEED,
        "reference_seed": REFERENCE_SEED,
        "analysis_role": "prespecified 2 x 2 factorial contrast",
    })
    factorial_distribution_parts.append(pd.DataFrame({
        "contrast": contrast,
        "replicate": np.arange(len(distribution)),
        "delta_macro_f1": distribution,
        "bootstrap_seed": FACTORIAL_BOOTSTRAP_SEED,
    }))

position_weight_factorial_results = pd.DataFrame(factorial_contrast_rows)
position_weight_factorial_bootstrap_distributions_df = pd.concat(
    factorial_distribution_parts, ignore_index=True
)
display(position_weight_factorial_cells_df)
display(position_weight_factorial_results)

# ============================================================
# DURABLE CONTROLLED-ABLATION AND 2 x 2 FACTORIAL CHECKPOINT
# ============================================================
if len(grouped_ablation_results) != 2:
    raise RuntimeError("Controlled grouped-ablation summary must contain two contrasts.")
if len(grouped_ablation_bootstrap_distributions_df) != 2 * ABLATION_BOOTSTRAP_N:
    raise RuntimeError("Controlled grouped-ablation bootstrap coverage is incomplete.")
if len(position_weight_factorial_cells_df) != 4:
    raise RuntimeError("Position-by-weight factorial must contain four cells.")
if len(position_weight_factorial_results) != 7:
    raise RuntimeError("Position-by-weight factorial must contain seven contrasts.")
if len(position_weight_factorial_bootstrap_distributions_df) != 7 * ABLATION_BOOTSTRAP_N:
    raise RuntimeError("Position-by-weight factorial bootstrap coverage is incomplete.")

expected_oof_keys = ["row_index", "pubmed_id", "true"]
for analysis_name, prediction_mapping in [
    ("fixed_feature_ablation", fixed_ablation_predictions),
    ("position_weight_factorial", position_weight_factorial),
]:
    for estimand_name, frame in prediction_mapping.items():
        if (
            len(frame) != len(classical_train_df)
            or frame["row_index"].duplicated().any()
            or set(frame["row_index"]) != set(range(len(classical_train_df)))
        ):
            raise RuntimeError(
                f"Incomplete fixed grouped OOF evidence for {analysis_name}/{estimand_name}."
            )
        if not set(expected_oof_keys + ["pred"]).issubset(frame.columns):
            raise RuntimeError(
                f"Fixed grouped OOF schema is incomplete for {analysis_name}/{estimand_name}."
            )

factorial_stage_artifacts = []
for filename, frame in [
    ("controlled_grouped_ablation_results.csv", grouped_ablation_results),
    (
        "controlled_grouped_ablation_bootstrap_distributions.csv",
        grouped_ablation_bootstrap_distributions_df,
    ),
    ("position_weight_factorial_cells.csv", position_weight_factorial_cells_df),
    ("position_weight_factorial_results.csv", position_weight_factorial_results),
    (
        "position_weight_factorial_bootstrap_distributions.csv",
        position_weight_factorial_bootstrap_distributions_df,
    ),
]:
    factorial_stage_artifacts.append(_stage_artifact_record(
        _atomic_stage_csv(frame, ANALYSIS_STAGE_DIR / filename),
        len(frame),
    ))

# A compressed, columnar restart checkpoint avoids writing the >1 million-row
# long-form OOF CSV twice. The terminal export still writes and manifests the
# reviewer-facing long-form CSV after revalidating every key.
fixed_oof_checkpoint = ANALYSIS_STAGE_DIR / "fixed_ablation_oof_predictions_checkpoint.npz"
fixed_oof_arrays = {}
reference_frame = fixed_ablation_predictions["lexical"].sort_values("row_index")
fixed_oof_arrays["row_index"] = reference_frame["row_index"].to_numpy(dtype=np.int64)
fixed_oof_arrays["pubmed_id"] = reference_frame["pubmed_id"].astype(str).to_numpy()
fixed_oof_arrays["true"] = reference_frame["true"].astype(str).to_numpy()
for analysis_name, prediction_mapping in [
    ("fixed_feature_ablation", fixed_ablation_predictions),
    ("position_weight_factorial", position_weight_factorial),
]:
    for estimand_name, frame in prediction_mapping.items():
        safe_key = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", f"{analysis_name}__{estimand_name}"
        )
        aligned = frame.sort_values("row_index")
        if not np.array_equal(
            aligned["row_index"].to_numpy(dtype=np.int64),
            fixed_oof_arrays["row_index"],
        ):
            raise RuntimeError(f"Fixed OOF row alignment failed for {safe_key}.")
        fixed_oof_arrays[f"pred__{safe_key}"] = aligned["pred"].astype(str).to_numpy()
fixed_oof_temporary = fixed_oof_checkpoint.with_name(
    fixed_oof_checkpoint.name + ".tmp.npz"
)
np.savez_compressed(fixed_oof_temporary, **fixed_oof_arrays)
if not fixed_oof_temporary.is_file() or fixed_oof_temporary.stat().st_size <= 0:
    raise RuntimeError("Compressed fixed-OOF checkpoint was not written.")
os.replace(fixed_oof_temporary, fixed_oof_checkpoint)
factorial_stage_artifacts.append(
    _stage_artifact_record(fixed_oof_checkpoint, len(classical_train_df))
)

factorial_stage_contract = {
    "contract_version": "plos_analysis_stage_v1",
    "stage": "controlled_ablation_and_factorial_complete",
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "n_training_sentences": len(classical_train_df),
    "n_training_abstracts": int(classical_groups.nunique()),
    "n_ablation_contrasts": len(grouped_ablation_results),
    "n_factorial_cells": len(position_weight_factorial_cells_df),
    "n_factorial_contrasts": len(position_weight_factorial_results),
    "bootstrap_replicates_per_contrast": ABLATION_BOOTSTRAP_N,
    "artifacts": factorial_stage_artifacts,
}
_atomic_stage_json(
    factorial_stage_contract,
    ANALYSIS_STAGE_DIR / "GROUPED_FACTORIAL_STAGE_COMPLETE.json",
)
print(
    "Durable grouped-ablation/factorial checkpoint:",
    ANALYSIS_STAGE_DIR / "GROUPED_FACTORIAL_STAGE_COMPLETE.json",
)

from sklearn.metrics import confusion_matrix


def aggregate_cv_confusions(prediction_frame, top_n=10):
    matrix = confusion_matrix(
        prediction_frame["true"], prediction_frame["pred"], labels=LABEL_ORDER
    )
    rows = []
    for true_index, true_label in enumerate(LABEL_ORDER):
        for pred_index, predicted_label in enumerate(LABEL_ORDER):
            if true_index != pred_index and matrix[true_index, pred_index] > 0:
                rows.append({
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[true_index, pred_index]),
                })
    return pd.DataFrame(rows, columns=["true_label", "predicted_label", "count"]).sort_values("count", ascending=False).head(top_n)


for model_name in ["S0_TFIDF_LogReg", "S5_NumberNormalized_LogReg", "S6_PositionAware_LogReg"]:
    if model_name in cv_fold_predictions:
        print(f"Top grouped-OOF confusions for {model_name}")
        display(aggregate_cv_confusions(cv_fold_predictions[model_name], top_n=10))

# Import the actual binding; module discovery alone is insufficient.
HAS_CRF = False
try:
    import sklearn_crfsuite
    import importlib.metadata
    CRF_PACKAGE_VERSION = importlib.metadata.version("sklearn-crfsuite")
    if PUBLICATION_RUN and CRF_PACKAGE_VERSION != "0.5.0":
        raise RuntimeError(
            f"Expected sklearn-crfsuite==0.5.0, found {CRF_PACKAGE_VERSION}"
        )
    HAS_CRF = True
except Exception:
    if RUN_CRF:
        raise
print("sklearn-crfsuite available:", HAS_CRF, "version:", CRF_PACKAGE_VERSION if HAS_CRF else None)


def df_to_abstract_sequences(df: pd.DataFrame):
    seqs_X = []
    seqs_y = []
    seq_ids = []
    for pubmed_id, group in df.sort_values(["pubmed_id", "file_order"]).groupby("pubmed_id"):
        group = group.reset_index(drop=True)
        texts = group["text"].tolist()
        labels = group["label"].tolist()
        feats_seq = []
        n = len(group)
        for i, row in group.iterrows():
            text = row["text"]
            toks = text.split()
            feat = {
                "bias": 1.0,
                "word_count": len(toks),
                "sent_idx": int(row["sentence_index_in_abstract"]),
                "rel_pos": float(row["relative_position"]),
                "is_first": int(row["is_first_sentence"]),
                "is_last": int(row["is_last_sentence"]),
                "first_word": toks[0].lower() if len(toks) > 0 else "",
                "last_word": toks[-1].lower() if len(toks) > 0 else "",
                "prefix3": text[:3].lower(),
                "contains_number": int(bool(re.search(r"\d", text))),
            }
            if i > 0:
                feat["-1:first_word"] = texts[i-1].split()[0].lower() if texts[i-1].split() else ""
            else:
                feat["BOS"] = True
            if i < n - 1:
                feat["+1:first_word"] = texts[i+1].split()[0].lower() if texts[i+1].split() else ""
            else:
                feat["EOS"] = True
            feats_seq.append(feat)
        seqs_X.append(feats_seq)
        seqs_y.append(labels)
        seq_ids.append(pubmed_id)
    return seqs_X, seqs_y, seq_ids

# CRF is a documented development-only sequence comparator.
if RUN_CRF and HAS_CRF:
    X_train_seq, y_train_seq, train_seq_ids = df_to_abstract_sequences(splits["train_seq"])
    X_dev_seq, y_dev_seq, dev_seq_ids = df_to_abstract_sequences(splits["dev_seq"])
    crf_model = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
        all_possible_transitions=True,
    )
    crf_model.fit(X_train_seq, y_train_seq)
    crf_dev_pred = crf_model.predict(X_dev_seq)
    y_true_flat = [label for sequence in y_dev_seq for label in sequence]
    y_pred_flat = [label for sequence in crf_dev_pred for label in sequence]
    crf_dev_order = splits["dev_seq"].sort_values(["pubmed_id", "file_order"]).reset_index(drop=True)
    if crf_dev_order["label"].astype(str).tolist() != [str(value) for value in y_true_flat]:
        raise AssertionError("CRF development output order differs from the canonical frame.")
    crf_development_predictions_df = crf_dev_order[["pubmed_id", "file_order", "label"]].copy()
    crf_development_predictions_df["prediction"] = np.asarray(y_pred_flat)
    crf_development_predictions_df["model"] = "S7_CRF_SequenceModel"
    row = record_result(
        model_name="S7_CRF_SequenceModel",
        split_name="dev",
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        meta={
            "uses_sequence": True,
            "selection_status": "development_only_not_eligible_for_final_test",
            "class_weight": "none",
        },
    )
    display(pd.DataFrame([row]))
    del crf_model, crf_dev_pred, X_train_seq, y_train_seq, X_dev_seq, y_dev_seq
    gc.collect()
else:
    raise RuntimeError("CRF was required but did not run.")


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# MASKED, PACKED ABSTRACT-LEVEL BiLSTM
# ============================================================
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader

MAX_SENTENCE_TOKENS = 64
# Architecture configuration is derived from training only, never from test outcomes.
MAX_ABSTRACT_SENTENCES = int(train_df.groupby("pubmed_id").size().max())
BILSTM_EMBEDDING_DIM = 100
BILSTM_HIDDEN_DIM = 128
BILSTM_LEARNING_RATE = 1e-3
BILSTM_MAX_GRAD_NORM = 1.0


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(str(text).lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_sentence(text, vocab, max_len=MAX_SENTENCE_TOKENS):
    tokens = str(text).lower().split()
    truncated = len(tokens) > max_len
    tokens = tokens[:max_len]
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids, truncated


class AbstractSequenceDataset(TorchDataset):
    def __init__(self, df, vocab, max_sent_len=MAX_SENTENCE_TOKENS, max_abs_len=MAX_ABSTRACT_SENTENCES):
        self.examples = []
        self.sentence_truncation_count = 0
        self.sentence_count = 0
        self.abstract_count = 0
        self.max_observed_abstract_sentences = 0
        ordered = df.sort_values(["pubmed_id", "file_order"])
        for pubmed_id, group in ordered.groupby("pubmed_id", sort=False):
            group = group.reset_index(drop=True)
            self.abstract_count += 1
            self.max_observed_abstract_sentences = max(
                self.max_observed_abstract_sentences, len(group)
            )
            if len(group) > max_abs_len:
                raise ValueError(
                    f"Abstract {pubmed_id} has {len(group)} sentences; training-derived cap is {max_abs_len}."
                )
            encoded = []
            for text in group["text"].tolist():
                ids, truncated = encode_sentence(text, vocab, max_sent_len)
                encoded.append(ids)
                self.sentence_count += 1
                self.sentence_truncation_count += int(truncated)
            labels = [LABEL2ID[label] for label in group["label"].tolist()]
            seq_len = len(encoded)
            encoded.extend([[0] * max_sent_len] * (max_abs_len - seq_len))
            labels.extend([-100] * (max_abs_len - seq_len))
            self.examples.append({
                "pubmed_id": str(pubmed_id),
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "seq_len": torch.tensor(seq_len, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class SentenceEncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=BILSTM_EMBEDDING_DIM,
                 hidden_dim=BILSTM_HIDDEN_DIM, num_labels=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.abstract_bilstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_dim // 2,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, seq_len):
        token_mask = (input_ids != 0).unsqueeze(-1)
        embedded = self.embedding(input_ids)
        token_count = token_mask.sum(dim=2).clamp_min(1)
        sentence_embedding = (embedded * token_mask).sum(dim=2) / token_count
        packed = pack_padded_sequence(
            sentence_embedding, seq_len.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.abstract_bilstm(packed)
        output, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.shape[1]
        )
        return self.classifier(output)


def train_bilstm_style_model(train_df_seq, epochs=5, batch_size=8, lr=1e-3, seed=42):
    set_seed(seed)
    vocab = build_vocab(train_df_seq["text"].tolist(), min_freq=2)
    train_ds = AbstractSequenceDataset(train_df_seq, vocab)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=BILSTM_DATALOADER_WORKERS,
        persistent_workers=BILSTM_DATALOADER_WORKERS > 0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda")
    model = SentenceEncoderBiLSTM(len(vocab)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            lengths = batch["seq_len"].to(device, non_blocking=True)
            logits = model(x, lengths)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), BILSTM_MAX_GRAD_NORM)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": mean_loss, "seed": seed})
        print(f"Epoch {epoch + 1}/{epochs} - train loss: {mean_loss:.4f}")
    return model, vocab, pd.DataFrame(history), train_ds


def predict_bilstm_style_model(model, vocab, eval_df, batch_size=8):
    ds = AbstractSequenceDataset(eval_df, vocab)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=BILSTM_DATALOADER_WORKERS,
        persistent_workers=BILSTM_DATALOADER_WORKERS > 0,
        pin_memory=torch.cuda.is_available(),
    )
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred, score_rows, pubmed_rows, position_rows = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            x = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["seq_len"].to(device, non_blocking=True)
            logits = model(x, lengths).detach().cpu().numpy()
            labels = batch["labels"].numpy()
            for batch_i, (label_seq, logit_seq, length) in enumerate(
                zip(labels, logits, batch["seq_len"].numpy())
            ):
                pmid = batch["pubmed_id"][batch_i]
                for position in range(int(length)):
                    label_id = int(label_seq[position])
                    score = logit_seq[position]
                    y_true.append(ID2LABEL[label_id])
                    y_pred.append(ID2LABEL[int(np.argmax(score))])
                    score_rows.append(score)
                    pubmed_rows.append(str(pmid))
                    position_rows.append(position)
    return {
        "dataset": ds,
        "y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
        "scores": np.asarray(score_rows),
        "score_classes": [ID2LABEL[i] for i in range(len(ID2LABEL))],
        "pubmed_id": np.asarray(pubmed_rows),
        "sentence_index": np.asarray(position_rows),
    }

def _bilstm_audit_row(dataset, split, phase, seed):
    return {
        "split": split, "phase": phase, "reference_seed": seed,
        "n_abstracts": dataset.abstract_count,
        "n_sentences": dataset.sentence_count,
        "max_observed_abstract_sentences": dataset.max_observed_abstract_sentences,
        "training_derived_max_abstract_sentences": MAX_ABSTRACT_SENTENCES,
        "max_sentence_tokens": MAX_SENTENCE_TOKENS,
        "n_sentence_truncated": dataset.sentence_truncation_count,
        "sentence_truncation_rate": (
            dataset.sentence_truncation_count / max(dataset.sentence_count, 1)
        ),
        "n_abstract_truncated": 0,
    }


def run_bilstm_style_experiment(train_df_seq, dev_df_seq, epochs=5, batch_size=8, lr=1e-3, seed=42):
    model, vocab, history, train_dataset = train_bilstm_style_model(
        train_df_seq, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed
    )
    dev_output = predict_bilstm_style_model(model, vocab, dev_df_seq, batch_size=batch_size)
    truncation_audit = pd.DataFrame([
        _bilstm_audit_row(train_dataset, "train", "development_selection", seed),
        _bilstm_audit_row(dev_output["dataset"], "development", "development_selection", seed),
    ])
    return model, vocab, history, dev_output, truncation_audit

if not RUN_BILSTM_STYLE:
    raise RuntimeError("Corrected BiLSTM is required by the locked protocol.")

bilstm_model, bilstm_vocab, bilstm_history, bilstm_dev_output, bilstm_truncation_audit_df = (
    run_bilstm_style_experiment(
        splits["train_seq"], splits["dev_seq"],
        epochs=BILSTM_EPOCHS_FINAL, batch_size=BILSTM_BATCH_SIZE,
        lr=BILSTM_LEARNING_RATE, seed=SEED,
    )
)
row = record_result(
    model_name="S8_BiLSTMStyle_SequentialModel", split_name="dev",
    y_true=bilstm_dev_output["y_true"], y_pred=bilstm_dev_output["y_pred"],
    meta={
        "uses_sequence": True, "masked_and_packed": True,
        "class_weight": "none", "selection_metric": "macro_f1",
    },
)
display(pd.DataFrame([row]))
display(bilstm_truncation_audit_df)
bilstm_dev_order = splits["dev_seq"].sort_values(["pubmed_id", "file_order"]).reset_index(drop=True)
if not np.array_equal(bilstm_dev_order["label"].astype(str).to_numpy(), bilstm_dev_output["y_true"]):
    raise AssertionError("BiLSTM development output order differs from the canonical frame.")
bilstm_development_predictions_df = bilstm_dev_order[["pubmed_id", "file_order", "label"]].copy()
bilstm_development_predictions_df["prediction"] = bilstm_dev_output["y_pred"]
bilstm_development_predictions_df["model"] = "S8_BiLSTMStyle_SequentialModel"
bilstm_development_scores = np.asarray(bilstm_dev_output["scores"]).copy()
bilstm_development_score_classes = list(bilstm_dev_output["score_classes"])
bilstm_truncation_audit_df = bilstm_truncation_audit_df.assign(model="S8_BiLSTMStyle_SequentialModel")
bilstm_model.to("cpu")
del bilstm_model, bilstm_vocab, bilstm_history, bilstm_dev_output
gc.collect()
torch.cuda.empty_cache()

try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
    HAS_TRANSFORMERS = True
except Exception:
    raise

if not HAS_TRANSFORMERS:
    raise RuntimeError("Transformers dependencies are required by this notebook.")
print("Transformers available:", HAS_TRANSFORMERS)

TRANSFORMER_MODELS = {
    "bert_base": {
        "repo_id": "google-bert/bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "display_name": "BERT-base uncased",
    },
    "scibert": {
        "repo_id": "allenai/scibert_scivocab_uncased",
        "revision": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
        "display_name": "SciBERT scivocab uncased",
    },
    "biomedbert": {
        "repo_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "revision": "d673b8835373c6fa116d6d8006b33d48734e305d",
        "display_name": "BiomedBERT abstract uncased (formerly PubMedBERT)",
    },
}
TRANSFORMER_MAX_LENGTH = 256


def add_prev_sentence_context(df):
    out = df.sort_values(["pubmed_id", "file_order"]).copy()
    out["prev_text"] = out.groupby("pubmed_id", sort=False)["text"].shift(1).fillna("")
    out["is_first_context_item"] = out["prev_text"].eq("")
    return out.reset_index(drop=True)


splits["train_sent_ctx"] = add_prev_sentence_context(splits["train_sent"])
splits["dev_sent_ctx"] = add_prev_sentence_context(splits["dev_sent"])

def compute_metrics_for_hf(eval_pred):
    logits, labels = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    labels_txt = [ID2LABEL[int(value)] for value in labels]
    preds_txt = [ID2LABEL[int(value)] for value in pred_ids]
    metrics = compute_metrics(labels_txt, preds_txt)
    return {key: metrics[key] for key in ["accuracy", "macro_f1", "weighted_f1"]}


def _load_fast_transformer_tokenizer(model_path, revision):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"A fast tokenizer is required for {model_path}.")
    if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
        raise RuntimeError(f"Tokenizer {model_path} does not define a padding token.")
    if TRANSFORMER_MAX_LENGTH > tokenizer.model_max_length:
        raise RuntimeError(
            f"TRANSFORMER_MAX_LENGTH={TRANSFORMER_MAX_LENGTH} exceeds the "
            f"checkpoint limit {tokenizer.model_max_length} for {model_path}."
        )
    return tokenizer


def _tokenize_preserve_current(tokenizer, previous_texts, current_texts, max_length=256):
    if len(previous_texts) != len(current_texts):
        raise ValueError("Previous/current tokenizer batches must have equal lengths.")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "Target-preserving tokenization requires a fast Hugging Face tokenizer. "
            "Load the checkpoint with AutoTokenizer(..., use_fast=True)."
        )
    if tokenizer.pad_token_id is None or tokenizer.pad_token is None:
        raise ValueError("The checkpoint tokenizer must define a padding token.")
    if tokenizer.padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported tokenizer padding side: {tokenizer.padding_side!r}")

    # Transformers 5 removed PreTrainedTokenizer.prepare_for_model. Work at the
    # public Tokenizers backend so checkpoint-specific post-processing, special
    # tokens, type IDs, truncation direction, and padding remain explicit.
    backend.no_truncation()
    backend.no_padding()
    previous_encodings = backend.encode_batch(
        [str(text) for text in previous_texts], add_special_tokens=False
    )
    current_encodings = backend.encode_batch(
        [str(text) for text in current_texts], add_special_tokens=False
    )

    batch = {"input_ids": [], "attention_mask": []}
    include_type_ids = "token_type_ids" in tokenizer.model_input_names
    if include_type_ids:
        batch["token_type_ids"] = []
    current_truncated, previous_truncated = [], []
    pad_type_id = int(getattr(tokenizer, "pad_token_type_id", 0) or 0)
    single_special_count = int(backend.num_special_tokens_to_add(False))
    pair_special_count = int(backend.num_special_tokens_to_add(True))
    if max_length <= max(single_special_count, pair_special_count):
        raise ValueError(f"max_length={max_length} leaves no text-token budget.")

    for previous_encoding, current_encoding in zip(
        previous_encodings, current_encodings, strict=True
    ):
        previous_length = len(previous_encoding.ids)
        current_length = len(current_encoding.ids)
        has_previous = previous_length > 0

        # Previous context is sacrificed before the current target. If the target
        # itself leaves no room for context, use a true single-input encoding and
        # its full budget. This also makes first-sentence context-mode inputs
        # identical to sentence-only inputs at the same max length.
        pair_current_budget = max_length - pair_special_count
        use_pair = has_previous and current_length < pair_current_budget
        if use_pair:
            remaining = pair_current_budget - current_length
            if previous_length > remaining:
                previous_encoding.truncate(remaining, direction="left")
            encoded = backend.post_process(
                previous_encoding, current_encoding, add_special_tokens=True
            )
            previous_kept = len(previous_encoding.ids)
        else:
            single_current_budget = max_length - single_special_count
            if current_length > single_current_budget:
                current_encoding.truncate(single_current_budget, direction="right")
            encoded = backend.post_process(current_encoding, add_special_tokens=True)
            previous_kept = 0

        if len(encoded.ids) > max_length:
            raise AssertionError("Tokenizer post-processing exceeded max_length.")
        encoded.pad(
            max_length, direction=tokenizer.padding_side,
            pad_id=int(tokenizer.pad_token_id), pad_type_id=pad_type_id,
            pad_token=str(tokenizer.pad_token),
        )
        if len(encoded.ids) != max_length:
            raise AssertionError("Tokenizer padding did not produce max_length inputs.")

        batch["input_ids"].append(list(encoded.ids))
        batch["attention_mask"].append(list(encoded.attention_mask))
        if include_type_ids:
            batch["token_type_ids"].append(list(encoded.type_ids))
        current_truncated.append(current_length > len(current_encoding.ids))
        previous_truncated.append(previous_length > previous_kept)

    batch["current_truncated"] = current_truncated
    batch["previous_truncated"] = previous_truncated
    return batch

def build_transformer_dataset(df, tokenizer, mode, max_length=TRANSFORMER_MAX_LENGTH):
    source_frame = add_prev_sentence_context(df)
    dataset = Dataset.from_pandas(
        source_frame[["text", "prev_text", "label_id", "pubmed_id", "file_order"]]
        .rename(columns={"label_id": "labels"}), preserve_index=False,
    )
    if mode == "sentence_only":
        def tokenize(batch):
            raw = tokenizer(
                batch["text"], add_special_tokens=False, padding=False,
                truncation=False, return_attention_mask=False,
                return_token_type_ids=False, verbose=False,
            )
            special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
            lengths = [len(ids) + special_token_count for ids in raw["input_ids"]]
            encoded = tokenizer(
                batch["text"], truncation=True, padding="max_length", max_length=max_length
            )
            encoded["current_truncated"] = [length > max_length for length in lengths]
            encoded["previous_truncated"] = [False] * len(lengths)
            return encoded
    elif mode == "prev_plus_current":
        def tokenize(batch):
            return _tokenize_preserve_current(
                tokenizer, batch["prev_text"], batch["text"], max_length=max_length
            )
    else:
        raise ValueError(f"Unknown transformer input mode: {mode}")

    dataset = dataset.map(tokenize, batched=True)
    current_flags = np.asarray(dataset["current_truncated"], dtype=bool)
    previous_flags = np.asarray(dataset["previous_truncated"], dtype=bool)
    audit = pd.DataFrame([{
        "mode": mode, "n_rows": len(dataset),
        "current_truncated_n": int(current_flags.sum()),
        "current_truncated_rate": float(current_flags.mean()),
        "previous_truncated_n": int(previous_flags.sum()),
        "previous_truncated_rate": float(previous_flags.mean()),
        "max_length": max_length,
    }])
    model_columns = [
        name for name in ["input_ids", "attention_mask", "token_type_ids", "labels"]
        if name in dataset.column_names
    ]
    dataset.set_format(type="torch", columns=model_columns, output_all_columns=False)
    return dataset, source_frame, audit


def make_transformer_training_arguments(output_dir, epochs, batch_size, seed, evaluate):
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch" if evaluate else "no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=TRANSFORMER_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=TRANSFORMER_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=2e-5,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_grad_norm=1.0,
        fp16=TRANSFORMER_USE_FP16,
        bf16=TRANSFORMER_USE_BF16,
        tf32=TRANSFORMER_USE_TF32,
        full_determinism=False,
        dataloader_num_workers=TRANSFORMER_DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=TRANSFORMER_DATALOADER_WORKERS > 0,
        dataloader_prefetch_factor=2,
        remove_unused_columns=True,
        gradient_checkpointing=False,
        load_best_model_at_end=False,
        report_to=[],
        seed=seed,
        data_seed=seed,
    )


def predict_transformer_logits_direct(model, dataset, batch_size):
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=TRANSFORMER_DATALOADER_WORKERS,
        persistent_workers=TRANSFORMER_DATALOADER_WORKERS > 0,
        pin_memory=True, prefetch_factor=2,
    )
    logits = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items() if key != "labels"
            }
            logits.append(model(**inputs).logits.detach().cpu().numpy())
    return np.concatenate(logits, axis=0)


def run_transformer_experiment(model_key, train_df, dev_df, mode="sentence_only",
                               epochs=2, batch_size=8):
    set_seed(SEED)  # reset before the randomly initialized classifier head is constructed
    model_spec = TRANSFORMER_MODELS[model_key]
    model_path = model_spec["repo_id"]
    revision = model_spec["revision"]
    tokenizer = _load_fast_transformer_tokenizer(model_path, revision)
    train_dataset, _, train_audit = build_transformer_dataset(train_df, tokenizer, mode)
    dev_dataset, dev_source, dev_audit = build_transformer_dataset(dev_df, tokenizer, mode)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, revision=revision, num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    args = make_transformer_training_arguments(
        TRANSFORMER_SCRATCH_ROOT / "selection" / f"{model_key}_{mode}",
        epochs, batch_size, SEED, evaluate=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset,
        eval_dataset=dev_dataset, compute_metrics=compute_metrics_for_hf,
    )
    trainer.train()
    prediction = trainer.predict(dev_dataset)
    dev_scores = np.asarray(prediction.predictions).copy()
    y_true = dev_source["label"].astype(str).to_numpy()
    y_pred = np.asarray([ID2LABEL[int(index)] for index in np.argmax(dev_scores, axis=1)])
    audit = pd.concat([
        train_audit.assign(split="train"),
        dev_audit.assign(split="development"),
    ], ignore_index=True).assign(
        checkpoint=model_path, revision=revision,
        phase="development_selection", reference_seed=REFERENCE_SEED,
    )
    model.to("cpu")
    del prediction, trainer, model, train_dataset, dev_dataset, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return y_true, y_pred, dev_scores, audit

# ---------------------------------------------------------------------
# Transformer development candidates
#
# Compatibility fix:
# Some datasets/torchvision combinations cause datasets' TorchFormatter
# to import torchvision.io.VideoReader, which may no longer exist.
# This audit is text-only, so torchvision support can safely be disabled
# inside `datasets` without affecting tokenization or model training.
# ---------------------------------------------------------------------

import gc
import pandas as pd
import torch

from datasets import config as datasets_config

try:
    # This succeeds on older compatible torchvision installations.
    from torchvision.io import VideoReader as _VideoReader  # noqa: F401
except (ImportError, AttributeError, RuntimeError, OSError):
    datasets_config.TORCHVISION_AVAILABLE = False

    # Ensure an already-imported TorchFormatter sees the same setting.
    try:
        import datasets.formatting.torch_formatter as _torch_formatter

        _torch_formatter.config.TORCHVISION_AVAILABLE = False
    except ImportError:
        pass

    print(
        "[compatibility] Disabled optional torchvision handling in "
        "Hugging Face Datasets; transformer text training is unaffected."
    )


if any(str(row.get("model_name", "")).startswith("S9_") for row in results_store):
    raise RuntimeError(
        "Transformer development rows already exist. Restart the kernel before rerunning "
        "the protocol so candidate ranking cannot contain duplicate partial-run rows."
    )

if not (RUN_TRANSFORMERS and HAS_TRANSFORMERS):
    raise RuntimeError(
        "All transformer candidate runs are required by the locked protocol."
    )


TRANSFORMER_CANDIDATES = [
    (model_key, mode)
    for model_key in ["bert_base", "scibert", "biomedbert"]
    for mode in ["sentence_only", "prev_plus_current"]
]

transformer_truncation_audits = []
transformer_development_predictions = {}

MAX_LOCKED_TRANSFORMER_SYSTEMS = 3
MAX_TRANSFORMER_TRAINING_BATCHES = int(
    math.ceil(len(splits["train_sent"]) / TRANSFORMER_BATCH_SIZE)
    * TRANSFORMER_EPOCHS_FINAL
    * (len(TRANSFORMER_CANDIDATES) + MAX_LOCKED_TRANSFORMER_SYSTEMS * 3)
)
print("Projected upper bound on G4 transformer training batches =", MAX_TRANSFORMER_TRAINING_BATCHES)
transformer_development_scores = {}
transformer_development_score_classes = {}

partial_results_path = (
    OUTPUT_DIR / "transformer_selection_results_partial.csv"
)
partial_truncation_path = (
    OUTPUT_DIR / "transformer_selection_truncation_partial.csv"
)


def save_transformer_partials() -> None:
    """Persist completed results without changing protocol behavior."""
    pd.DataFrame(results_store).to_csv(
        partial_results_path,
        index=False,
    )

    if transformer_truncation_audits:
        pd.concat(
            transformer_truncation_audits,
            ignore_index=True,
        ).to_csv(
            partial_truncation_path,
            index=False,
        )


for model_key, mode in TRANSFORMER_CANDIDATES:
    model_label = f"S9_{model_key}_{mode}"

    print(
        f"Running transformer development candidate: "
        f"{model_key} | {mode}"
    )

    y_true = None
    y_pred = None
    dev_scores = None
    audit = None

    try:
        y_true, y_pred, dev_scores, audit = run_transformer_experiment(
            model_key,
            splits["train_sent"],
            splits["dev_sent"],
            mode=mode,
            epochs=TRANSFORMER_EPOCHS_FINAL,
            batch_size=TRANSFORMER_BATCH_SIZE,
        )

        canonical_dev_labels = splits["dev_sent"].sort_values(
            ["pubmed_id", "file_order"]
        )["label"].astype(str).to_numpy()
        if not np.array_equal(np.asarray(y_true).astype(str), canonical_dev_labels):
            raise AssertionError("Transformer development output order differs from the canonical frame.")
        transformer_development_predictions[model_label] = np.asarray(y_pred).copy()
        transformer_development_scores[model_label] = np.asarray(dev_scores).copy()
        transformer_development_score_classes[model_label] = [
            ID2LABEL[index] for index in range(len(ID2LABEL))
        ]

        record_result(
            model_name=model_label,
            split_name="dev",
            y_true=y_true,
            y_pred=y_pred,
            meta={
                "family": "transformer",
                "context_mode": mode,
                "selection_metric": "macro_f1",
                "class_weight": "none",
                "checkpoint": TRANSFORMER_MODELS[model_key]["repo_id"],
                "revision": TRANSFORMER_MODELS[model_key]["revision"],
            },
        )

        if not isinstance(audit, pd.DataFrame):
            audit = pd.DataFrame(audit)

        transformer_truncation_audits.append(
            audit.assign(model=model_label)
        )

        save_transformer_partials()

    except Exception as exc:
        # Preserve every successfully completed preceding candidate, but
        # stop immediately because all six runs are protocol-required.
        save_transformer_partials()

        raise RuntimeError(
            "Transformer candidate failed after partial outputs were saved: "
            f"{model_label}"
        ) from exc

    finally:
        del y_true, y_pred, dev_scores, audit

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


expected_transformer_labels = {
    f"S9_{model_key}_{mode}" for model_key, mode in TRANSFORMER_CANDIDATES
}
if (
    set(transformer_development_predictions) != expected_transformer_labels
    or set(transformer_development_scores) != expected_transformer_labels
):
    raise RuntimeError("Transformer development prediction/score coverage is incomplete.")
candidate_result_names = [
    row.get("model_name") for row in results_store
    if str(row.get("model_name", "")).startswith("S9_") and row.get("split_name") == "dev"
]
if len(candidate_result_names) != len(set(candidate_result_names)):
    raise RuntimeError("Duplicate transformer development result rows detected.")

if len(transformer_truncation_audits) != len(TRANSFORMER_CANDIDATES):
    raise RuntimeError(
        "Transformer candidate execution was incomplete: "
        f"{len(transformer_truncation_audits)} of "
        f"{len(TRANSFORMER_CANDIDATES)} runs produced truncation audits."
    )


transformer_truncation_audit_df = pd.concat(
    transformer_truncation_audits,
    ignore_index=True,
)

# Write the complete accumulated state once more after all six runs.
save_transformer_partials()

display(transformer_truncation_audit_df)

dev_results = summarize_results_table(results_store)

print("Classical CV summary")
display(classical_cv_summary)

print("Hold-out development summary for sequential / transformer models")
display(dev_results)

CLASSICAL_SELECTION_METRIC = "macro_f1_mean"
HOLDOUT_SELECTION_METRIC = "macro_f1"

# These declared orders are prespecified deterministic tie-breakers only; the
# primary sort key remains Macro-F1 descending.
ELIGIBLE_CLASSICAL_MODELS = [
    "S0_TFIDF_LogReg",
    "S1_TFIDF_LinearSVM",
    "S2_TFIDF_MNB",
    "S5_NumberNormalized_LogReg",
    "S6_PositionAware_LogReg",
]
CLASSICAL_SENSITIVITY_ONLY_MODELS = ["S1b_TFIDF_LinearSVM_OVO"]
TRANSFORMER_TIE_BREAK_ORDER = [
    f"S9_{model_key}_{mode}"
    for model_key in ["bert_base", "scibert", "biomedbert"]
    for mode in ["sentence_only", "prev_plus_current"]
]
SEQUENCE_DEVELOPMENT_MODELS = [
    "S7_CRF_SequenceModel",
    "S8_BiLSTMStyle_SequentialModel",
]

classical_candidate_cols = [
    "model_name", "macro_f1_mean", "macro_f1_std", "accuracy_mean",
]
classical_candidate_table = classical_cv_summary[
    classical_cv_summary["model_name"].isin(ELIGIBLE_CLASSICAL_MODELS)
][classical_candidate_cols].copy()
classical_candidate_table["tie_break_order"] = classical_candidate_table[
    "model_name"
].map({name: index for index, name in enumerate(ELIGIBLE_CLASSICAL_MODELS)})
classical_candidate_table = classical_candidate_table.sort_values(
    [CLASSICAL_SELECTION_METRIC, "tie_break_order"],
    ascending=[False, True], kind="mergesort",
).reset_index(drop=True)

development_aggregate = dev_results[dev_results["split_name"] == "dev"].copy()
transformer_candidate_table = development_aggregate[
    development_aggregate["model_name"].isin(TRANSFORMER_TIE_BREAK_ORDER)
][["model_name", "macro_f1", "weighted_f1", "accuracy"]].copy()
transformer_candidate_table["tie_break_order"] = transformer_candidate_table[
    "model_name"
].map({name: index for index, name in enumerate(TRANSFORMER_TIE_BREAK_ORDER)})
transformer_candidate_table = transformer_candidate_table.sort_values(
    [HOLDOUT_SELECTION_METRIC, "tie_break_order"],
    ascending=[False, True], kind="mergesort",
).reset_index(drop=True)

sequence_development_table = development_aggregate[
    development_aggregate["model_name"].isin(SEQUENCE_DEVELOPMENT_MODELS)
][["model_name", "macro_f1", "weighted_f1", "accuracy"]].copy()
sequence_development_table["selection_role"] = sequence_development_table[
    "model_name"
].map({
    "S7_CRF_SequenceModel": "development-only exploratory comparator",
    "S8_BiLSTMStyle_SequentialModel": "mandatory final sequence comparator",
})

if set(transformer_candidate_table["model_name"]) != set(TRANSFORMER_TIE_BREAK_ORDER):
    raise RuntimeError("Transformer candidate table is incomplete or duplicated.")
if transformer_candidate_table["model_name"].duplicated().any():
    raise RuntimeError("Transformer candidate table contains duplicate model rows.")

display(classical_candidate_table)
display(transformer_candidate_table)
display(sequence_development_table)

# ============================================================
# HASHED FINAL-SYSTEM AND COMPARISON LOCK (NO CORRECTED TEST PREDICTIONS)
# ============================================================
N_FINAL_CLASSICAL = 4
BASELINE_MODEL = "S0_TFIDF_LogReg"
POSITION_MODEL = "S6_PositionAware_LogReg"
BILSTM_MODEL = "S8_BiLSTMStyle_SequentialModel"

top_classical = classical_candidate_table["model_name"].head(N_FINAL_CLASSICAL).tolist()
top_transformers = transformer_candidate_table["model_name"].head(2).tolist()
if len(top_transformers) != 2 or len(set(top_transformers)) != 2:
    raise RuntimeError("The locked rule requires two distinct completed transformer candidates.")

PRIMARY_TRANSFORMER, SECOND_TRANSFORMER = top_transformers
if PRIMARY_TRANSFORMER.endswith("_prev_plus_current"):
    PRIMARY_TRANSFORMER_COUNTERPART = PRIMARY_TRANSFORMER.replace(
        "_prev_plus_current", "_sentence_only"
    )
elif PRIMARY_TRANSFORMER.endswith("_sentence_only"):
    PRIMARY_TRANSFORMER_COUNTERPART = PRIMARY_TRANSFORMER.replace(
        "_sentence_only", "_prev_plus_current"
    )
else:
    raise RuntimeError(f"Unrecognised transformer mode: {PRIMARY_TRANSFORMER}")
if PRIMARY_TRANSFORMER_COUNTERPART not in set(transformer_candidate_table["model_name"]):
    raise RuntimeError("The paired sentence/context transformer counterpart is missing.")

FINAL_SYSTEMS = list(dict.fromkeys(
    top_classical + [BASELINE_MODEL, POSITION_MODEL, BILSTM_MODEL]
    + top_transformers + [PRIMARY_TRANSFORMER_COUNTERPART]
))
required_final_systems = {
    BASELINE_MODEL, POSITION_MODEL, BILSTM_MODEL,
    PRIMARY_TRANSFORMER, SECOND_TRANSFORMER, PRIMARY_TRANSFORMER_COUNTERPART,
}
if required_final_systems - set(FINAL_SYSTEMS):
    raise AssertionError("Locked final-system rule omitted a required system.")
if "S7_CRF_SequenceModel" in FINAL_SYSTEMS:
    raise AssertionError("CRF is development-only and cannot enter final test evaluation.")

selection_audit_parts = []
classical_audit = classical_candidate_table.copy()
classical_audit["candidate_group"] = "classical_training_grouped_cv"
classical_audit["selection_stage"] = "training_only_grouped_nested_cv_outer_estimate"
classical_audit["selection_metric"] = CLASSICAL_SELECTION_METRIC
classical_audit["rank_within_candidate_group"] = np.arange(1, len(classical_audit) + 1)
classical_audit["selected_for_final_test"] = classical_audit["model_name"].isin(FINAL_SYSTEMS)
selection_audit_parts.append(classical_audit)

transformer_audit = transformer_candidate_table.copy()
transformer_audit["candidate_group"] = "transformer_development_candidates"
transformer_audit["selection_stage"] = "development_aggregate_macro_f1"
transformer_audit["selection_metric"] = HOLDOUT_SELECTION_METRIC
transformer_audit["rank_within_candidate_group"] = np.arange(1, len(transformer_audit) + 1)
transformer_audit["selected_for_final_test"] = transformer_audit["model_name"].isin(FINAL_SYSTEMS)
selection_audit_parts.append(transformer_audit)

sequence_audit = sequence_development_table.copy()
sequence_audit["candidate_group"] = "fixed_sequence_comparators"
sequence_audit["selection_stage"] = sequence_audit["model_name"].map({
    "S7_CRF_SequenceModel": "development_only_ineligible_for_test",
    "S8_BiLSTMStyle_SequentialModel": "mandatory_final_comparator_by_protocol",
})
sequence_audit["selection_metric"] = "development Macro-F1 (descriptive; no cross-family rank)"
sequence_audit["rank_within_candidate_group"] = np.nan
sequence_audit["selected_for_final_test"] = sequence_audit["model_name"].isin(FINAL_SYSTEMS)
selection_audit_parts.append(sequence_audit)

final_system_selection_audit_df = pd.concat(
    selection_audit_parts, ignore_index=True, sort=False
)
final_system_selection_audit_df["locked_before_corrected_test_prediction"] = True
final_system_selection_audit_df["reference_seed"] = REFERENCE_SEED

raw_comparison_specs = [
    (PRIMARY_TRANSFORMER, BILSTM_MODEL, "primary_transformer_vs_bilstm", "primary", True),
    (PRIMARY_TRANSFORMER, PRIMARY_TRANSFORMER_COUNTERPART, "context_input_mode", "secondary_descriptive", False),
    (PRIMARY_TRANSFORMER, SECOND_TRANSFORMER, "second_ranked_transformer", "secondary_descriptive", False),
    (PRIMARY_TRANSFORMER, POSITION_MODEL, "position_aware_classical", "secondary_descriptive", False),
]
comparison_records = {}
for model_a, model_b, family, role, formal in raw_comparison_specs:
    if model_a == model_b:
        continue
    key = (model_a, model_b)
    if key not in comparison_records:
        comparison_records[key] = {
            "model_a": model_a, "model_b": model_b,
            "comparison_family": family, "analysis_role": role,
            "formal_inference_allowed": bool(formal),
        }
    else:
        comparison_records[key]["comparison_family"] += "+" + family
pretest_comparison_lock_df = pd.DataFrame(comparison_records.values())
if int(pretest_comparison_lock_df["formal_inference_allowed"].sum()) != 1:
    raise AssertionError("Exactly one prespecified primary comparison is required.")
if not set(pretest_comparison_lock_df["model_a"]).union(
    pretest_comparison_lock_df["model_b"]
).issubset(set(FINAL_SYSTEMS)):
    raise AssertionError("A prespecified comparison member is absent from FINAL_SYSTEMS.")
comparison_pairs = [
    (
        row.model_a, row.model_b, row.comparison_family,
        row.analysis_role, bool(row.formal_inference_allowed),
    )
    for row in pretest_comparison_lock_df.itertuples(index=False)
]

PRETEST_LOCK_DIR = Path(OUTPUT_DIR) / "pretest_lock"
PRETEST_LOCK_DIR.mkdir(parents=True, exist_ok=False)
PRETEST_SELECTION_CSV = PRETEST_LOCK_DIR / "final_system_selection_audit_pretest.csv"
PRETEST_COMPARISON_CSV = PRETEST_LOCK_DIR / "comparison_definitions_pretest.csv"
PRETEST_LOCK_JSON = PRETEST_LOCK_DIR / "SELECTION_AND_COMPARISON_LOCK.json"
final_system_selection_audit_df.to_csv(PRETEST_SELECTION_CSV, index=False, lineterminator="\n")
pretest_comparison_lock_df.to_csv(PRETEST_COMPARISON_CSV, index=False, lineterminator="\n")
pretest_lock_payload = {
    "lock_version": "plos_r1_selection_comparison_lock_v2",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "data_sha256": {row["split"]: row["sha256"] for row in PUBLICATION_FILE_PREFLIGHT},
    "test_access_disclosure": (
        "fixed public benchmark test split was evaluated in the original submission; "
        "no corrected-run predictions or performance summaries informed this lock"
    ),
    "classical_tie_break_order": ELIGIBLE_CLASSICAL_MODELS,
    "transformer_tie_break_order": TRANSFORMER_TIE_BREAK_ORDER,
    "final_systems": FINAL_SYSTEMS,
    "reference_seed": REFERENCE_SEED,
    "deep_seeds": [42, 43, 44] if PUBLICATION_RUN else [42],
    "comparisons": pretest_comparison_lock_df.to_dict(orient="records"),
    "selection_csv_sha256": _sha256_file(PRETEST_SELECTION_CSV),
    "comparison_csv_sha256": _sha256_file(PRETEST_COMPARISON_CSV),
}
PRETEST_LOCK_JSON.write_text(
    json.dumps(pretest_lock_payload, indent=2, sort_keys=True), encoding="utf-8"
)
PRETEST_SELECTION_LOCK_SHA256 = _sha256_file(PRETEST_LOCK_JSON)

FINAL_SYSTEMS_LOCKED = True
TEST_OUTCOMES_ACCESSED = False
print("Final systems and comparisons locked before corrected-run test prediction:")
for index, name in enumerate(FINAL_SYSTEMS, start=1):
    print(f"{index}. {name}")
print("Pre-test lock SHA-256:", PRETEST_SELECTION_LOCK_SHA256)
display(final_system_selection_audit_df)
display(pretest_comparison_lock_df)

# ============================================================
# PLOS REVISION: CLEAN FINAL FIT, TEST INFERENCE, AND SCORE CAPTURE
# ============================================================
# Corrected-run test prediction starts only after the durable selection/comparison lock.
# Repeated inference is timing-only and cannot change any locked choice.
# Resource values are implementation-specific: each row records whether preprocessing
# is included and the exact stage scope. Classical GPU memory is not applicable.

from time import perf_counter
from sklearn.base import clone

if not globals().get("FINAL_SYSTEMS_LOCKED", False):
    raise RuntimeError("Final systems must be locked before corrected-run test prediction.")
for lock_path in [PRETEST_SELECTION_CSV, PRETEST_COMPARISON_CSV, PRETEST_LOCK_JSON]:
    if not Path(lock_path).is_file() or Path(lock_path).stat().st_size <= 0:
        raise RuntimeError(f"Pre-test lock artifact is missing: {lock_path}")
if _sha256_file(PRETEST_LOCK_JSON) != PRETEST_SELECTION_LOCK_SHA256:
    raise RuntimeError("Pre-test selection/comparison lock changed before test prediction.")
locked_payload = json.loads(PRETEST_LOCK_JSON.read_text(encoding="utf-8"))
if locked_payload["final_systems"] != FINAL_SYSTEMS:
    raise RuntimeError("In-memory FINAL_SYSTEMS differs from the durable pre-test lock.")
if locked_payload["source_notebook_sha256"] != SOURCE_NOTEBOOK_SHA256:
    raise RuntimeError("Source SHA-256 differs from the durable pre-test lock.")
if _sha256_file(PRETEST_SELECTION_CSV) != locked_payload["selection_csv_sha256"]:
    raise RuntimeError("Pre-test system-selection CSV changed before test prediction.")
if _sha256_file(PRETEST_COMPARISON_CSV) != locked_payload["comparison_csv_sha256"]:
    raise RuntimeError("Pre-test comparison-definition CSV changed before test prediction.")
REVISION_DEEP_SEEDS = [42, 43, 44] if PUBLICATION_RUN else [42]
INFERENCE_REPEATS = 5 if PUBLICATION_RUN else 2

full_train_sent = splits["train_sent"].copy().reset_index(drop=True)
full_train_seq = splits["train_seq"].copy().reset_index(drop=True)
dev_sent = splits["dev_sent"].copy().reset_index(drop=True)
test_sent = splits["test_sent"].copy().reset_index(drop=True)
# Fixed label encoding is applied only after the selection/comparison lock.
test_sent["label_id"] = label_encoder.transform(test_sent["label"])
TEST_OUTCOMES_ACCESSED = True

final_test_results = []
final_test_predictions = {}
final_test_predictions_by_seed = {}
final_test_scores = {}
final_dev_scores = {}
final_fitted_systems = {}
revision_efficiency_records = []
revision_efficiency_repeat_records = []
transformer_final_truncation_audits = []
bilstm_final_truncation_audits = []
final_test_y_true = test_sent["label"].astype(str).to_numpy()
final_test_pubmed_ids = test_sent["pubmed_id"].astype(str).to_numpy()

FINAL_STAGE_DIR = Path(OUTPUT_DIR) / "plos_revision_stage_checkpoints"
FINAL_STAGE_DIR.mkdir(parents=True, exist_ok=True)

def _persist_final_stage(stage_name):
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage_name))
    _atomic_stage_csv(
        pd.DataFrame(final_test_results),
        FINAL_STAGE_DIR / "partial_final_test_results.csv",
    )
    _atomic_stage_csv(
        pd.DataFrame(revision_efficiency_records),
        FINAL_STAGE_DIR / "partial_efficiency.csv",
    )

    prediction_checkpoint_rows = []
    for (model_name, seed_value), prediction in sorted(
        final_test_predictions_by_seed.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name))
        prediction_array = np.asarray(prediction, dtype=np.str_)
        if (
            len(prediction_array) != len(final_test_y_true)
            or not set(prediction_array).issubset(set(LABEL_ORDER))
        ):
            raise RuntimeError(
                f"Invalid checkpoint prediction vector for {model_name}/{seed_value}."
            )
        prediction_path = (
            FINAL_STAGE_DIR
            / f"prediction__{safe_model}__seed_{seed_value}.npy"
        )
        temporary_prediction_path = prediction_path.with_name(
            prediction_path.name + ".tmp.npy"
        )
        np.save(
            temporary_prediction_path,
            prediction_array,
            allow_pickle=False,
        )
        if (
            not temporary_prediction_path.is_file()
            or temporary_prediction_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Prediction checkpoint write failed: {temporary_prediction_path}"
            )
        os.replace(temporary_prediction_path, prediction_path)
        prediction_checkpoint_rows.append({
            "model": str(model_name),
            "seed": str(seed_value),
            "file": prediction_path.name,
            "n_predictions": int(len(prediction_array)),
            "dtype": str(prediction_array.dtype),
            "bytes": int(prediction_path.stat().st_size),
            "sha256": _sha256_file(prediction_path),
        })

    for array_name, values in [
        ("final_test_y_true.npy", np.asarray(final_test_y_true, dtype=np.str_)),
        (
            "final_test_pubmed_ids.npy",
            np.asarray(final_test_pubmed_ids, dtype=np.str_),
        ),
    ]:
        array_path = FINAL_STAGE_DIR / array_name
        temporary_array_path = array_path.with_name(
            array_path.name + ".tmp.npy"
        )
        np.save(temporary_array_path, values, allow_pickle=False)
        if (
            not temporary_array_path.is_file()
            or temporary_array_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Final-test bootstrap input checkpoint failed: {array_path}"
            )
        os.replace(temporary_array_path, array_path)

    prediction_checkpoint_manifest = {
        "contract_version": "plos_final_prediction_checkpoint_v1",
        "run_id": RUN_ID,
        "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "latest_completed_stage": safe_stage,
        "n_result_rows": len(final_test_results),
        "n_test_sentences": len(final_test_y_true),
        "n_test_abstracts": int(
            pd.Series(final_test_pubmed_ids).nunique()
        ),
        "predictions": prediction_checkpoint_rows,
        "true_labels_file": "final_test_y_true.npy",
        "pubmed_ids_file": "final_test_pubmed_ids.npy",
    }
    _atomic_stage_json(
        prediction_checkpoint_manifest,
        FINAL_STAGE_DIR / "prediction_checkpoint_manifest.json",
    )
    _atomic_stage_json(
        {
            "latest_completed_stage": safe_stage,
            "run_id": RUN_ID,
            "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
            "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
            "n_result_rows": len(final_test_results),
            "n_prediction_vectors": len(prediction_checkpoint_rows),
        },
        FINAL_STAGE_DIR / "stage_status.json",
    )
    print("Persisted final-run stage:", safe_stage)


def _begin_stage_resource_profile(gpu_applicable):
    gc.collect()
    cpu_monitor = PeakCPUMemoryMonitor(interval_seconds=0.05)
    if gpu_applicable and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gpu_baseline_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_baseline_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()
    else:
        gpu_baseline_allocated_gb = np.nan
        gpu_baseline_reserved_gb = np.nan
    cpu_monitor.start()
    return {
        "cpu_monitor": cpu_monitor,
        "gpu_applicable": bool(gpu_applicable and torch.cuda.is_available()),
        "gpu_baseline_allocated_gb": gpu_baseline_allocated_gb,
        "gpu_baseline_reserved_gb": gpu_baseline_reserved_gb,
    }


def _end_stage_resource_profile(profile):
    if profile["gpu_applicable"]:
        torch.cuda.synchronize()
        gpu_peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        gpu_peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        gpu_peak_allocated_gb = np.nan
        gpu_peak_reserved_gb = np.nan
    cpu_stats = profile["cpu_monitor"].stop()
    return {
        **cpu_stats,
        "gpu_baseline_allocated_gb": profile["gpu_baseline_allocated_gb"],
        "gpu_baseline_reserved_gb": profile["gpu_baseline_reserved_gb"],
        "gpu_peak_allocated_gb": gpu_peak_allocated_gb,
        "gpu_peak_reserved_gb": gpu_peak_reserved_gb,
        "gpu_peak_increment_allocated_gb": (
            gpu_peak_allocated_gb - profile["gpu_baseline_allocated_gb"]
            if profile["gpu_applicable"] else np.nan
        ),
        "gpu_peak_increment_reserved_gb": (
            gpu_peak_reserved_gb - profile["gpu_baseline_reserved_gb"]
            if profile["gpu_applicable"] else np.nan
        ),
        "memory_scope": (
            "process CUDA allocator reset immediately before stage; baseline, absolute peak, and increment reported"
            if profile["gpu_applicable"]
            else "CPU process RSS sampled every 0.05 s; GPU not applicable"
        ),
    }


def _record_timing(
    model_name, family, stage, elapsed_values, n_sentences, seed=None,
    resource_stats=None, model_info=None, batch_size=None,
    precision_setting=None, execution_device=None,
    stage_scope=None, preprocessing_included=None,
):
    values = np.asarray(elapsed_values, dtype=float)
    resource_stats = resource_stats or {}
    if model_info is not None:
        introspection = get_model_introspection_for_efficiency(model_name, model_info)
        size_note = (
            "deep models: torch parameter elements and exact parameter-tensor bytes; "
            "classical models: fitted predictor coefficient/intercept elements and serialized fitted pipeline"
        )
    else:
        introspection = {
            "n_parameters_total": np.nan,
            "n_parameters_trainable": np.nan,
            "parameter_count_method": "not_recomputed_see_training_row",
            "estimated_model_size_mb": np.nan,
            "model_size_method": "not_recomputed_see_training_row",
        }
        size_note = "architecture/storage fields are recorded on the corresponding training row"
    snapshot = get_hardware_snapshot()
    for repeat_index, elapsed_seconds in enumerate(values, start=1):
        revision_efficiency_repeat_records.append({
            "model": model_name,
            "family": family,
            "stage": stage,
            "seed": seed,
            "repeat_index": repeat_index,
            "n_sentences": int(n_sentences),
            "elapsed_seconds": float(elapsed_seconds),
            "seconds_per_1000_sentences": float(elapsed_seconds / n_sentences * 1000),
            "warmup_excluded_from_timing": stage == "inference",
        "stage_scope": stage_scope,
        "preprocessing_included": preprocessing_included,
        })
    revision_efficiency_records.append({
        "model": model_name,
        "family": family,
        "stage": stage,
        "seed": seed,
        "n_sentences": int(n_sentences),
        "n_repeats": int(len(values)),
        "elapsed_seconds_median": float(np.median(values)),
        "elapsed_seconds_q1": float(np.quantile(values, 0.25)),
        "elapsed_seconds_q3": float(np.quantile(values, 0.75)),
        "seconds_per_1000_sentences_median": float(np.median(values) / n_sentences * 1000),
        "execution_device": execution_device,
        "device_name": snapshot.get("gpu_name") if execution_device == "CUDA" else snapshot.get("processor"),
        "batch_size": batch_size,
        "precision_setting": precision_setting,
        "cpu_start_rss_gb": resource_stats.get("cpu_start_rss_gb", np.nan),
        "cpu_peak_rss_gb": resource_stats.get("cpu_peak_rss_gb", np.nan),
        "cpu_end_rss_gb": resource_stats.get("cpu_end_rss_gb", np.nan),
        "cpu_peak_increment_gb": resource_stats.get("cpu_peak_increment_gb", np.nan),
        "gpu_baseline_allocated_gb": resource_stats.get("gpu_baseline_allocated_gb", np.nan),
        "gpu_baseline_reserved_gb": resource_stats.get("gpu_baseline_reserved_gb", np.nan),
        "gpu_peak_allocated_gb": resource_stats.get("gpu_peak_allocated_gb", np.nan),
        "gpu_peak_reserved_gb": resource_stats.get("gpu_peak_reserved_gb", np.nan),
        "gpu_peak_increment_allocated_gb": resource_stats.get("gpu_peak_increment_allocated_gb", np.nan),
        "gpu_peak_increment_reserved_gb": resource_stats.get("gpu_peak_increment_reserved_gb", np.nan),
        "memory_scope": resource_stats.get("memory_scope", "not_measured"),
        "n_parameters_total": introspection.get("n_parameters_total", np.nan),
        "n_parameters_trainable": introspection.get("n_parameters_trainable", np.nan),
        "parameter_count_method": introspection.get("parameter_count_method"),
        "estimated_model_size_mb": introspection.get("estimated_model_size_mb", np.nan),
        "model_size_method": introspection.get("model_size_method"),
        "model_size_scope": size_note,
        "stage_scope": stage_scope,
        "preprocessing_included": preprocessing_included,
        "timing_comparability": "implementation-specific; stage scopes differ across families",
        "cpu_count": snapshot.get("cpu_count"),
        "total_ram_gb": snapshot.get("total_ram_gb"),
        "gpu_name": snapshot.get("gpu_name"),
        "cuda_version": snapshot.get("cuda_version"),
    })


def _repeat_inference(callable_fn, repeats=5):
    _ = callable_fn()  # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times, value = [], None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        value = callable_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(perf_counter() - start)
    return value, times


def _classical_input(model_name, estimator, df):
    if isinstance(estimator, PositionAwareTextClassifier):
        return df
    text_col = CLASSICAL_MODEL_SPECS[model_name]["text_col"]
    return df[[text_col]]


def _scores_and_classes(estimator, X):
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        score_type = "decision_function_margin_or_multinomial_score"
    elif hasattr(estimator, "predict_proba"):
        scores = np.log(np.clip(estimator.predict_proba(X), 1e-12, 1.0))
        score_type = "log_probability"
    else:
        raise RuntimeError(f"Continuous scores are required for {type(estimator).__name__}.")
    scores = np.asarray(scores, dtype=float)
    classes = [str(value) for value in estimator.classes_]
    if scores.ndim != 2 or scores.shape[1] != len(classes) or not np.isfinite(scores).all():
        raise RuntimeError("Estimator score matrix is non-finite or has an invalid shape.")
    return scores, classes, score_type


def _fit_final_transformer(model_key, mode, seed):
    from functools import partial

    set_seed(seed)
    model_spec = TRANSFORMER_MODELS[model_key]
    model_path = model_spec["repo_id"]
    revision = model_spec["revision"]
    tokenizer = _load_fast_transformer_tokenizer(model_path, revision)
    train_dataset, _, train_audit = build_transformer_dataset(
        full_train_sent, tokenizer, mode
    )
    dev_dataset, dev_source, dev_audit = build_transformer_dataset(dev_sent, tokenizer, mode)
    test_dataset, test_source, test_audit = build_transformer_dataset(test_sent, tokenizer, mode)
    if not np.array_equal(
        dev_source["label"].astype(str).to_numpy(), dev_sent["label"].astype(str).to_numpy()
    ):
        raise AssertionError("Transformer development ordering differs from the canonical frame.")
    if not np.array_equal(
        test_source["label"].astype(str).to_numpy(), final_test_y_true
    ):
        raise AssertionError("Transformer test ordering differs from the canonical frame.")
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, revision=revision, num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    args = make_transformer_training_arguments(
        TRANSFORMER_SCRATCH_ROOT / "final" / f"{model_key}_{mode}_seed{seed}",
        TRANSFORMER_EPOCHS_FINAL, TRANSFORMER_BATCH_SIZE, seed, evaluate=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    training_profile = _begin_stage_resource_profile(gpu_applicable=True)
    start = perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    training_elapsed = perf_counter() - start
    training_resources = _end_stage_resource_profile(training_profile)

    dev_prediction = trainer.predict(dev_dataset)
    inference_profile = _begin_stage_resource_profile(gpu_applicable=True)
    test_predict_callable = partial(
        predict_transformer_logits_direct,
        trainer.model, test_dataset, TRANSFORMER_EVAL_BATCH_SIZE,
    )
    test_scores, inference_times = _repeat_inference(
        test_predict_callable, repeats=INFERENCE_REPEATS
    )
    inference_resources = _end_stage_resource_profile(inference_profile)
    dev_scores = np.asarray(dev_prediction.predictions, dtype=float).copy()
    test_scores = np.asarray(test_scores, dtype=float).copy()
    if not np.isfinite(dev_scores).all() or not np.isfinite(test_scores).all():
        raise RuntimeError("Transformer score matrices contain non-finite values.")
    audit = pd.concat([
        train_audit.assign(split="train"),
        dev_audit.assign(split="development"),
        test_audit.assign(split="test"),
    ], ignore_index=True).assign(
        checkpoint=model_path, revision=revision, phase="locked_final",
        reference_seed=seed,
    )
    retained_model = trainer.model
    retained_model.to("cpu")
    del test_predict_callable
    del trainer
    del model, dev_prediction
    del train_dataset, dev_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "model": retained_model, "tokenizer": tokenizer,
        "dev_scores": dev_scores, "test_scores": test_scores,
        "score_classes": [ID2LABEL[i] for i in range(len(ID2LABEL))],
        "training_elapsed": training_elapsed, "inference_times": inference_times,
        "training_resources": training_resources,
        "inference_resources": inference_resources,
        "truncation_audit": audit,
        "checkpoint": model_path, "revision": revision,
    }

for system_name in FINAL_SYSTEMS:
    print("Final locked evaluation:", system_name)

    if system_name in CLASSICAL_MODEL_SPECS:
        estimator = clone(classical_best_estimators[system_name])
        training_profile = _begin_stage_resource_profile(gpu_applicable=False)
        start = perf_counter()
        estimator.fit(full_train_sent, full_train_sent["label"])
        training_elapsed = perf_counter() - start
        training_resources = _end_stage_resource_profile(training_profile)
        X_dev = _classical_input(system_name, estimator, dev_sent)
        X_test = _classical_input(system_name, estimator, test_sent)
        inference_profile = _begin_stage_resource_profile(gpu_applicable=False)
        test_pred, inference_times = _repeat_inference(
            lambda: estimator.predict(X_test), repeats=INFERENCE_REPEATS
        )
        inference_resources = _end_stage_resource_profile(inference_profile)
        dev_scores, score_classes, score_type = _scores_and_classes(estimator, X_dev)
        test_scores, test_score_classes, test_score_type = _scores_and_classes(estimator, X_test)
        if test_score_classes != score_classes or test_score_type != score_type:
            raise RuntimeError("Classical development/test score semantics differ.")
        final_test_predictions[system_name] = np.asarray(test_pred)
        final_test_predictions_by_seed[(system_name, "deterministic")] = np.asarray(test_pred)
        final_test_scores[system_name] = {"scores": test_scores, "classes": score_classes, "score_type": score_type}
        final_dev_scores[system_name] = {"scores": dev_scores, "classes": score_classes, "score_type": score_type, "y_true": dev_sent["label"].to_numpy()}
        final_fitted_systems[system_name] = {
            "family": "classical", "estimator": estimator,
            "text_col": CLASSICAL_MODEL_SPECS[system_name]["text_col"],
        }
        final_test_results.append({
            "model_name": system_name, "seed": "deterministic",
            **compute_metrics(final_test_y_true, test_pred),
        })
        _record_timing(
            system_name, "classical", "training", [training_elapsed], len(full_train_sent),
            seed="deterministic", resource_stats=training_resources,
            model_info=final_fitted_systems[system_name], batch_size=None,
            precision_setting="scikit-learn default float64 sparse/dense arithmetic",
            execution_device="CPU",
            stage_scope="TF-IDF fitting plus estimator fitting",
            preprocessing_included=True,
        )
        _record_timing(
            system_name, "classical", "inference", inference_times, len(test_sent),
            seed="deterministic", resource_stats=inference_resources,
            model_info=None, batch_size=None,
            precision_setting="scikit-learn default float64 sparse/dense arithmetic",
            execution_device="CPU",
            stage_scope="TF-IDF transformation plus estimator prediction; warm-up excluded",
            preprocessing_included=True,
        )
        _persist_final_stage(f"{system_name}__complete")
        continue

    if system_name == "S8_BiLSTMStyle_SequentialModel":
        for seed in REVISION_DEEP_SEEDS:
            training_profile = _begin_stage_resource_profile(gpu_applicable=True)
            start = perf_counter()
            model, vocab, history, train_dataset = train_bilstm_style_model(
                full_train_seq, epochs=BILSTM_EPOCHS_FINAL,
                batch_size=BILSTM_BATCH_SIZE, lr=1e-3, seed=seed,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_elapsed = perf_counter() - start
            training_resources = _end_stage_resource_profile(training_profile)
            dev_output = predict_bilstm_style_model(model, vocab, dev_sent, BILSTM_BATCH_SIZE)
            inference_profile = _begin_stage_resource_profile(gpu_applicable=True)
            test_output, inference_times = _repeat_inference(
                lambda: predict_bilstm_style_model(model, vocab, test_sent, BILSTM_BATCH_SIZE),
                repeats=INFERENCE_REPEATS,
            )
            inference_resources = _end_stage_resource_profile(inference_profile)
            if not np.array_equal(test_output["y_true"], final_test_y_true):
                raise AssertionError("BiLSTM test ordering does not match the canonical test order.")
            final_test_predictions_by_seed[(system_name, seed)] = test_output["y_pred"]
            final_test_results.append({
                "model_name": system_name, "seed": seed,
                **compute_metrics(test_output["y_true"], test_output["y_pred"]),
            })
            _record_timing(
                system_name, "bilstm", "training", [training_elapsed], len(full_train_seq), seed,
                resource_stats=training_resources,
                model_info={"family": "sequence_bilstm", "model": model},
                batch_size=BILSTM_BATCH_SIZE,
                precision_setting="float32; automatic mixed precision disabled",
                execution_device="CUDA",
                stage_scope="vocabulary construction, tensorisation, model construction, and training",
                preprocessing_included=True,
            )
            _record_timing(
                system_name, "bilstm", "inference", inference_times, len(test_sent), seed,
                resource_stats=inference_resources, model_info=None,
                batch_size=BILSTM_BATCH_SIZE,
                precision_setting="float32; automatic mixed precision disabled",
                execution_device="CUDA",
                stage_scope="tensorisation, batching, and model prediction; warm-up excluded",
                preprocessing_included=True,
            )
            if seed == REVISION_DEEP_SEEDS[0]:
                final_test_predictions[system_name] = test_output["y_pred"]
                final_test_scores[system_name] = {"scores": test_output["scores"], "classes": test_output["score_classes"], "score_type": "logit"}
                final_dev_scores[system_name] = {"scores": dev_output["scores"], "classes": dev_output["score_classes"], "score_type": "logit", "y_true": dev_output["y_true"]}
                model.to("cpu")
                final_fitted_systems[system_name] = {"family": "sequence_bilstm", "model": model, "vocab": vocab}
                bilstm_final_truncation_audits.extend([
                    _bilstm_audit_row(train_dataset, "train", "locked_final", seed),
                    _bilstm_audit_row(dev_output["dataset"], "development", "locked_final", seed),
                    _bilstm_audit_row(test_output["dataset"], "test", "locked_final", seed),
                ])
            else:
                model.to("cpu")
            _persist_final_stage(f"{system_name}__seed_{seed}")
            del model, vocab, history, train_dataset, dev_output, test_output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        continue

    if system_name.startswith("S9_"):
        model_key = "biomedbert" if "biomedbert" in system_name else "scibert" if "scibert" in system_name else "bert_base"
        mode = "prev_plus_current" if "prev_plus_current" in system_name else "sentence_only"
        for seed in REVISION_DEEP_SEEDS:
            output = _fit_final_transformer(model_key, mode, seed)
            test_pred = np.asarray([ID2LABEL[int(i)] for i in np.argmax(output["test_scores"], axis=1)])
            final_test_predictions_by_seed[(system_name, seed)] = test_pred
            final_test_results.append({
                "model_name": system_name, "seed": seed,
                **compute_metrics(final_test_y_true, test_pred),
            })
            _record_timing(
                system_name, "transformer", "training", [output["training_elapsed"]],
                len(full_train_sent), seed,
                resource_stats=output["training_resources"],
                model_info={"family": "transformer", "model": output["model"]},
                batch_size=TRANSFORMER_BATCH_SIZE,
                precision_setting="BF16 automatic mixed precision training; float32 master weights; TF32 enabled",
                execution_device="CUDA",
                stage_scope="fine-tuning only; checkpoint loading and tokenisation excluded",
                preprocessing_included=False,
            )
            _record_timing(
                system_name, "transformer", "inference", output["inference_times"],
                len(test_sent), seed,
                resource_stats=output["inference_resources"], model_info=None,
                batch_size=TRANSFORMER_BATCH_SIZE,
                precision_setting="float32 direct evaluation forward pass; fixed padding; TF32 enabled",
                execution_device="CUDA",
                stage_scope="prepared-dataset forward passes; tokenisation excluded; warm-up excluded",
                preprocessing_included=False,
            )
            transformer_final_truncation_audits.append(
                output["truncation_audit"].assign(model=system_name)
            )
            if seed == REVISION_DEEP_SEEDS[0]:
                final_test_predictions[system_name] = test_pred
                final_test_scores[system_name] = {"scores": output["test_scores"], "classes": output["score_classes"], "score_type": "logit"}
                final_dev_scores[system_name] = {"scores": output["dev_scores"], "classes": output["score_classes"], "score_type": "logit", "y_true": dev_sent["label"].to_numpy()}
                final_fitted_systems[system_name] = {
                    "family": "transformer", "model": output["model"],
                    "tokenizer": output["tokenizer"], "model_key": model_key,
                    "mode": mode, "max_length": TRANSFORMER_MAX_LENGTH,
                    "truncation_audit": output["truncation_audit"],
                    "checkpoint": output["checkpoint"], "revision": output["revision"],
                }
                torch.cuda.empty_cache()
            _persist_final_stage(f"{system_name}__seed_{seed}")
            del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        continue

    raise RuntimeError(f"Unrecognised locked final system: {system_name}")

final_test_results_df = pd.DataFrame(final_test_results)
deep_seed_summary_df = (
    final_test_results_df[
        final_test_results_df["model_name"].str.startswith(("S8_", "S9_"))
    ].groupby("model_name")
    .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
         accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), n_seeds=("seed", "count"))
    .reset_index()
)
revision_efficiency_df = pd.DataFrame(revision_efficiency_records)
revision_efficiency_repeats_df = pd.DataFrame(revision_efficiency_repeat_records)
if set(final_test_predictions) != set(FINAL_SYSTEMS):
    missing = sorted(set(FINAL_SYSTEMS) - set(final_test_predictions))
    extra = sorted(set(final_test_predictions) - set(FINAL_SYSTEMS))
    raise RuntimeError(f"Final prediction coverage mismatch; missing={missing}, extra={extra}")
if set(final_test_scores) != set(FINAL_SYSTEMS) or set(final_dev_scores) != set(FINAL_SYSTEMS):
    raise RuntimeError("Final development/test score coverage differs from FINAL_SYSTEMS.")
if set(final_fitted_systems) != set(FINAL_SYSTEMS):
    raise RuntimeError("Retained fitted-system coverage differs from FINAL_SYSTEMS.")

transformer_truncation_audit_df = pd.concat(
    [transformer_truncation_audit_df] + transformer_final_truncation_audits,
    ignore_index=True,
)
if bilstm_final_truncation_audits:
    bilstm_truncation_audit_df = pd.concat([
        bilstm_truncation_audit_df,
        pd.DataFrame(bilstm_final_truncation_audits).assign(model=BILSTM_MODEL),
    ], ignore_index=True)
for audit_name, audit_frame in [
    ("transformer", transformer_truncation_audit_df),
    ("bilstm", bilstm_truncation_audit_df),
]:
    audit_keys = ["model", "phase", "split", "reference_seed"]
    missing_keys = [key for key in audit_keys if key not in audit_frame.columns]
    if missing_keys or audit_frame.duplicated(audit_keys).any():
        raise RuntimeError(
            f"{audit_name} truncation audit has missing keys or duplicate rows: {missing_keys}"
        )
if revision_efficiency_repeats_df.empty:
    raise RuntimeError("Raw efficiency repeat table is empty.")
required_efficiency_columns = [
    "execution_device", "device_name", "batch_size", "precision_setting",
    "cpu_start_rss_gb", "cpu_peak_rss_gb", "cpu_peak_increment_gb",
    "gpu_baseline_allocated_gb", "gpu_peak_allocated_gb",
    "gpu_peak_increment_allocated_gb", "memory_scope",
    "n_parameters_total", "n_parameters_trainable", "parameter_count_method",
    "estimated_model_size_mb", "model_size_method", "model_size_scope",
    "stage_scope", "preprocessing_included", "timing_comparability",
]
missing_efficiency_columns = [
    column for column in required_efficiency_columns
    if column not in revision_efficiency_df.columns
]
if missing_efficiency_columns:
    raise RuntimeError(f"Efficiency export is missing required fields: {missing_efficiency_columns}")
training_efficiency_rows = revision_efficiency_df[
    revision_efficiency_df["stage"].astype(str).eq("training")
].copy()
if PUBLICATION_RUN:
    if training_efficiency_rows.empty:
        raise RuntimeError("Publication efficiency export has no training rows.")
    training_size_fields = training_efficiency_rows[[
        "n_parameters_total", "n_parameters_trainable", "estimated_model_size_mb"
    ]].apply(pd.to_numeric, errors="coerce")
    invalid_training_size = (
        training_size_fields.isna().any(axis=1)
        | (training_size_fields <= 0).any(axis=1)
    )
    missing_training_methods = (
        training_efficiency_rows[["parameter_count_method", "model_size_method"]]
        .fillna("").astype(str).apply(lambda column: column.str.strip().eq(""))
        .any(axis=1)
    )
    invalid_training_size = invalid_training_size | missing_training_methods
    if invalid_training_size.any():
        bad_models = training_efficiency_rows.loc[
            invalid_training_size, ["model", "seed"]
        ].to_dict("records")
        raise RuntimeError(
            "Publication training rows lack positive parameter/storage evidence: "
            f"{bad_models}"
        )
if PUBLICATION_RUN and revision_efficiency_df[[
    "cpu_start_rss_gb", "cpu_peak_rss_gb", "cpu_peak_increment_gb"
]].isna().any().any():
    raise RuntimeError("Publication CPU RSS measurements contain missing values.")
deep_rows = revision_efficiency_df[revision_efficiency_df["execution_device"] == "CUDA"]
if PUBLICATION_RUN and deep_rows[[
    "gpu_baseline_allocated_gb", "gpu_peak_allocated_gb", "gpu_peak_increment_allocated_gb"
]].isna().any().any():
    raise RuntimeError("Publication CUDA memory measurements contain missing values.")
REFERENCE_PREDICTION_SEED = {
    model_name: ("deterministic" if model_name in CLASSICAL_MODEL_SPECS else REFERENCE_SEED)
    for model_name in final_test_predictions
}
display(final_test_results_df)
display(deep_seed_summary_df)
display(revision_efficiency_df)

_persist_final_stage("all_final_systems_complete")

# ============================================================
# PLOS REVISION: ABSTRACT-CLUSTER BOOTSTRAP
# ============================================================
# Complete abstracts, rather than individual sentences, are resampled so the
# within-abstract dependence structure is preserved. The reviewer-specified
# 10,000 replicates are retained; draws are processed in small bounded batches.

BOOTSTRAP_N_REPLICATES = 10000
BOOTSTRAP_RANDOM_STATE = SEED
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_BATCH_SIZE = 64
BOOT_LABELS = list(LABEL_ORDER)
BOOT_LABEL_TO_ID = {label: i for i, label in enumerate(BOOT_LABELS)}


def _confusion_batch_to_macro_f1(confusions):
    cms = np.asarray(confusions, dtype=np.float64)
    if cms.ndim == 2:
        cms = cms[None, ...]
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    precision_den = cms.sum(axis=-2)
    recall_den = cms.sum(axis=-1)
    precision = np.divide(
        tp, precision_den, out=np.zeros_like(tp), where=precision_den > 0
    )
    recall = np.divide(
        tp, recall_den, out=np.zeros_like(tp), where=recall_den > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return f1.mean(axis=-1)


def _confusion_to_macro_f1(cm):
    return float(_confusion_batch_to_macro_f1(cm)[0])


def _abstract_confusions(y_true, y_pred, pubmed_ids):
    abstract_codes, abstract_ids = pd.factorize(pd.Series(pubmed_ids), sort=False)
    true_codes = pd.Categorical(y_true, categories=BOOT_LABELS).codes
    pred_codes = pd.Categorical(y_pred, categories=BOOT_LABELS).codes
    if (abstract_codes < 0).any() or (true_codes < 0).any() or (pred_codes < 0).any():
        raise AssertionError("Bootstrap inputs contain missing IDs or labels outside LABEL_ORDER.")
    matrices = np.zeros(
        (len(abstract_ids), len(BOOT_LABELS), len(BOOT_LABELS)), dtype=np.int32
    )
    np.add.at(matrices, (abstract_codes, true_codes, pred_codes), 1)
    return np.asarray(abstract_ids), matrices


def _batched_cluster_bootstrap_scores(
    matrices, n_bootstrap=10000, seed=42, batch_size=BOOTSTRAP_BATCH_SIZE
):
    n_abstracts = int(matrices.shape[0])
    if n_abstracts <= 0:
        raise AssertionError("Cluster bootstrap requires at least one abstract.")
    if int(n_bootstrap) <= 0 or int(batch_size) <= 0:
        raise ValueError("n_bootstrap and batch_size must be positive.")
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_abstracts, 1.0 / n_abstracts, dtype=np.float64)
    flat_confusions = matrices.reshape(n_abstracts, -1).astype(np.float32, copy=False)
    scores = np.empty(int(n_bootstrap), dtype=np.float64)
    for start in range(0, int(n_bootstrap), int(batch_size)):
        stop = min(start + int(batch_size), int(n_bootstrap))
        weights = rng.multinomial(
            n_abstracts, probabilities, size=stop - start
        ).astype(np.float32, copy=False)
        aggregated = (weights @ flat_confusions).reshape(
            stop - start, len(BOOT_LABELS), len(BOOT_LABELS)
        )
        scores[start:stop] = _confusion_batch_to_macro_f1(aggregated)
    return scores


def cluster_bootstrap_macro_f1(
    y_true, y_pred, pubmed_ids, n_bootstrap=10000, seed=42
):
    _, matrices = _abstract_confusions(y_true, y_pred, pubmed_ids)
    return _batched_cluster_bootstrap_scores(
        matrices, n_bootstrap=n_bootstrap, seed=seed
    )


bootstrap_summary_rows = []
bootstrap_distributions = {}

for model_name, prediction in final_test_predictions.items():
    # Identical seed and abstract order give each model the same cluster draws.
    scores = cluster_bootstrap_macro_f1(
        final_test_y_true,
        np.asarray(prediction),
        final_test_pubmed_ids,
        n_bootstrap=BOOTSTRAP_N_REPLICATES,
        seed=BOOTSTRAP_RANDOM_STATE,
    )
    bootstrap_distributions[model_name] = scores
    observed = f1_score(
        final_test_y_true, prediction, labels=LABEL_ORDER,
        average="macro", zero_division=0
    )
    bootstrap_summary_rows.append({
        "Model": model_name,
        "Macro-F1": observed,
        "Bootstrap median Macro-F1": float(np.median(scores)),
        "95% CI low": float(np.quantile(scores, BOOTSTRAP_ALPHA / 2)),
        "95% CI high": float(np.quantile(scores, 1 - BOOTSTRAP_ALPHA / 2)),
        "n_test_sentences": int(len(final_test_y_true)),
        "n_test_abstracts": int(pd.Series(final_test_pubmed_ids).nunique()),
        "n_cluster_bootstrap": BOOTSTRAP_N_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
        "bootstrap_batch_size": BOOTSTRAP_BATCH_SIZE,
        "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
        "ci_method": "95% percentile complete-abstract cluster bootstrap",
        "uncertainty_scope": "conditional on the fixed reference prediction vector",
    })

bootstrap_summary_df = pd.DataFrame(bootstrap_summary_rows).sort_values(
    "Macro-F1", ascending=False
)
display(bootstrap_summary_df)

# ============================================================
# PRELOCKED PAIRED ABSTRACT-CLUSTER COMPARISONS
# ============================================================
if _sha256_file(PRETEST_LOCK_JSON) != PRETEST_SELECTION_LOCK_SHA256:
    raise RuntimeError("Pre-test lock changed before paired comparisons.")
if int(pretest_comparison_lock_df["formal_inference_allowed"].sum()) != 1:
    raise AssertionError("Exactly one primary comparison must remain locked.")


def paired_cluster_bootstrap_delta(model_a, model_b):
    # Cell 102 uses the same complete-abstract order, seed, and multinomial
    # stream for every model. Reusing those already-computed marginal
    # distributions makes the paired contrast exact and avoids four
    # redundant 10,000-replicate matrix multiplications.
    if model_a not in bootstrap_distributions or model_b not in bootstrap_distributions:
        raise AssertionError(
            f"Required final bootstrap distribution missing: {model_a}, {model_b}"
        )
    scores_a = np.asarray(bootstrap_distributions[model_a], dtype=np.float64)
    scores_b = np.asarray(bootstrap_distributions[model_b], dtype=np.float64)
    if (
        scores_a.shape != (BOOTSTRAP_N_REPLICATES,)
        or scores_b.shape != (BOOTSTRAP_N_REPLICATES,)
        or not np.isfinite(scores_a).all()
        or not np.isfinite(scores_b).all()
    ):
        raise RuntimeError(
            f"Invalid marginal bootstrap distributions for {model_a}/{model_b}."
        )
    y = np.asarray(final_test_y_true)
    a = np.asarray(final_test_predictions[model_a])
    b = np.asarray(final_test_predictions[model_b])
    observed = (
        f1_score(y, a, labels=LABEL_ORDER, average="macro", zero_division=0)
        - f1_score(y, b, labels=LABEL_ORDER, average="macro", zero_division=0)
    )
    deltas = scores_a - scores_b
    return observed, deltas


paired_rows = []
paired_bootstrap_distribution_parts = []
for model_a, model_b, family, analysis_role, formal_inference_allowed in comparison_pairs:
    if model_a not in final_test_predictions or model_b not in final_test_predictions:
        raise AssertionError(f"Required final prediction missing: {model_a}, {model_b}")
    observed, deltas = paired_cluster_bootstrap_delta(model_a, model_b)
    paired_bootstrap_distribution_parts.append(pd.DataFrame({
        "Model A": model_a,
        "Model B": model_b,
        "Comparison family": family,
        "analysis_role": analysis_role,
        "formal_inference_allowed": formal_inference_allowed,
        "replicate": np.arange(len(deltas)),
        "delta_macro_f1_a_minus_b": deltas,
        "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
        "reference_seed_a": REFERENCE_PREDICTION_SEED[model_a],
        "reference_seed_b": REFERENCE_PREDICTION_SEED[model_b],
    }))
    paired_rows.append({
        "Model A": model_a,
        "Model B": model_b,
        "Comparison family": family,
        "analysis_role": analysis_role,
        "formal_inference_allowed": formal_inference_allowed,
        "Delta Macro-F1 (A-B)": observed,
        "Delta Macro-F1 percentage points (A-B)": 100.0 * observed,
        "95% CI low": float(np.quantile(deltas, BOOTSTRAP_ALPHA / 2)),
        "95% CI high": float(np.quantile(deltas, 1 - BOOTSTRAP_ALPHA / 2)),
        "bootstrap_proportion_delta_gt_0": float(np.mean(deltas > 0)),
        "bootstrap_proportion_interpretation": "resampling summary; not a p-value or posterior probability",
        "n_test_sentences": int(len(final_test_y_true)),
        "n_test_abstracts": int(pd.Series(final_test_pubmed_ids).nunique()),
        "n_cluster_bootstrap": BOOTSTRAP_N_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
        "bootstrap_batch_size": BOOTSTRAP_BATCH_SIZE,
        "ci_alpha": BOOTSTRAP_ALPHA,
        "ci_method": "two-sided percentile complete-abstract cluster bootstrap",
        "multiplicity_policy": (
            "sole prespecified primary contrast; no multiplicity adjustment"
            if formal_inference_allowed
            else "secondary descriptive contrast; no formal significance claim"
        ),
        "uncertainty_scope": "conditional on fixed seed-42 predictions; selection/refitting uncertainty excluded",
        "reference_seed_a": REFERENCE_PREDICTION_SEED[model_a],
        "reference_seed_b": REFERENCE_PREDICTION_SEED[model_b],
    })

paired_bootstrap_comparison_df = pd.DataFrame(paired_rows)
if int(paired_bootstrap_comparison_df["formal_inference_allowed"].sum()) != 1:
    raise AssertionError("Paired output must contain exactly one primary comparison.")
paired_bootstrap_distributions_df = pd.concat(
    paired_bootstrap_distribution_parts, ignore_index=True
)
display(paired_bootstrap_comparison_df)

# ============================================================
# DURABLE FINAL COMPLETE-ABSTRACT BOOTSTRAP CHECKPOINT
# ============================================================
bootstrap_long = pd.concat([
    pd.DataFrame({
        "model": model_name,
        "replicate": np.arange(len(values)),
        "macro_f1": values,
        "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
        "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
    })
    for model_name, values in bootstrap_distributions.items()
], ignore_index=True)

if set(bootstrap_distributions) != set(FINAL_SYSTEMS):
    raise RuntimeError("Unpaired final bootstrap coverage differs from FINAL_SYSTEMS.")
if bootstrap_long.groupby("model").size().to_dict() != {
    name: BOOTSTRAP_N_REPLICATES for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Unpaired final bootstrap distributions are incomplete.")
paired_stage_counts = paired_bootstrap_distributions_df.groupby(
    ["Model A", "Model B", "Comparison family"]
).size()
if (
    len(paired_stage_counts) != len(pretest_comparison_lock_df)
    or not paired_stage_counts.eq(BOOTSTRAP_N_REPLICATES).all()
):
    raise RuntimeError("Paired final bootstrap distributions are incomplete.")
if int(paired_bootstrap_comparison_df["formal_inference_allowed"].sum()) != 1:
    raise RuntimeError("Exactly one prespecified comparison may permit formal inference.")

bootstrap_stage_artifacts = []
for filename, frame in [
    ("final_test_cluster_bootstrap_summary.csv", bootstrap_summary_df),
    ("abstract_cluster_bootstrap_distributions.csv", bootstrap_long),
    ("final_test_paired_cluster_bootstrap.csv", paired_bootstrap_comparison_df),
    (
        "final_test_paired_cluster_bootstrap_distributions.csv",
        paired_bootstrap_distributions_df,
    ),
]:
    bootstrap_stage_artifacts.append(_stage_artifact_record(
        _atomic_stage_csv(frame, ANALYSIS_STAGE_DIR / filename),
        len(frame),
    ))

final_bootstrap_stage_contract = {
    "contract_version": "plos_analysis_stage_v1",
    "stage": "final_complete_abstract_bootstrap_complete",
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "reference_seed": REFERENCE_SEED,
    "n_models": len(FINAL_SYSTEMS),
    "n_comparisons": len(paired_bootstrap_comparison_df),
    "n_primary_comparisons": int(
        paired_bootstrap_comparison_df["formal_inference_allowed"].sum()
    ),
    "bootstrap_replicates_per_model_or_comparison": BOOTSTRAP_N_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
    "bootstrap_batch_size": BOOTSTRAP_BATCH_SIZE,
    "sampling_unit": "complete PubMed abstract",
    "artifacts": bootstrap_stage_artifacts,
}
_atomic_stage_json(
    final_bootstrap_stage_contract,
    ANALYSIS_STAGE_DIR / "FINAL_BOOTSTRAP_STAGE_COMPLETE.json",
)
print(
    "Durable final bootstrap checkpoint:",
    ANALYSIS_STAGE_DIR / "FINAL_BOOTSTRAP_STAGE_COMPLETE.json",
)

# ============================================================
# SAMPLE SIZE / PRECISION INTERPRETATION FOR THE FINAL TEST SET
# ============================================================

if final_test_y_true is None:
    final_test_class_counts_df = pd.DataFrame()
    print("Final test-set class summary is not available yet.")
else:
    final_test_class_counts_df = (
        pd.Series(final_test_y_true)
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
        .rename_axis("label")
        .reset_index(name="count")
    )
    final_test_class_counts_df["proportion"] = final_test_class_counts_df["count"] / final_test_class_counts_df["count"].sum()

    final_test_n = int(len(final_test_y_true))
    display(final_test_class_counts_df)

    minority_row = final_test_class_counts_df.sort_values("count", ascending=True).iloc[0]
    majority_row = final_test_class_counts_df.sort_values("count", ascending=False).iloc[0]

    print(f"Fixed public benchmark test-set size: N = {final_test_n}")
    print(
        f"The most frequent class is '{majority_row['label']}' (n={int(majority_row['count'])}), while the least frequent class is "
        f"'{minority_row['label']}' (n={int(minority_row['count'])})."
    )
    print(
        "In practical reporting terms, broader confidence intervals are more likely for minority classes because fewer "
        "test examples contribute to those estimates. This is especially relevant for macro-averaged metrics, which give "
        "equal weight to each class regardless of frequency."
    )

# Implementation-specific timing/resource summary.
revision_efficiency_df = pd.DataFrame(revision_efficiency_records)
display(revision_efficiency_df)
print(
    "The protocol requires one clean restart-and-run-all. Each stage resets/samples process baselines, but stages "
    "are not isolated in separate processes. Use stage_scope and preprocessing_included; "
    "do not interpret cross-family timing as hardware-independent algorithmic speed."
)

# ============================================================
# DATASET INTEGRITY, PROVENANCE, ORDER, AND CLASS COUNTS
# ============================================================
import hashlib

if not FINAL_SYSTEMS_LOCKED or not TEST_OUTCOMES_ACCESSED:
    raise RuntimeError("Dataset result exports must follow final-system locking and test evaluation.")

REVISION_OUTPUT_DIR = Path(OUTPUT_DIR) / "plos_revision_outputs"
REVISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


split_frames = {"train": train_df, "development": dev_df, "test": test_df}
integrity_rows = []
class_count_rows = []
for split_name, frame in split_frames.items():
    raw_order_failures = RAW_INPUT_ORDER_AUDIT[split_name]
    integrity_rows.append({
        "split": split_name,
        "n_sentences": int(len(frame)),
        "n_abstracts": int(frame["pubmed_id"].nunique()),
        "duplicate_pubmed_file_order": int(frame.duplicated(["pubmed_id", "file_order"]).sum()),
        "duplicate_pubmed_text_label": int(frame.duplicated(["pubmed_id", "text", "label"]).sum()),
        "unexpected_labels": "|".join(sorted(set(frame["label"]) - set(LABEL_ORDER))),
        "raw_input_abstract_order_failures": raw_order_failures,
    })
    counts = frame["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    for label, count in counts.items():
        class_count_rows.append({
            "split": split_name, "class": label, "count": int(count),
            "proportion": float(count / len(frame)),
        })

id_sets = {name: set(frame["pubmed_id"].astype(str)) for name, frame in split_frames.items()}
split_overlap_rows = [
    {
        "split_a": left, "split_b": right,
        "overlapping_abstract_ids": len(id_sets[left].intersection(id_sets[right])),
    }
    for left, right in [
        ("train", "development"), ("train", "test"), ("development", "test")
    ]
]
dataset_integrity_df = pd.DataFrame(integrity_rows)
dataset_split_overlap_df = pd.DataFrame(split_overlap_rows)
dataset_split_class_counts_df = pd.DataFrame(class_count_rows)
dataset_hash_df = pd.DataFrame([
    {
        "split": split_name, "path_used_for_run": str(path),
        "sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
    }
    for split_name, path in [
        ("train", train_path), ("development", dev_path), ("test", test_path)
    ]
])

if dataset_integrity_df["duplicate_pubmed_file_order"].sum() != 0:
    raise RuntimeError("Duplicate PubMedID/file-order keys detected.")
if not dataset_integrity_df["unexpected_labels"].eq("").all():
    raise RuntimeError("Unexpected labels detected.")
if dataset_integrity_df["raw_input_abstract_order_failures"].sum() != 0:
    raise RuntimeError("Raw within-abstract file order is not strictly increasing.")
if dataset_split_overlap_df["overlapping_abstract_ids"].sum() != 0:
    raise RuntimeError("Abstract identifiers overlap across official splits.")

dataset_integrity_df.to_csv(REVISION_OUTPUT_DIR / "dataset_integrity.csv", index=False)
dataset_split_overlap_df.to_csv(
    REVISION_OUTPUT_DIR / "dataset_split_abstract_overlap.csv", index=False
)
dataset_hash_df.to_csv(REVISION_OUTPUT_DIR / "dataset_file_sha256.csv", index=False)
dataset_split_class_counts_df.to_csv(
    REVISION_OUTPUT_DIR / "dataset_split_class_counts.csv", index=False
)
display(dataset_integrity_df)
display(dataset_split_overlap_df)
display(dataset_split_class_counts_df)

# ============================================================
# CONSOLIDATED MODEL, DECISION, TUNING, AND SOFTWARE SPECIFICATION
# ============================================================
EXPECTED_TRANSFORMER_REPOSITORIES = {
    "BERT": "google-bert/bert-base-uncased",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BiomedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
}
if {spec["repo_id"] for spec in TRANSFORMER_MODELS.values()} != set(
    EXPECTED_TRANSFORMER_REPOSITORIES.values()
):
    raise RuntimeError("Transformer repository registry differs from the frozen specification.")
model_reproducibility_rows = []


def add_classical_spec(model_id, estimator_name, multiclass_strategy, grid, class_weight,
                       preprocessing, decision_rule):
    model_reproducibility_rows.append({
        "model_id": model_id, "model": estimator_name, "family": "classical",
        "checkpoint": "not_applicable", "revision": "not_applicable",
        "input_mode": "current sentence",
        "input_and_preprocessing": preprocessing,
        "features": "word TF-IDF; position variables only for S6",
        "multiclass_strategy": multiclass_strategy,
        "hyperparameter_grid": json.dumps(grid, sort_keys=True),
        "selected_hyperparameters": json.dumps(
            classical_best_params.get(model_id, {}), sort_keys=True
        ),
        "selection": "outer 5-fold grouped estimate; inner 4-fold tuning; separate 4-fold full-training refit search; Macro-F1",
        "class_weight": class_weight,
        "decision_rule": decision_rule,
        "threshold_tuning": "none",
        "first_sentence_context": "not_applicable",
        "seeds": "42; deterministic estimator fit",
    })


add_classical_spec(
    "S0_TFIDF_LogReg", "LogisticRegression", "native five-class softmax (lbfgs)",
    CLASSICAL_MODEL_SPECS["S0_TFIDF_LogReg"]["param_grid"],
    str(classical_best_params.get("S0_TFIDF_LogReg", {}).get("clf__class_weight")),
    "Unicode; lowercase; token_pattern (?u) word tokens length >=2; word 1-2 grams; min_df=1; smoothed L2 TF-IDF",
    "maximum class probability (argmax-equivalent to largest multinomial score)",
)
add_classical_spec(
    "S1_TFIDF_LinearSVM", "LinearSVC", "one-vs-rest",
    CLASSICAL_MODEL_SPECS["S1_TFIDF_LinearSVM"]["param_grid"], "none",
    "same word TF-IDF as S0",
    "largest one-vs-rest decision score",
)
add_classical_spec(
    "S1b_TFIDF_LinearSVM_OVO", "OneVsOneClassifier(LinearSVC)",
    "one-vs-one sensitivity analysis; 10 binary pair classifiers",
    CLASSICAL_MODEL_SPECS["S1b_TFIDF_LinearSVM_OVO"]["param_grid"], "none",
    "same word TF-IDF as S0",
    "OneVsOneClassifier vote/confidence tie-break; no threshold",
)
add_classical_spec(
    "S2_TFIDF_MNB", "MultinomialNB", "native five-class MultinomialNB",
    CLASSICAL_MODEL_SPECS["S2_TFIDF_MNB"]["param_grid"], "not supported/none",
    "same nonnegative word TF-IDF as S0",
    "largest posterior log-probability",
)
add_classical_spec(
    "S5_NumberNormalized_LogReg", "LogisticRegression", "native five-class softmax (lbfgs)",
    CLASSICAL_MODEL_SPECS["S5_NumberNormalized_LogReg"]["param_grid"],
    str(classical_best_params.get("S5_NumberNormalized_LogReg", {}).get("clf__class_weight")),
    "each digit replaced by 0, followed by the S0 TF-IDF pipeline",
    "maximum class probability (argmax-equivalent to largest multinomial score)",
)
add_classical_spec(
    "S6_PositionAware_LogReg", "PositionAwareTextClassifier(LogisticRegression)",
    "native five-class softmax (lbfgs)",
    CLASSICAL_MODEL_SPECS["S6_PositionAware_LogReg"]["param_grid"],
    str(classical_best_params.get("S6_PositionAware_LogReg", {}).get("class_weight")),
    "S0 TF-IDF plus standardised index, relative position, first-sentence, and last-sentence variables",
    "maximum class probability (argmax-equivalent to largest multinomial score)",
)

model_reproducibility_rows.extend([
    {
        "model_id": "S7_CRF_SequenceModel", "model": "sklearn_crfsuite.CRF",
        "family": "sequence_crf", "checkpoint": "not_applicable", "revision": "sklearn-crfsuite==0.5.0",
        "input_mode": "complete ordered abstract",
        "input_and_preprocessing": "feature values derived from lowercased text; order by PubMed ID and physical file order",
        "features": "word count, sentence position, first/last word, prefix, numeric flag, adjacent first words, BOS/EOS",
        "multiclass_strategy": "global linear-chain sequence decoding",
        "hyperparameter_grid": "none; fixed development-only comparator",
        "selected_hyperparameters": "algorithm=lbfgs;c1=0.1;c2=0.1;max_iterations=100;all_possible_transitions=True",
        "selection": "development-only; ineligible for final test selection", "class_weight": "none",
        "decision_rule": "highest-scoring label sequence", "threshold_tuning": "none",
        "first_sentence_context": "BOS feature; no previous sentence", "seeds": "deterministic library fit",
    },
    {
        "model_id": "S8_BiLSTMStyle_SequentialModel", "model": "masked packed BiLSTM",
        "family": "sequence_bilstm", "checkpoint": "not_applicable", "revision": "not_applicable",
        "input_mode": "complete ordered abstract",
        "input_and_preprocessing": f"lowercase whitespace tokens; min_freq=2; max {MAX_SENTENCE_TOKENS} tokens; training-derived abstract cap {MAX_ABSTRACT_SENTENCES}",
        "features": f"learned {BILSTM_EMBEDDING_DIM}-d embeddings; masked mean; packed bidirectional LSTM hidden={BILSTM_HIDDEN_DIM}",
        "multiclass_strategy": "single five-class output layer",
        "hyperparameter_grid": "none; fixed before development evaluation",
        "selected_hyperparameters": f"Adam lr={BILSTM_LEARNING_RATE}; betas=.9,.999; eps=1e-8; weight_decay=0; grad_clip={BILSTM_MAX_GRAD_NORM}; batch={BILSTM_BATCH_SIZE}; epochs={BILSTM_EPOCHS_FINAL}",
        "selection": "mandatory final sequence comparator by protocol; development Macro-F1 descriptive; final seeds 42,43,44", "class_weight": "none",
        "decision_rule": "argmax logits", "threshold_tuning": "none",
        "first_sentence_context": "sequence begins at first sentence; no previous-label input",
        "seeds": "42,43,44; reference_seed=42",
    },
])

for model_key, model_spec in TRANSFORMER_MODELS.items():
    for input_mode in ["sentence_only", "prev_plus_current"]:
        model_reproducibility_rows.append({
            "model_id": f"S9_{model_key}_{input_mode}",
            "model": model_spec["display_name"], "family": "transformer",
            "checkpoint": model_spec["repo_id"], "revision": model_spec["revision"],
            "input_mode": input_mode,
            "input_and_preprocessing": f"immutable checkpoint fast AutoTokenizer/Tokenizers backend; max_length={TRANSFORMER_MAX_LENGTH}; fixed length-256 padding; previous context truncated before current target; first sentence uses full single-input budget",
            "features": "pretrained contextual representation plus randomly initialised five-class head",
            "multiclass_strategy": "single five-class softmax/logit head",
            "hyperparameter_grid": "none; fixed settings across checkpoints and input modes",
            "selected_hyperparameters": f"fused AdamW lr=2e-5; weight_decay=.01; linear schedule; warmup=0; grad_accum={TRANSFORMER_GRADIENT_ACCUMULATION_STEPS}; max_grad_norm=1; train_batch={TRANSFORMER_BATCH_SIZE}; eval_batch={TRANSFORMER_EVAL_BATCH_SIZE}; epochs={TRANSFORMER_EPOCHS_FINAL}; fp16={TRANSFORMER_USE_FP16}; bf16={TRANSFORMER_USE_BF16}; tf32={TRANSFORMER_USE_TF32}",
            "selection": "development Macro-F1; final seeds 42,43,44", "class_weight": "none",
            "decision_rule": "argmax logits", "threshold_tuning": "none",
            "first_sentence_context": (
                "empty previous string; first sentence encoded alone"
                if input_mode == "prev_plus_current" else "not_applicable; current sentence only"
            ),
            "seeds": "42,43,44; reference_seed=42",
        })

model_reproducibility_df = pd.DataFrame(model_reproducibility_rows)
model_reproducibility_df.to_csv(
    REVISION_OUTPUT_DIR / "model_reproducibility_specification.csv", index=False
)
software_hardware_df = (
    pd.DataFrame([ENVIRONMENT_SNAPSHOT]).T.rename(columns={0: "value"})
    .reset_index(names="field")
)
software_hardware_df.to_csv(
    REVISION_OUTPUT_DIR / "software_hardware_environment.csv", index=False
)
display(model_reproducibility_df)

# ============================================================
# COMPLEMENTARY METRICS, MULTICLASS CALIBRATION, AND FIGURE HELPERS
# ============================================================
# Calibration comparisons are descriptive point estimates unless a separately
# prespecified cluster interval is reported. Temperature is fitted on development
# scores after selection and is applied unchanged to the fixed benchmark test set.

from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

CALIBRATION_BINS = 15
CALIBRATION_TEMPERATURE_BOUNDS = (0.05, 20.0)
METRIC_LABELS = list(LABEL_ORDER)
METRIC_LABEL_TO_ID = {label: i for i, label in enumerate(METRIC_LABELS)}
PLOS_FIGURE_FONT = "Arial"
from matplotlib import font_manager

PLOS_FONT_PATH = os.environ.get("PLOS_FONT_PATH", "").strip()
if PLOS_FONT_PATH:
    supplied_font = Path(PLOS_FONT_PATH).expanduser()
    if not supplied_font.is_file():
        raise FileNotFoundError(f"PLOS_FONT_PATH does not identify a font file: {supplied_font}")
    font_manager.fontManager.addfont(str(supplied_font))
try:
    PLOS_FIGURE_FONT_FILE = font_manager.findfont(
        PLOS_FIGURE_FONT, fallback_to_default=False
    )
except ValueError as exc:
    raise RuntimeError(
        "Arial is required for the publication figures but is unavailable in this runtime. "
        "Install a licensed Arial font or set PLOS_FONT_PATH to an Arial .ttf file, restart, "
        "and Run all. The figure gate does not permit silent font fallback."
    ) from exc
plt.rcParams.update({
    "font.family": PLOS_FIGURE_FONT,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


def _align_scores(scores, source_classes):
    scores = np.asarray(scores, dtype=float)
    source_classes = [str(value) for value in source_classes]
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError(f"Score matrix must be non-empty and two-dimensional; got {scores.shape}.")
    if len(source_classes) != scores.shape[1] or len(set(source_classes)) != len(source_classes):
        raise ValueError("Score classes do not uniquely match score columns.")
    if set(source_classes) != set(METRIC_LABELS):
        raise ValueError(f"Score classes differ from the fixed label set: {source_classes}")
    if not np.isfinite(scores).all():
        raise ValueError("Score matrix contains non-finite values.")
    return scores[:, [source_classes.index(label) for label in METRIC_LABELS]]


def _softmax(scores):
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("Softmax probabilities are non-finite or do not sum to one.")
    return probabilities


def _ids(labels):
    values = [str(label) for label in labels]
    unknown = sorted(set(values) - set(METRIC_LABELS))
    if unknown:
        raise ValueError(f"Unknown metric labels: {unknown}")
    return np.asarray([METRIC_LABEL_TO_ID[value] for value in values], dtype=int)


def fit_temperature(dev_scores, dev_labels):
    dev_scores = np.asarray(dev_scores, dtype=float)
    y = _ids(dev_labels)
    if len(y) != len(dev_scores):
        raise ValueError("Development scores and labels have different lengths.")
    lower, upper = CALIBRATION_TEMPERATURE_BOUNDS

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        return log_loss(
            y, _softmax(dev_scores / temperature),
            labels=np.arange(len(METRIC_LABELS)),
        )

    result = minimize_scalar(
        objective, bounds=(np.log(lower), np.log(upper)), method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    boundary_tolerance = 1e-3
    if temperature <= lower * (1 + boundary_tolerance) or temperature >= upper * (1 - boundary_tolerance):
        raise RuntimeError(
            f"Fitted temperature {temperature:.8g} reached the prespecified search boundary "
            f"[{lower}, {upper}]; widen or justify the range before publication."
        )
    return temperature


def expected_calibration_error(
    y_ids, probabilities, n_bins=15, prediction_ids=None
):
    """Equal-width top-label ECE over [0, 1] with an explicit tie policy."""
    confidence = probabilities.max(axis=1)
    prediction = (
        probabilities.argmax(axis=1)
        if prediction_ids is None
        else np.asarray(prediction_ids, dtype=int)
    )
    if prediction.shape != np.asarray(y_ids).shape:
        raise ValueError("Top-label predictions and true labels have different shapes.")
    correct = prediction == y_ids
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for bin_index in range(n_bins):
        left, right = edges[bin_index], edges[bin_index + 1]
        mask = (confidence >= left) & (
            confidence < right if bin_index < n_bins - 1 else confidence <= right
        )
        if not mask.any():
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(bin_accuracy - bin_confidence)
        rows.append({
            "bin": bin_index + 1, "left": left, "right": right,
            "n": int(mask.sum()), "accuracy": bin_accuracy,
            "confidence": bin_confidence, "weight": weight,
        })
    return float(ece), pd.DataFrame(rows)


def per_class_ece(y_ids, probabilities, n_bins=15):
    """One-vs-rest equal-width ECE for each class."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    values = {}
    for class_id, label in enumerate(METRIC_LABELS):
        target = y_ids == class_id
        prob = probabilities[:, class_id]
        ece = 0.0
        for bin_index in range(n_bins):
            left, right = edges[bin_index], edges[bin_index + 1]
            mask = (prob >= left) & (
                prob < right if bin_index < n_bins - 1 else prob <= right
            )
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(target[mask].mean()) - float(prob[mask].mean())
                )
        values[label] = float(ece)
    return values


def probability_metrics(y_labels, probabilities, predicted_labels=None):
    probabilities = np.asarray(probabilities, dtype=float)
    y = _ids(y_labels)
    if probabilities.shape != (len(y), len(METRIC_LABELS)):
        raise ValueError(f"Probability matrix has invalid shape {probabilities.shape}.")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("Probability matrix is invalid.")
    one_hot = np.eye(len(METRIC_LABELS))[y]
    pred = (
        probabilities.argmax(axis=1)
        if predicted_labels is None
        else _ids(predicted_labels)
    )
    if pred.shape != y.shape:
        raise ValueError("Metric predictions and true labels have different shapes.")
    ece, reliability = expected_calibration_error(
        y, probabilities, CALIBRATION_BINS, prediction_ids=pred
    )
    class_ece = per_class_ece(y, probabilities, CALIBRATION_BINS)
    return {
        "NLL": log_loss(y, probabilities, labels=np.arange(len(METRIC_LABELS))),
        "multiclass_Brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "top_label_equal_width_ECE_15": ece,
        "mean_classwise_equal_width_ECE_15": float(np.mean(list(class_ece.values()))),
        "macro_OVR_AUROC": roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"),
        "macro_OVR_AUPRC": average_precision_score(one_hot, probabilities, average="macro"),
        "MCC": matthews_corrcoef(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_F1": f1_score(
            y, pred, labels=np.arange(len(METRIC_LABELS)), average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(y, pred),
    }, reliability, class_ece


PUBLICATION_CLASS_LABELS = [label.title() for label in LABEL_ORDER]
if "generated_figure_artifacts" not in globals():
    generated_figure_artifacts = []


def publication_model_label(model_name):
    direct = {
        "S0_TFIDF_LogReg": "TF-IDF logistic regression",
        "S1_TFIDF_LinearSVM": "TF-IDF linear SVM",
        "S2_TFIDF_MNB": "TF-IDF multinomial naive Bayes",
        "S5_NumberNormalized_LogReg": "Number-normalised TF-IDF logistic regression",
        "S6_PositionAware_LogReg": "Position-aware logistic regression",
        "S8_BiLSTMStyle_SequentialModel": "BiLSTM sequence model",
    }
    if model_name in direct:
        return direct[model_name]
    if str(model_name).startswith("S9_"):
        checkpoint = (
            "BiomedBERT" if "biomedbert" in model_name
            else "SciBERT" if "scibert" in model_name
            else "BERT-base"
        )
        mode = (
            "previous + current"
            if str(model_name).endswith("prev_plus_current")
            else "sentence only"
        )
        return f"{checkpoint} ({mode})"
    return str(model_name).replace("_", " ")


def save_publication_figure(fig, stem):
    """Save one deterministic white-background PNG/PDF/RGB-LZW-TIFF figure."""
    output_paths = {
        "png": REVISION_OUTPUT_DIR / f"{stem}.png",
        "pdf": REVISION_OUTPUT_DIR / f"{stem}.pdf",
        "tif": REVISION_OUTPUT_DIR / f"{stem}.tif",
    }
    fig.savefig(
        output_paths["png"], format="png", dpi=300,
        bbox_inches="tight", facecolor="white", metadata={"Software": "matplotlib"},
    )
    fig.savefig(
        output_paths["pdf"], format="pdf", bbox_inches="tight", facecolor="white",
        metadata={"Creator": "matplotlib", "Producer": "matplotlib", "CreationDate": None, "ModDate": None},
    )
    try:
        fig.savefig(
            output_paths["tif"], format="tiff", dpi=600,
            bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        from PIL import Image
        with Image.open(output_paths["tif"]) as rendered:
            rendered.load()
            if rendered.mode == "RGBA":
                flattened = Image.new("RGB", rendered.size, "white")
                flattened.paste(rendered, mask=rendered.getchannel("A"))
            else:
                flattened = rendered.convert("RGB")
        temporary_tif = output_paths["tif"].with_suffix(".rgb_tmp.tif")
        flattened.save(
            temporary_tif, format="TIFF", compression="tiff_lzw", dpi=(600, 600)
        )
        temporary_tif.replace(output_paths["tif"])
    except Exception as exc:
        raise RuntimeError(f"Could not create RGB 600-dpi LZW TIFF for {stem}: {exc}") from exc
    for output_path in output_paths.values():
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Publication figure was not written correctly: {output_path}")
        generated_figure_artifacts.append(output_path.name)
    return output_paths


calibration_rows = []
calibration_classwise_ece_rows = []
calibration_reliability_tables = {}
raw_test_probabilities = {}
calibrated_test_probabilities = {}

for model_name, test_pack in final_test_scores.items():
    if model_name not in final_dev_scores:
        raise RuntimeError(f"Missing development scores for calibration: {model_name}")
    dev_pack = final_dev_scores[model_name]
    dev_scores = _align_scores(dev_pack["scores"], dev_pack["classes"])
    test_scores = _align_scores(test_pack["scores"], test_pack["classes"])
    if len(dev_scores) != len(dev_pack["y_true"]) or len(test_scores) != len(final_test_y_true):
        raise RuntimeError(f"Calibration score length mismatch for {model_name}.")
    if dev_pack.get("score_type") != test_pack.get("score_type"):
        raise RuntimeError(f"Development/test score semantics differ for {model_name}.")
    temperature = fit_temperature(dev_scores, dev_pack["y_true"])
    raw_prob = _softmax(test_scores)
    calibrated_prob = _softmax(test_scores / temperature)
    if not np.array_equal(raw_prob.argmax(axis=1), calibrated_prob.argmax(axis=1)):
        raise AssertionError("Positive scalar temperature changed predicted labels.")
    raw_test_probabilities[model_name] = raw_prob
    calibrated_test_probabilities[model_name] = calibrated_prob

    state_metrics = {}
    for state, prob in [("uncalibrated", raw_prob), ("temperature_scaled", calibrated_prob)]:
        metrics, reliability, class_ece = probability_metrics(
            final_test_y_true, prob, final_test_predictions[model_name]
        )
        state_metrics[state] = metrics
        calibration_rows.append({
            "model": model_name, "state": state,
            "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
            "score_type_before_softmax": test_pack["score_type"],
            "score_to_probability_transform": "softmax(raw_score / temperature); temperature=1 when uncalibrated",
            "top_label_tie_policy": (
                "saved reference prediction from raw-score argmax in native model class order"
            ),
            "temperature_fitted_on_development": temperature,
            "temperature_search_lower": CALIBRATION_TEMPERATURE_BOUNDS[0],
            "temperature_search_upper": CALIBRATION_TEMPERATURE_BOUNDS[1],
            "calibration_bin_definition": "15 equal-width bins on [0,1]",
            "analysis_role": "descriptive point estimate; no between-model calibration significance claim",
            "uncertainty_scope": "temperature fixed from development; no calibration bootstrap interval",
            **metrics,
        })
        for class_label, ece_value in class_ece.items():
            calibration_classwise_ece_rows.append({
                "model": model_name, "state": state,
                "class": class_label, "classwise_equal_width_ECE_15": ece_value,
                "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
            })
        calibration_reliability_tables[(model_name, state)] = reliability.assign(
            reference_seed=REFERENCE_PREDICTION_SEED[model_name],
            calibration_bins=CALIBRATION_BINS,
        )
    for invariant_metric in ["macro_F1", "MCC", "balanced_accuracy", "accuracy"]:
        if not np.isclose(
            state_metrics["uncalibrated"][invariant_metric],
            state_metrics["temperature_scaled"][invariant_metric],
            rtol=0, atol=1e-12,
        ):
            raise AssertionError(f"Temperature scaling changed {invariant_metric} for {model_name}.")

calibration_metrics_df = pd.DataFrame(calibration_rows)
calibration_classwise_ece_df = pd.DataFrame(calibration_classwise_ece_rows)
calibration_reliability_bins_df = pd.concat([
    table.assign(model=model_name, state=state)
    for (model_name, state), table in calibration_reliability_tables.items()
], ignore_index=True)
calibration_metrics_df.to_csv(
    REVISION_OUTPUT_DIR / "calibration_and_complementary_metrics.csv", index=False
)
calibration_classwise_ece_df.to_csv(
    REVISION_OUTPUT_DIR / "calibration_classwise_ece.csv", index=False
)
calibration_reliability_bins_df.to_csv(
    REVISION_OUTPUT_DIR / "calibration_reliability_bins.csv", index=False
)
display(calibration_metrics_df)
display(calibration_classwise_ece_df)

# One composite Supporting Figure file corresponds to the manuscript's singular S2 Fig.
calibration_reliability_bins_df.to_csv(
    REVISION_OUTPUT_DIR / "S2_fig_source.csv", index=False, lineterminator="\n"
)
generated_figure_artifacts.append("S2_fig_source.csv")
model_order = list(calibrated_test_probabilities)
n_columns = 3
n_rows = int(math.ceil(len(model_order) / n_columns))
fig, axes = plt.subplots(
    n_rows, n_columns, figsize=(7.25, min(8.5, 2.6 * n_rows)),
    constrained_layout=True, squeeze=False,
)
for axis, model_name in zip(axes.flat, model_order):
    for state, color, marker_style in [
        ("uncalibrated", "#D55E00", "o"),
        ("temperature_scaled", "#0072B2", "s"),
    ]:
        rel = calibration_reliability_tables[(model_name, state)]
        axis.plot(
            rel["confidence"], rel["accuracy"], marker=marker_style,
            linewidth=1.3, markersize=3.5, color=color,
            label=state.replace("_", " "),
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=0.8)
    axis.set(xlim=(0, 1), ylim=(0, 1), title=publication_model_label(model_name))
    axis.grid(alpha=0.18, linewidth=0.5)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Observed accuracy")
for axis in axes.flat[len(model_order):]:
    axis.set_visible(False)
axes.flat[0].legend(frameon=False, loc="lower right")
save_publication_figure(fig, "S2_fig")
plt.close(fig)

# Paired raw Macro-F1 differences: one cited S3 Fig and one exact source table.
paired_plot_source_df = paired_bootstrap_comparison_df.copy()
paired_plot_source_df["comparison"] = (
    paired_plot_source_df["Model A"].map(publication_model_label)
    + " minus "
    + paired_plot_source_df["Model B"].map(publication_model_label)
)
paired_plot_source_df.to_csv(
    REVISION_OUTPUT_DIR / "S3_fig_source.csv", index=False, lineterminator="\n"
)
generated_figure_artifacts.append("S3_fig_source.csv")
plot_frame = paired_plot_source_df.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.25, 4.8), constrained_layout=True)
y_position = np.arange(len(plot_frame))
point = plot_frame["Delta Macro-F1 (A-B)"].to_numpy()
low = plot_frame["95% CI low"].to_numpy()
high = plot_frame["95% CI high"].to_numpy()
ax.hlines(y_position, low, high, color="#4C78A8", linewidth=1.5)
ax.scatter(point, y_position, color="#0072B2", s=22, zorder=3)
ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
ax.set_yticks(y_position, labels=plot_frame["comparison"])
ax.set_xlabel("Macro-F1 difference (model A minus model B)")
ax.set_ylabel("")
ax.grid(axis="x", alpha=0.2, linewidth=0.6)
save_publication_figure(fig, "S3_fig")
plt.close(fig)

# ============================================================
# LOCKED TEST ERROR ANALYSIS AND COMPOSITE CONFUSION FIGURES
# ============================================================
from sklearn.metrics import precision_recall_fscore_support

per_class_rows = []
confusion_long_rows = []
for model_name, prediction in final_test_predictions.items():
    precision, recall, f1_values, support = precision_recall_fscore_support(
        final_test_y_true, prediction, labels=LABEL_ORDER, zero_division=0
    )
    for label, p, r, f, n in zip(LABEL_ORDER, precision, recall, f1_values, support):
        per_class_rows.append({
            "model": model_name, "class": label,
            "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
            "precision": p, "recall": r, "F1": f, "support": int(n),
        })
    cm = confusion_matrix(final_test_y_true, prediction, labels=LABEL_ORDER)
    if (cm.sum(axis=1) == 0).any():
        raise RuntimeError(f"A fixed test class has zero support for {model_name}.")
    for i, true_label in enumerate(LABEL_ORDER):
        for j, predicted_label in enumerate(LABEL_ORDER):
            confusion_long_rows.append({
                "model": model_name,
                "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": int(cm[i, j]),
                "row_proportion": float(cm[i, j] / cm[i].sum()),
            })

per_class_test_metrics_df = pd.DataFrame(per_class_rows)
final_test_confusion_long_df = pd.DataFrame(confusion_long_rows)
per_class_test_metrics_df.to_csv(
    REVISION_OUTPUT_DIR / "per_class_test_precision_recall_f1.csv", index=False
)
final_test_confusion_long_df.to_csv(
    REVISION_OUTPUT_DIR / "all_final_test_confusion_matrices_long.csv", index=False
)
display(per_class_test_metrics_df)

# Descriptive post-lock error patterns cannot feed back into any choice.
if PRIMARY_TRANSFORMER not in final_test_predictions:
    raise RuntimeError(f"Primary transformer missing from locked predictions: {PRIMARY_TRANSFORMER}")
top_model_name = PRIMARY_TRANSFORMER
top_errors = (
    final_test_confusion_long_df[
        (final_test_confusion_long_df["model"] == top_model_name)
        & (final_test_confusion_long_df["true_label"] != final_test_confusion_long_df["predicted_label"])
    ]
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)
display(top_errors.head(15))


def _confusion_arrays(source_frame):
    count_matrix = (
        source_frame.pivot(index="true_label", columns="predicted_label", values="count")
        .reindex(index=LABEL_ORDER, columns=LABEL_ORDER)
        .to_numpy(dtype=int)
    )
    proportion_matrix = (
        source_frame.pivot(index="true_label", columns="predicted_label", values="row_proportion")
        .reindex(index=LABEL_ORDER, columns=LABEL_ORDER)
        .to_numpy(dtype=float)
    )
    if count_matrix.shape != (len(LABEL_ORDER), len(LABEL_ORDER)):
        raise RuntimeError("Confusion source does not form the fixed 5 x 5 grid.")
    return count_matrix, proportion_matrix


def draw_locked_confusion_matrix(axis, source_frame, title=None, annotation_size=8):
    count_matrix, proportion_matrix = _confusion_arrays(source_frame)
    image = axis.imshow(
        proportion_matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal"
    )
    for row in range(len(LABEL_ORDER)):
        for column in range(len(LABEL_ORDER)):
            proportion = proportion_matrix[row, column]
            count = count_matrix[row, column]
            axis.text(
                column, row, f"{count:,}\n{proportion * 100:.1f}%",
                ha="center", va="center", fontsize=annotation_size,
                color="white" if proportion >= 0.50 else "#202020",
            )
    axis.set_xticks(np.arange(len(LABEL_ORDER)), labels=PUBLICATION_CLASS_LABELS)
    axis.set_yticks(np.arange(len(LABEL_ORDER)), labels=PUBLICATION_CLASS_LABELS)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    axis.set_xlabel("Predicted role")
    axis.set_ylabel("True role")
    if title:
        axis.set_title(title)
    return image


# One complete source and one multi-panel file correspond to the singular S1 Fig caption.
final_test_confusion_long_df.to_csv(
    REVISION_OUTPUT_DIR / "S1_fig_source.csv", index=False, lineterminator="\n"
)
generated_figure_artifacts.append("S1_fig_source.csv")
model_order = list(final_test_predictions)
n_columns = 3
n_rows = int(math.ceil(len(model_order) / n_columns))
fig, axes = plt.subplots(
    n_rows, n_columns, figsize=(7.25, min(8.5, 2.75 * n_rows)),
    constrained_layout=True, squeeze=False,
)
last_image = None
for axis, model_name in zip(axes.flat, model_order):
    model_source = final_test_confusion_long_df[
        final_test_confusion_long_df["model"] == model_name
    ].copy()
    if len(model_source) != len(LABEL_ORDER) ** 2:
        raise RuntimeError(f"Incomplete confusion source grid for {model_name}")
    last_image = draw_locked_confusion_matrix(
        axis, model_source,
        title=f"{publication_model_label(model_name)}; seed {REFERENCE_PREDICTION_SEED[model_name]}",
        annotation_size=8,
    )
for axis in axes.flat[len(model_order):]:
    axis.set_visible(False)
if last_image is not None:
    colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    colorbar.set_label("Row proportion")
save_publication_figure(fig, "S1_fig")
plt.close(fig)

# Main Fig 1 is the prespecified primary transformer only.
primary_source = final_test_confusion_long_df[
    final_test_confusion_long_df["model"] == PRIMARY_TRANSFORMER
].copy()
primary_source_name = "Fig1_primary_transformer_confusion_matrix_source.csv"
primary_source.to_csv(
    REVISION_OUTPUT_DIR / primary_source_name, index=False, lineterminator="\n"
)
generated_figure_artifacts.append(primary_source_name)
fig, ax = plt.subplots(figsize=(6.75, 5.6), constrained_layout=True)
image = draw_locked_confusion_matrix(ax, primary_source, annotation_size=8)
colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
colorbar.set_label("Row proportion")
save_publication_figure(fig, "Fig1")
plt.close(fig)

CANONICAL_FIG1_ARTIFACTS = [
    "Fig1.png", "Fig1.pdf", "Fig1.tif",
    "Fig1_primary_transformer_confusion_matrix_source.csv",
]
for artifact_name in CANONICAL_FIG1_ARTIFACTS:
    artifact_path = REVISION_OUTPUT_DIR / artifact_name
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        raise RuntimeError(f"Canonical primary Fig 1 artifact missing or empty: {artifact_path}")

# ============================================================
# CONTEXT-FAITHFUL AUTOMATIC PERTURBATION INFERENCE
# ============================================================
# These outputs quantify prediction sensitivity/consistency only. They do not
# establish grammaticality, semantic preservation, label preservation, or error.
from decimal import Decimal, InvalidOperation, localcontext

PERTURBATION_VERSION = "plos_r1_context_faithful_v4"
PERTURBATION_INFERENCE_VERSION = "plos_r1_native_order_tie_safe_serial_v1"
PERTURBATION_INFERENCE_DATALOADER_WORKERS = 0
TRANSFORMER_PREDICTION_TIE_POLICY = (
    "argmax in native model class order before export score-column reordering"
)
PERTURBATION_SEED = 42
PERTURBATION_BOOTSTRAP_SEED = 42017
PERTURBATION_BOOTSTRAP_N = 10_000 if PUBLICATION_RUN else 200

PARAPHRASE_REPLACEMENTS = [
    (r"\bstudy\b", "research"), (r"\btrial\b", "study"),
    (r"\bpatients\b", "participants"), (r"\bpatient\b", "participant"),
    (r"\banalysis\b", "evaluation"), (r"\binvestigate\b", "examine"),
    (r"\binvestigated\b", "examined"), (r"\bassess\b", "evaluate"),
    (r"\bassessed\b", "evaluated"), (r"\beffects\b", "impacts"),
    (r"\beffect\b", "impact"), (r"\boutcomes\b", "endpoints"),
    (r"\boutcome\b", "endpoint"), (r"\bcompared\b", "contrasted"),
    (r"\bcompare\b", "contrast"),
]
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])")
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|neither|nor|without|cannot)\b|n(?:'|\u2019)t\b", re.IGNORECASE
)

AUXILIARY_PATTERN = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|will|would|may|might|should)\b",
    re.IGNORECASE,
)
TERMINAL_PUNCTUATION_PATTERN = re.compile(
    r"((?=[\s.!?'\"\)\]\}\u2019\u201d]*[.!?\)\]\}])[\s.!?'\"\)\]\}\u2019\u201d]+)$"
)


def _stable_rng(stable_key):
    digest = hashlib.sha256(str(stable_key).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _operation_record(text, changed, branch, start, end, original, replacement, **extra):
    record = {
        "perturbed_text": changed,
        "rule_branch": branch,
        "target_span_start": int(start),
        "target_span_end": int(end),
        "target_original": str(original),
        "target_replacement": str(replacement),
        "n_operations": 1,
    }
    record.update(extra)
    if changed == str(text):
        raise AssertionError(f"Perturbation branch {branch!r} did not change the text.")
    return record


def _nonzero_numeric_matches(text):
    matches = []
    for match in NUMBER_PATTERN.finditer(str(text)):
        try:
            if Decimal(match.group(0)) != 0:
                matches.append(match)
        except InvalidOperation:
            continue
    return matches


def _plain_decimal(value):
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"-0", "+0", ""}:
        rendered = "0"
    return rendered


def perturb_numeric_v4(text, stable_key):
    text = str(text)
    matches = _nonzero_numeric_matches(text)
    if not matches:
        raise ValueError("Numeric perturbation requires a nonzero numeric span.")
    match = matches[_stable_rng(stable_key).randrange(len(matches))]
    factor = Decimal("0.9") if _stable_rng(stable_key + "|direction").randrange(2) == 0 else Decimal("1.1")
    original_value = Decimal(match.group(0))
    with localcontext() as context:
        context.prec = max(28, len(match.group(0).replace(".", "").replace("+", "").replace("-", "")) + 12)
        changed_value = original_value * factor
    replacement = _plain_decimal(changed_value)
    changed = text[:match.start()] + replacement + text[match.end():]
    if Decimal(replacement) != original_value * factor:
        raise AssertionError("Numeric replacement is not exactly the declared Decimal factor.")
    return _operation_record(
        text, changed, "one_nonzero_numeric_span_exact_decimal_factor",
        match.start(), match.end(), match.group(0), replacement,
        numeric_factor=str(factor), numeric_original_decimal=str(original_value),
        numeric_replacement_decimal=str(Decimal(replacement)),
    )


def _match_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement.lower()


def _paraphrase_matches(text):
    candidates = []
    for order, (pattern, replacement) in enumerate(PARAPHRASE_REPLACEMENTS):
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            candidates.append((match.start(), order, match, replacement))
    return sorted(candidates, key=lambda row: (row[0], row[1]))


def perturb_paraphrase_v4(text, stable_key=None):
    text = str(text)
    candidates = _paraphrase_matches(text)
    if not candidates:
        raise ValueError("Paraphrase perturbation requires a listed lexical target.")
    _, _, match, base_replacement = candidates[0]
    replacement = _match_case(match.group(0), base_replacement)
    changed = text[:match.start()] + replacement + text[match.end():]
    return _operation_record(
        text, changed, "earliest_listed_case_preserving_lexical_substitution",
        match.start(), match.end(), match.group(0), replacement,
    )


def perturb_structure_v4(text, stable_key=None):
    text = str(text).strip()
    terminal_match = TERMINAL_PUNCTUATION_PATTERN.search(text)
    terminal = terminal_match.group(1) if terminal_match else ""
    body = text[:terminal_match.start()].rstrip() if terminal_match else text
    clauses = [part.strip() for part in body.split(",")]
    if len(clauses) < 2 or any(not part for part in clauses):
        raise ValueError("Structure perturbation requires at least two nonempty comma clauses.")
    changed = ", ".join(reversed(clauses)) + terminal
    if terminal and not changed.endswith(terminal):
        raise AssertionError("Terminal punctuation was not reapplied at the sentence end.")
    return _operation_record(
        text, changed, "reverse_comma_clauses_preserve_terminal_punctuation",
        0, len(text), text, changed, terminal_punctuation=terminal,
    )


def perturb_negation_v4(text, stable_key=None):
    text = str(text).strip()
    if NEGATION_PATTERN.search(text):
        raise ValueError("Already-negated text is ineligible for negation insertion.")
    match = AUXILIARY_PATTERN.search(text)
    if match:
        replacement = match.group(0) + " not"
        changed = text[:match.start()] + replacement + text[match.end():]
        return _operation_record(
            text, changed, "insert_not_after_first_auxiliary",
            match.start(), match.end(), match.group(0), replacement,
        )
    lowered = text[:1].lower() + text[1:] if text else text
    prefix = "It is not the case that "
    changed = prefix + lowered
    return _operation_record(
        text, changed, "single_scope_negation_rewrite", 0, len(text), text, changed,
    )


PERTURBATION_RULES = {
    "negation": {
        "n_per_label": 500 if PUBLICATION_RUN else 2,
        "eligible": lambda text: bool(str(text).strip()) and not NEGATION_PATTERN.search(str(text)),
        "transform": perturb_negation_v4,
    },
    "numeric": {
        "n_per_label": 100 if PUBLICATION_RUN else 2,
        "eligible": lambda text: bool(_nonzero_numeric_matches(text)),
        "transform": perturb_numeric_v4,
    },
    "paraphrase": {
        "n_per_label": 100 if PUBLICATION_RUN else 2,
        "eligible": lambda text: bool(_paraphrase_matches(text)),
        "transform": perturb_paraphrase_v4,
    },
    "structure": {
        "n_per_label": 100 if PUBLICATION_RUN else 2,
        "eligible": lambda text: (
            len([part.strip() for part in TERMINAL_PUNCTUATION_PATTERN.sub("", str(text).strip()).split(",")]) >= 2
            and all(part.strip() for part in TERMINAL_PUNCTUATION_PATTERN.sub("", str(text).strip()).split(","))
        ),
        "transform": perturb_structure_v4,
    },
}


def _stratified_target_sample(frame, n_per_label, eligibility, seed):
    use = frame[frame["text"].map(eligibility)].copy()
    parts = []
    for offset, label in enumerate(LABEL_ORDER):
        group = use[use["label"] == label]
        if len(group) < n_per_label:
            raise ValueError(
                f"Insufficient eligible {label} items: need {n_per_label}, found {len(group)}"
            )
        parts.append(group.sample(n=n_per_label, random_state=seed + offset))
    return pd.concat(parts, ignore_index=True)


canonical_test = add_prev_sentence_context(test_sent).copy().reset_index(drop=True)
canonical_test["canonical_row_index"] = np.arange(len(canonical_test))
eligibility_rows = []
benchmark_parts = []
for rule_index, (perturbation_type, specification) in enumerate(PERTURBATION_RULES.items()):
    eligibility_mask = canonical_test["text"].map(specification["eligible"])
    for label in LABEL_ORDER:
        label_mask = canonical_test["label"] == label
        eligibility_rows.append({
            "perturbation_type": perturbation_type,
            "label": label,
            "n_available": int(label_mask.sum()),
            "n_eligible": int((label_mask & eligibility_mask).sum()),
            "n_sampled": int(specification["n_per_label"]),
            "sampling_seed": PERTURBATION_SEED + rule_index,
        })
    sampled = _stratified_target_sample(
        canonical_test, specification["n_per_label"], specification["eligible"],
        PERTURBATION_SEED + rule_index,
    )
    sampled["perturbation_type"] = perturbation_type
    operation_rows = []
    for row in sampled.itertuples(index=False):
        stable_key = (
            f"{row.pubmed_id}|{row.file_order}|{perturbation_type}|{PERTURBATION_VERSION}"
        )
        operation_rows.append(specification["transform"](row.text, stable_key))
    sampled = pd.concat(
        [sampled.reset_index(drop=True), pd.DataFrame(operation_rows)], axis=1
    )
    sampled["original_text"] = sampled["text"].astype(str)
    sampled["changed_text"] = sampled["original_text"] != sampled["perturbed_text"]
    sampled["perturbation_version"] = PERTURBATION_VERSION
    sampled["benchmark_item_id"] = sampled.apply(
        lambda row: hashlib.sha256(
            (
                f"test|{row.pubmed_id}|{row.file_order}|{row.perturbation_type}|"
                f"{PERTURBATION_VERSION}|{row.original_text}|{row.perturbed_text}"
            ).encode("utf-8")
        ).hexdigest(), axis=1,
    )
    benchmark_parts.append(sampled)

context_faithful_perturbation_eligibility_df = pd.DataFrame(eligibility_rows)
context_faithful_benchmark_df = pd.concat(benchmark_parts, ignore_index=True)
expected_counts = {
    rule: int(specification["n_per_label"] * len(LABEL_ORDER))
    for rule, specification in PERTURBATION_RULES.items()
}
if not context_faithful_benchmark_df["benchmark_item_id"].is_unique:
    raise AssertionError("Perturbation item IDs are not unique.")
if not context_faithful_benchmark_df["changed_text"].all():
    raise AssertionError("Every sampled perturbation must change its source text.")
if not context_faithful_benchmark_df["n_operations"].eq(1).all():
    raise AssertionError("Every perturbation item must contain exactly one rule operation.")
for operation in context_faithful_benchmark_df.itertuples(index=False):
    reconstructed = (
        operation.original_text[:operation.target_span_start]
        + operation.target_replacement
        + operation.original_text[operation.target_span_end:]
    )
    if reconstructed != operation.perturbed_text:
        raise AssertionError(
            f"Recorded perturbation span does not reconstruct {operation.benchmark_item_id}."
        )
if context_faithful_benchmark_df.groupby("perturbation_type").size().to_dict() != expected_counts:
    raise AssertionError("Perturbation stratum counts do not match the prespecified sample.")
numeric_rows = context_faithful_benchmark_df.query("perturbation_type == 'numeric'")
for row in numeric_rows.itertuples(index=False):
    if Decimal(row.numeric_replacement_decimal) != (
        Decimal(row.numeric_original_decimal) * Decimal(row.numeric_factor)
    ):
        raise AssertionError("A numeric item is not exactly +/-10% under Decimal arithmetic.")
if context_faithful_benchmark_df.query("perturbation_type == 'negation'")[
    "original_text"
].map(lambda text: bool(NEGATION_PATTERN.search(text))).any():
    raise AssertionError("Already-negated text entered the negation benchmark.")
for row in context_faithful_benchmark_df.query(
    "perturbation_type == 'structure'"
).itertuples(index=False):
    suffix_match = TERMINAL_PUNCTUATION_PATTERN.search(row.original_text.strip())
    expected_suffix = suffix_match.group(1) if suffix_match else ""
    if row.terminal_punctuation != expected_suffix or (
        expected_suffix and not row.perturbed_text.endswith(expected_suffix)
    ):
        raise AssertionError("A structure perturbation did not preserve its full terminal suffix.")


def _reorder_score_matrix(scores, classes):
    scores = np.asarray(scores, dtype=float)
    classes = [str(value) for value in classes]
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise AssertionError("Score matrix shape does not match its class list.")
    if set(classes) != set(LABEL_ORDER) or not np.isfinite(scores).all():
        raise AssertionError("Score matrix classes/finiteness violate the publication contract.")
    return scores[:, [classes.index(label) for label in LABEL_ORDER]]


def _classical_perturbation_outputs(system, benchmark, text_column):
    estimator = system["estimator"]
    X = benchmark.copy()
    X["text"] = X[text_column].astype(str)
    X["text_num_norm"] = X["text"].map(normalize_numbers)
    model_input = X if isinstance(estimator, PositionAwareTextClassifier) else X[[system["text_col"]]]
    prediction = np.asarray(estimator.predict(model_input))
    scores, classes, score_type = _scores_and_classes(estimator, model_input)
    return prediction, _reorder_score_matrix(scores, classes), score_type


def _prepared_transformer_dataset(benchmark, tokenizer, max_length, mode, text_column):
    prepared = benchmark[[text_column, "prev_text", "label_id"]].rename(
        columns={text_column: "text", "label_id": "labels"}
    )
    dataset = Dataset.from_pandas(prepared, preserve_index=False)
    if mode == "sentence_only":
        dataset = dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, padding="max_length",
                max_length=max_length, verbose=False,
            ), batched=True,
        )
    elif mode == "prev_plus_current":
        dataset = dataset.map(
            lambda batch: _tokenize_preserve_current(
                tokenizer, batch["prev_text"], batch["text"], max_length=max_length
            ), batched=True,
        )
    else:
        raise ValueError(f"Unknown transformer input mode: {mode}")
    columns = [
        name for name in ["input_ids", "attention_mask", "token_type_ids", "labels"]
        if name in dataset.column_names
    ]
    dataset.set_format(type="torch", columns=columns, output_all_columns=False)
    return dataset


def _predict_transformer_logits_for_perturbation(model, dataset, batch_size):
    """Direct, serial inference that is safe inside a live Colab/Jupyter process."""
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=PERTURBATION_INFERENCE_DATALOADER_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    logits = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items() if key != "labels"
            }
            logits.append(model(**inputs).logits.detach().cpu().numpy())
    if not logits:
        raise AssertionError("Perturbation transformer inference produced no logits.")
    return np.concatenate(logits, axis=0)


_test_abstract_lookup = {
    str(pubmed_id): abstract.sort_values("file_order").reset_index(drop=True)
    for pubmed_id, abstract in test_sent.groupby("pubmed_id", sort=False)
}


def _bilstm_target_perturbation_outputs(system, benchmark, full_test):
    expanded_parts, target_position = [], {}
    for item in benchmark.itertuples(index=False):
        abstract = _test_abstract_lookup.get(str(item.pubmed_id))
        if abstract is None:
            raise AssertionError("Perturbation PubMed ID is absent from the test lookup.")
        abstract = abstract.copy()
        target_mask = abstract["file_order"] == item.file_order
        if int(target_mask.sum()) != 1:
            raise AssertionError("Target sentence was not uniquely located in its abstract.")
        position = int(np.flatnonzero(target_mask.to_numpy())[0])
        abstract.loc[target_mask, "text"] = item.perturbed_text
        abstract["pubmed_id"] = item.benchmark_item_id
        target_position[item.benchmark_item_id] = position
        expanded_parts.append(abstract)
    expanded = pd.concat(expanded_parts, ignore_index=True)
    output = predict_bilstm_style_model(
        system["model"], system["vocab"], expanded, BILSTM_BATCH_SIZE
    )
    lookup = pd.DataFrame({
        "benchmark_item_id": output["pubmed_id"],
        "sentence_index": output["sentence_index"],
        "prediction": output["y_pred"],
        "score_row": list(output["scores"]),
    })
    lookup["target_position"] = lookup["benchmark_item_id"].map(target_position)
    if lookup["target_position"].isna().any():
        raise AssertionError("BiLSTM perturbation target position lookup is incomplete.")
    lookup = lookup[lookup["sentence_index"].eq(lookup["target_position"])]
    if lookup["benchmark_item_id"].duplicated().any() or len(lookup) != len(benchmark):
        raise AssertionError("BiLSTM perturbation target lookup is incomplete or duplicated.")
    lookup = lookup.set_index("benchmark_item_id")
    ids = benchmark["benchmark_item_id"].tolist()
    prediction = lookup.loc[ids, "prediction"].to_numpy()
    scores = np.vstack(lookup.loc[ids, "score_row"].to_numpy())
    return prediction, _reorder_score_matrix(scores, output["score_classes"]), "logit"


perturbation_prediction_rows = []
for system_name, system in final_fitted_systems.items():
    benchmark = context_faithful_benchmark_df.copy()
    row_indices = benchmark["canonical_row_index"].to_numpy(dtype=int)
    if system["family"] == "classical":
        original_prediction, original_scores, score_type = _classical_perturbation_outputs(
            system, benchmark, "original_text"
        )
        perturbed_prediction, perturbed_scores, perturbed_score_type = _classical_perturbation_outputs(
            system, benchmark, "perturbed_text"
        )
    elif system["family"] == "transformer":
        original_dataset = _prepared_transformer_dataset(
            benchmark, system["tokenizer"], system["max_length"], system["mode"],
            "original_text",
        )
        perturbed_dataset = _prepared_transformer_dataset(
            benchmark, system["tokenizer"], system["max_length"], system["mode"],
            "perturbed_text",
        )
        native_original_scores = _predict_transformer_logits_for_perturbation(
            system["model"], original_dataset, TRANSFORMER_EVAL_BATCH_SIZE
        )
        native_perturbed_scores = _predict_transformer_logits_for_perturbation(
            system["model"], perturbed_dataset, TRANSFORMER_EVAL_BATCH_SIZE
        )
        transformer_score_classes = [
            str(ID2LABEL[index]) for index in range(len(ID2LABEL))
        ]
        locked_score_classes = [
            str(value) for value in final_test_scores[system_name]["classes"]
        ]
        if transformer_score_classes != locked_score_classes:
            raise AssertionError(
                f"Native transformer class order differs from locked scores for {system_name}."
            )
        original_prediction = np.asarray(transformer_score_classes)[
            np.argmax(native_original_scores, axis=1)
        ]
        perturbed_prediction = np.asarray(transformer_score_classes)[
            np.argmax(native_perturbed_scores, axis=1)
        ]
        original_scores = _reorder_score_matrix(
            native_original_scores, transformer_score_classes
        )
        perturbed_scores = _reorder_score_matrix(
            native_perturbed_scores, transformer_score_classes
        )
        score_type = perturbed_score_type = "logit"
        system["model"].to("cpu")
        del original_dataset, perturbed_dataset, native_original_scores, native_perturbed_scores
        gc.collect()
        torch.cuda.empty_cache()
    elif system["family"] == "sequence_bilstm":
        system["model"].to("cuda")
        _previous_bilstm_workers = BILSTM_DATALOADER_WORKERS
        try:
            BILSTM_DATALOADER_WORKERS = PERTURBATION_INFERENCE_DATALOADER_WORKERS
            original_output = predict_bilstm_style_model(
                system["model"], system["vocab"], test_sent, BILSTM_BATCH_SIZE
            )
            if not np.array_equal(original_output["y_true"], final_test_y_true):
                raise AssertionError("Recomputed BiLSTM original ordering is not canonical.")
            original_prediction = np.asarray(original_output["y_pred"])[row_indices]
            original_scores = _reorder_score_matrix(
                original_output["scores"], original_output["score_classes"]
            )[row_indices]
            perturbed_prediction, perturbed_scores, perturbed_score_type = (
                _bilstm_target_perturbation_outputs(system, benchmark, test_sent)
            )
        finally:
            BILSTM_DATALOADER_WORKERS = _previous_bilstm_workers
        score_type = "logit"
        system["model"].to("cpu")
        del original_output
        gc.collect()
        torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"Unsupported perturbation family: {system['family']}")

    if perturbed_score_type != score_type:
        raise AssertionError("Original and perturbed score semantics differ.")
    locked_original_prediction = np.asarray(final_test_predictions[system_name])[row_indices]
    locked_original_scores = _reorder_score_matrix(
        final_test_scores[system_name]["scores"], final_test_scores[system_name]["classes"]
    )[row_indices]
    if not np.array_equal(original_prediction, locked_original_prediction):
        raise AssertionError(f"Matched-path original predictions differ for {system_name}.")
    if not np.allclose(original_scores, locked_original_scores, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"Matched-path original scores differ for {system_name}.")

    model_rows = benchmark[[
        "benchmark_item_id", "perturbation_version", "perturbation_type",
        "pubmed_id", "file_order", "canonical_row_index", "label",
        "original_text", "perturbed_text", "prev_text",
        "sentence_index_in_abstract", "relative_position", "rule_branch",
        "target_span_start", "target_span_end", "target_original",
        "target_replacement", "n_operations", "numeric_factor",
        "numeric_original_decimal", "numeric_replacement_decimal",
        "terminal_punctuation",
    ]].copy()
    model_rows["model"] = system_name
    model_rows["reference_seed"] = REFERENCE_PREDICTION_SEED[system_name]
    model_rows["score_type"] = score_type
    model_rows["score_class_order"] = "|".join(LABEL_ORDER)
    model_rows["perturbation_inference_version"] = PERTURBATION_INFERENCE_VERSION
    model_rows["prediction_tie_policy"] = (
        TRANSFORMER_PREDICTION_TIE_POLICY
        if system["family"] == "transformer"
        else "native fitted-model prediction method"
    )
    model_rows["native_score_class_order"] = (
        "|".join(transformer_score_classes)
        if system["family"] == "transformer"
        else "not_applicable_or_model_native"
    )
    model_rows["inference_dataloader_workers"] = (
        PERTURBATION_INFERENCE_DATALOADER_WORKERS
        if system["family"] in {"transformer", "sequence_bilstm"}
        else 0
    )
    model_rows["pred_original"] = original_prediction
    model_rows["pred_perturbed"] = perturbed_prediction
    model_rows["prediction_changed"] = original_prediction != perturbed_prediction
    model_rows["prediction_consistent"] = ~model_rows["prediction_changed"]
    for class_index, label in enumerate(LABEL_ORDER):
        model_rows[f"score_original__{label}"] = original_scores[:, class_index]
        model_rows[f"score_perturbed__{label}"] = perturbed_scores[:, class_index]
    model_rows["interpretation_scope"] = (
        "automatic prediction sensitivity/consistency; semantic validity and correctness not assessed"
    )
    perturbation_prediction_rows.append(model_rows)

context_faithful_perturbation_predictions_df = pd.concat(
    perturbation_prediction_rows, ignore_index=True
)
if context_faithful_perturbation_predictions_df.duplicated(
    ["model", "benchmark_item_id"]
).any():
    raise AssertionError("Perturbation model-item keys are duplicated.")

# Cluster bootstrap complete PubMed abstracts. Within each rule, every model
# receives exactly the same sampled cluster weights in each replicate.
summary_rows = []
distribution_parts = []
for rule_index, perturbation_type in enumerate(PERTURBATION_RULES):
    rule_frame = context_faithful_perturbation_predictions_df.query(
        "perturbation_type == @perturbation_type"
    )
    models = list(FINAL_SYSTEMS)
    cluster_ids = sorted(rule_frame["pubmed_id"].astype(str).unique())
    model_cluster_arrays = {}
    for model_name in models:
        model_frame = rule_frame[rule_frame["model"] == model_name].copy()
        aggregated = (
            model_frame.assign(pubmed_id=model_frame["pubmed_id"].astype(str))
            .groupby("pubmed_id", as_index=True)
            .agg(changed=("prediction_changed", "sum"), items=("benchmark_item_id", "size"))
            .reindex(cluster_ids, fill_value=0)
        )
        model_cluster_arrays[model_name] = (
            aggregated["changed"].to_numpy(dtype=float),
            aggregated["items"].to_numpy(dtype=float),
        )
    bootstrap_seed = PERTURBATION_BOOTSTRAP_SEED + rule_index
    rng = np.random.default_rng(bootstrap_seed)
    replicate_values = {model_name: np.empty(PERTURBATION_BOOTSTRAP_N) for model_name in models}
    chunk_size = 250
    for start in range(0, PERTURBATION_BOOTSTRAP_N, chunk_size):
        stop = min(start + chunk_size, PERTURBATION_BOOTSTRAP_N)
        draws = rng.integers(0, len(cluster_ids), size=(stop - start, len(cluster_ids)))
        for model_name in models:
            changed, items = model_cluster_arrays[model_name]
            denominator = items[draws].sum(axis=1)
            if np.any(denominator <= 0):
                raise AssertionError("A perturbation bootstrap replicate has no sampled items.")
            replicate_values[model_name][start:stop] = changed[draws].sum(axis=1) / denominator
    for model_name in models:
        model_frame = rule_frame[rule_frame["model"] == model_name]
        values = replicate_values[model_name]
        observed_change = float(model_frame["prediction_changed"].mean())
        change_low, change_high = np.percentile(values, [2.5, 97.5])
        summary_rows.append({
            "model": model_name,
            "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
            "perturbation_type": perturbation_type,
            "n_items": int(model_frame["benchmark_item_id"].nunique()),
            "n_abstracts": int(model_frame["pubmed_id"].astype(str).nunique()),
            "prediction_change_rate": observed_change,
            "prediction_change_rate_ci_low": float(change_low),
            "prediction_change_rate_ci_high": float(change_high),
            "prediction_consistency_rate": 1.0 - observed_change,
            "prediction_consistency_rate_ci_low": 1.0 - float(change_high),
            "prediction_consistency_rate_ci_high": 1.0 - float(change_low),
            "bootstrap_replicates": PERTURBATION_BOOTSTRAP_N,
            "bootstrap_seed": bootstrap_seed,
            "perturbation_inference_version": PERTURBATION_INFERENCE_VERSION,
            "inference_dataloader_workers": PERTURBATION_INFERENCE_DATALOADER_WORKERS,
            "ci_method": "percentile abstract-cluster bootstrap",
            "uncertainty_scope": "fixed fitted reference-seed model and fixed sampled perturbation items",
            "interpretation_scope": (
                "automatic prediction sensitivity/consistency; semantic validity and correctness not assessed"
            ),
        })
        distribution_parts.append(pd.DataFrame({
            "model": model_name,
            "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
            "perturbation_type": perturbation_type,
            "replicate": np.arange(PERTURBATION_BOOTSTRAP_N),
            "prediction_change_rate": values,
            "prediction_consistency_rate": 1.0 - values,
            "bootstrap_seed": bootstrap_seed,
            "cluster_unit": "pubmed_id",
            "perturbation_inference_version": PERTURBATION_INFERENCE_VERSION,
            "inference_dataloader_workers": PERTURBATION_INFERENCE_DATALOADER_WORKERS,
        }))

context_faithful_perturbation_summary_df = pd.DataFrame(summary_rows)
context_faithful_perturbation_cluster_bootstrap_distributions_df = pd.concat(
    distribution_parts, ignore_index=True
)
context_faithful_perturbation_predictions_df.to_csv(
    REVISION_OUTPUT_DIR / "context_faithful_perturbation_item_predictions.csv",
    index=False, lineterminator="\n",
)
context_faithful_perturbation_summary_df.to_csv(
    REVISION_OUTPUT_DIR / "context_faithful_perturbation_summary.csv",
    index=False, lineterminator="\n",
)
context_faithful_perturbation_cluster_bootstrap_distributions_df.to_csv(
    REVISION_OUTPUT_DIR / "context_faithful_perturbation_cluster_bootstrap_distributions.csv",
    index=False, lineterminator="\n",
)
context_faithful_perturbation_eligibility_df.to_csv(
    REVISION_OUTPUT_DIR / "context_faithful_perturbation_eligibility.csv",
    index=False, lineterminator="\n",
)
display(context_faithful_perturbation_summary_df)

# ============================================================
# FAIL-CLOSED ITEM-LEVEL EXPORT, ARTIFACT MANIFEST, AND SENTINEL
# ============================================================
import shutil
from datetime import datetime, timezone


ANALYSIS_STAGE_DIR = Path(OUTPUT_DIR) / "plos_revision_analysis_checkpoints"
required_stage_contract_names = [
    "GROUPED_CV_STAGE_COMPLETE.json",
    "GROUPED_FACTORIAL_STAGE_COMPLETE.json",
    "FINAL_BOOTSTRAP_STAGE_COMPLETE.json",
]
analysis_stage_contracts = {}
for contract_name in required_stage_contract_names:
    contract_path = ANALYSIS_STAGE_DIR / contract_name
    if not contract_path.is_file() or contract_path.stat().st_size <= 0:
        raise RuntimeError(f"Required durable analysis checkpoint is absent: {contract_path}")
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    for key, expected in {
        "run_id": RUN_ID,
        "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    }.items():
        if contract_payload.get(key) != expected:
            raise RuntimeError(
                f"Analysis checkpoint {contract_name} has mismatched {key}: "
                f"{contract_payload.get(key)!r}"
            )
    staged_artifacts = contract_payload.get("artifacts", [])
    if not staged_artifacts:
        raise RuntimeError(f"Analysis checkpoint {contract_name} has no artifacts.")
    for staged in staged_artifacts:
        staged_path = ANALYSIS_STAGE_DIR / staged["file"]
        if (
            not staged_path.is_file()
            or staged_path.stat().st_size != int(staged["bytes"])
            or _sha256_file(staged_path) != staged["sha256"]
        ):
            raise RuntimeError(
                f"Analysis checkpoint artifact failed read-back: {staged_path}"
            )
    analysis_stage_contracts[contract_name] = contract_payload

expected_cv_models = set(CLASSICAL_MODEL_SPECS)
if (
    set(classical_cv_results) != expected_cv_models
    or set(cv_fold_predictions) != expected_cv_models
    or set(classical_nested_params) != expected_cv_models
):
    raise RuntimeError(
        "Final export requires exact grouped-CV model-key coverage."
    )
expected_factorial_cells = set(FACTORIAL_CELL_ORDER)
if set(position_weight_factorial_cells_df["cell"]) != expected_factorial_cells:
    raise RuntimeError("Final export factorial cells differ from the 2 x 2 contract.")
expected_factorial_contrasts = {
    "position_effect_unweighted",
    "position_effect_balanced",
    "weight_effect_lexical",
    "weight_effect_position",
    "position_main_effect_average",
    "weight_main_effect_average",
    "position_by_weight_interaction",
}
if set(position_weight_factorial_results["contrast"]) != expected_factorial_contrasts:
    raise RuntimeError("Final export factorial contrasts differ from the contract.")
factorial_export_counts = position_weight_factorial_bootstrap_distributions_df.groupby(
    "contrast"
).size()
if (
    set(factorial_export_counts.index) != expected_factorial_contrasts
    or not factorial_export_counts.eq(ABLATION_BOOTSTRAP_N).all()
    or not np.isfinite(
        position_weight_factorial_bootstrap_distributions_df["delta_macro_f1"]
    ).all()
):
    raise RuntimeError("Final export factorial bootstrap coverage is invalid.")
expected_ablation_changes = {"digit normalisation", "position features"}
if set(grouped_ablation_results["feature_change"]) != expected_ablation_changes:
    raise RuntimeError("Final export controlled-ablation contrasts are invalid.")
ablation_export_counts = grouped_ablation_bootstrap_distributions_df.groupby(
    "feature_change"
).size()
if (
    set(ablation_export_counts.index) != expected_ablation_changes
    or not ablation_export_counts.eq(ABLATION_BOOTSTRAP_N).all()
    or not np.isfinite(
        grouped_ablation_bootstrap_distributions_df[
            "delta_macro_f1_changed_minus_lexical"
        ]
    ).all()
):
    raise RuntimeError("Final export controlled-ablation bootstrap coverage is invalid.")

paired_key_columns_output = ["Model A", "Model B", "Comparison family"]
paired_key_columns_lock = ["model_a", "model_b", "comparison_family"]
expected_paired_keys = set(
    map(
        tuple,
        pretest_comparison_lock_df[paired_key_columns_lock]
        .astype(str)
        .to_numpy(),
    )
)
observed_paired_keys = set(
    map(
        tuple,
        paired_bootstrap_comparison_df[paired_key_columns_output]
        .astype(str)
        .to_numpy(),
    )
)
if observed_paired_keys != expected_paired_keys:
    raise RuntimeError("Paired-bootstrap summary keys differ from the pre-test lock.")
observed_paired_distribution_keys = set(
    map(
        tuple,
        paired_bootstrap_distributions_df[paired_key_columns_output]
        .astype(str)
        .drop_duplicates()
        .to_numpy(),
    )
)
if observed_paired_distribution_keys != expected_paired_keys:
    raise RuntimeError(
        "Paired-bootstrap distribution keys differ from the pre-test lock."
    )

# Development evidence used for candidate selection and calibration.
development_prediction_parts = []


def _development_prediction_frame(base, prediction, model_name, candidate_group):
    frame = base[["pubmed_id", "file_order", "label"]].rename(
        columns={"label": "true_label"}
    ).copy().reset_index(drop=True)
    prediction = np.asarray(prediction).astype(str)
    if len(prediction) != len(frame):
        raise AssertionError(f"Development prediction length mismatch for {model_name}.")
    frame["prediction"] = prediction
    frame["model"] = model_name
    frame["candidate_group"] = candidate_group
    frame["split"] = "development"
    return frame


for model_name, prediction in classical_dev_predictions.items():
    development_prediction_parts.append(_development_prediction_frame(
        classical_dev_df, prediction, model_name, "classical_training_only_selection"
    ))
if "crf_development_predictions_df" not in globals():
    raise RuntimeError("CRF item-level development predictions were not retained.")
crf_base = crf_development_predictions_df.rename(columns={"label": "true_label"}).copy()
crf_base["candidate_group"] = "sequence_development_candidate"
crf_base["split"] = "development"
development_prediction_parts.append(crf_base[[
    "pubmed_id", "file_order", "true_label", "prediction", "model",
    "candidate_group", "split",
]])
if "bilstm_development_predictions_df" not in globals():
    raise RuntimeError("BiLSTM item-level development predictions were not retained.")
bilstm_base = bilstm_development_predictions_df.rename(columns={"label": "true_label"}).copy()
bilstm_base["candidate_group"] = "sequence_development_candidate"
bilstm_base["split"] = "development"
development_prediction_parts.append(bilstm_base[[
    "pubmed_id", "file_order", "true_label", "prediction", "model",
    "candidate_group", "split",
]])
canonical_transformer_dev = splits["dev_sent"].sort_values(
    ["pubmed_id", "file_order"]
).reset_index(drop=True)
for model_name, prediction in transformer_development_predictions.items():
    development_prediction_parts.append(_development_prediction_frame(
        canonical_transformer_dev, prediction, model_name,
        "transformer_development_candidate",
    ))
development_item_predictions_long_df = pd.concat(
    development_prediction_parts, ignore_index=True
)
if development_item_predictions_long_df.duplicated(
    ["model", "pubmed_id", "file_order"]
).any():
    raise AssertionError("Development model-item prediction keys are duplicated.")
if not set(development_item_predictions_long_df["true_label"]).issubset(set(LABEL_ORDER)):
    raise AssertionError("Development evidence contains an unknown true label.")
development_item_predictions_long_df.to_csv(
    REVISION_OUTPUT_DIR / "development_item_predictions_long.csv",
    index=False, lineterminator="\n",
)

development_score_npz = {}
for model_name, scores in transformer_development_scores.items():
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    development_score_npz[f"selection_scores__{safe}"] = np.asarray(scores, dtype=float)
    development_score_npz[f"selection_classes__{safe}"] = np.asarray(
        transformer_development_score_classes[model_name], dtype=str
    )
development_score_npz["selection_scores__S8_BiLSTMStyle_SequentialModel"] = np.asarray(
    bilstm_development_scores, dtype=float
)
development_score_npz["selection_classes__S8_BiLSTMStyle_SequentialModel"] = np.asarray(
    bilstm_development_score_classes, dtype=str
)
for model_name, pack in final_dev_scores.items():
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    development_score_npz[f"calibration_scores__{safe}"] = np.asarray(
        pack["scores"], dtype=float
    )
    development_score_npz[f"calibration_classes__{safe}"] = np.asarray(
        pack["classes"], dtype=str
    )
    development_score_npz[f"calibration_true_labels__{safe}"] = np.asarray(
        pack["y_true"], dtype=str
    )
    development_score_npz[f"calibration_score_type__{safe}"] = np.asarray(
        [pack["score_type"]], dtype=str
    )
if not development_score_npz:
    raise RuntimeError("No development score evidence was prepared for export.")
np.savez_compressed(
    REVISION_OUTPUT_DIR / "development_scores_for_selection_and_calibration.npz",
    **development_score_npz,
)

# Final reference and all-seed predictions.
seed_prediction_rows = []
for (model_name, seed_value), prediction in final_test_predictions_by_seed.items():
    seed_frame = test_sent[["pubmed_id", "file_order", "label"]].rename(
        columns={"label": "true_label"}
    ).copy()
    seed_frame["model"] = model_name
    seed_frame["seed"] = str(seed_value)
    seed_frame["is_reference_seed"] = str(seed_value) == str(
        REFERENCE_PREDICTION_SEED.get(model_name)
    )
    seed_frame["prediction"] = np.asarray(prediction)
    seed_prediction_rows.append(seed_frame)
final_test_predictions_by_seed_long_df = pd.concat(seed_prediction_rows, ignore_index=True)
final_test_predictions_by_seed_long_df.to_csv(
    REVISION_OUTPUT_DIR / "final_test_predictions_by_seed_long.csv", index=False
)

prediction_export = test_sent[[
    "pubmed_id", "file_order", "sentence_index_in_abstract",
    "relative_position", "label", "text",
]].copy()
prediction_export.insert(0, "split", "test")
for model_name, prediction in final_test_predictions.items():
    prediction_export[f"pred__{model_name}__reference_seed_{REFERENCE_PREDICTION_SEED[model_name]}"] = np.asarray(prediction)
prediction_export.to_csv(
    REVISION_OUTPUT_DIR / "final_test_item_predictions.csv", index=False
)

score_npz = {}
for model_name, pack in final_test_scores.items():
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    score_npz[f"scores__{safe}"] = np.asarray(pack["scores"])
    score_npz[f"classes__{safe}"] = np.asarray(pack["classes"], dtype=str)
    score_npz[f"score_type__{safe}"] = np.asarray([pack["score_type"]], dtype=str)
    score_npz[f"reference_seed__{safe}"] = np.asarray(
        [str(REFERENCE_PREDICTION_SEED[model_name])], dtype=str
    )
for model_name, probabilities in calibrated_test_probabilities.items():
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    score_npz[f"raw_probabilities__{safe}"] = np.asarray(
        raw_test_probabilities[model_name], dtype=float
    )
    score_npz[f"temperature_scaled_probabilities__{safe}"] = np.asarray(
        probabilities, dtype=float
    )
    score_npz[f"probability_classes__{safe}"] = np.asarray(
        LABEL_ORDER, dtype=str
    )
np.savez_compressed(
    REVISION_OUTPUT_DIR / "final_test_scores_and_probabilities.npz", **score_npz
)

bootstrap_long = pd.concat([
    pd.DataFrame({
        "model": model_name, "replicate": np.arange(len(values)),
        "macro_f1": values, "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
        "reference_seed": REFERENCE_PREDICTION_SEED[model_name],
    })
    for model_name, values in bootstrap_distributions.items()
], ignore_index=True)
bootstrap_long.to_csv(
    REVISION_OUTPUT_DIR / "abstract_cluster_bootstrap_distributions.csv", index=False
)

# Training-only CV and selection provenance.
classical_cv_summary.to_csv(
    REVISION_OUTPUT_DIR / "classical_nested_cv_summary.csv", index=False
)
classical_full_search_best_params_df.to_csv(
    REVISION_OUTPUT_DIR / "full_training_grouped_search_best_params.csv", index=False
)
dev_results.to_csv(
    REVISION_OUTPUT_DIR / "development_model_results.csv", index=False
)
final_system_selection_audit_df.to_csv(
    REVISION_OUTPUT_DIR / "final_system_selection_audit.csv", index=False
)
transformer_truncation_audit_df.to_csv(
    REVISION_OUTPUT_DIR / "transformer_truncation_audit.csv", index=False
)
bilstm_truncation_audit_df.to_csv(
    REVISION_OUTPUT_DIR / "bilstm_truncation_audit.csv", index=False
)

dynamic_cv_artifacts = []
for model_name, fold_df in classical_cv_results.items():
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    fold_name = f"nested_grouped_cv_fold_metrics__{safe}.csv"
    oof_name = f"nested_grouped_cv_oof_predictions__{safe}.csv"
    params_name = f"nested_grouped_cv_inner_best_params__{safe}.csv"
    fold_df.to_csv(REVISION_OUTPUT_DIR / fold_name, index=False)
    cv_fold_predictions[model_name].to_csv(REVISION_OUTPUT_DIR / oof_name, index=False)
    classical_nested_params[model_name].to_csv(
        REVISION_OUTPUT_DIR / params_name, index=False
    )
    dynamic_cv_artifacts.extend([fold_name, oof_name, params_name])

grouped_ablation_results.to_csv(
    REVISION_OUTPUT_DIR / "controlled_grouped_ablation_results.csv", index=False
)
position_weight_factorial_results.to_csv(
    REVISION_OUTPUT_DIR / "position_weight_factorial_results.csv", index=False
)
grouped_ablation_bootstrap_distributions_df.to_csv(
    REVISION_OUTPUT_DIR / "controlled_grouped_ablation_bootstrap_distributions.csv",
    index=False, lineterminator="\n",
)
position_weight_factorial_cells_df.to_csv(
    REVISION_OUTPUT_DIR / "position_weight_factorial_cells.csv",
    index=False, lineterminator="\n",
)
position_weight_factorial_bootstrap_distributions_df.to_csv(
    REVISION_OUTPUT_DIR / "position_weight_factorial_bootstrap_distributions.csv",
    index=False, lineterminator="\n",
)
fixed_ablation_oof_parts = []
for analysis_set, prediction_mapping in [
    ("fixed_feature_ablation", fixed_ablation_predictions),
    ("position_weight_factorial", position_weight_factorial),
]:
    for estimand_name, prediction_frame in prediction_mapping.items():
        exported = prediction_frame.copy()
        exported["analysis_set"] = analysis_set
        exported["estimand_name"] = estimand_name
        fixed_ablation_oof_parts.append(exported)
fixed_ablation_oof_predictions_long_df = pd.concat(
    fixed_ablation_oof_parts, ignore_index=True
)
if fixed_ablation_oof_predictions_long_df.duplicated(
    ["analysis_set", "estimand_name", "row_index"]
).any():
    raise AssertionError("Fixed-ablation OOF keys are duplicated.")
fixed_ablation_oof_predictions_long_df.to_csv(
    REVISION_OUTPUT_DIR / "fixed_ablation_oof_predictions_long.csv",
    index=False, lineterminator="\n",
)
bootstrap_summary_df.to_csv(
    REVISION_OUTPUT_DIR / "final_test_cluster_bootstrap_summary.csv", index=False
)
paired_bootstrap_comparison_df.to_csv(
    REVISION_OUTPUT_DIR / "final_test_paired_cluster_bootstrap.csv", index=False
)
paired_bootstrap_distributions_df.to_csv(
    REVISION_OUTPUT_DIR / "final_test_paired_cluster_bootstrap_distributions.csv", index=False
)
deep_seed_summary_df.to_csv(
    REVISION_OUTPUT_DIR / "deep_model_seed_summary.csv", index=False
)
final_test_results_df.to_csv(
    REVISION_OUTPUT_DIR / "final_test_results_by_seed.csv", index=False
)
revision_efficiency_df.to_csv(
    REVISION_OUTPUT_DIR / "implementation_specific_efficiency.csv", index=False
)
revision_efficiency_repeats_df.to_csv(
    REVISION_OUTPUT_DIR / "implementation_specific_efficiency_repeats.csv", index=False
)
final_test_class_counts_df.to_csv(
    REVISION_OUTPUT_DIR / "final_test_class_counts.csv", index=False
)

# Copy exact run environment records into the deposit-ready artifact directory.
for environment_name in [
    "environment_snapshot.json", "requirements_snapshot.txt", "compute_hardware_snapshot.json",
]:
    source_path = OUTPUT_DIR / environment_name
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise RuntimeError(f"Missing environment record: {source_path}")
    shutil.copy2(source_path, REVISION_OUTPUT_DIR / environment_name)

# Preserve the exact pre-test selection/comparison lock in the run deposit.
for lock_source in [PRETEST_SELECTION_CSV, PRETEST_COMPARISON_CSV, PRETEST_LOCK_JSON]:
    if not Path(lock_source).is_file() or Path(lock_source).stat().st_size <= 0:
        raise RuntimeError(f"Pre-test lock artifact is missing: {lock_source}")
    shutil.copy2(lock_source, REVISION_OUTPUT_DIR / Path(lock_source).name)
if sha256_file(REVISION_OUTPUT_DIR / PRETEST_LOCK_JSON.name) != PRETEST_SELECTION_LOCK_SHA256:
    raise RuntimeError("Copied pre-test lock hash differs from the in-memory lock hash.")

run_metadata = {
    "run_id": RUN_ID,
    "publication_run": PUBLICATION_RUN,
    "smoke_run": SMOKE_RUN,
    "export_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256 if SOURCE_NOTEBOOK_SHA256 else None,
    "authoritative_data_verified": all(row["verified"] for row in PUBLICATION_FILE_PREFLIGHT),
    "final_systems_locked_before_test": FINAL_SYSTEMS_LOCKED,
    "reference_seed": REFERENCE_SEED,
    "deep_seeds": REVISION_DEEP_SEEDS,
    "final_systems": FINAL_SYSTEMS,
    "primary_transformer": PRIMARY_TRANSFORMER,
    "pretest_selection_comparison_lock_sha256": PRETEST_SELECTION_LOCK_SHA256,
    "publication_figure_font": PLOS_FIGURE_FONT,
    "publication_figure_font_file": str(PLOS_FIGURE_FONT_FILE),
    "fixed_test_disclosure": (
        "public benchmark test split was evaluated in the original submission; "
        "corrected-run choices and comparisons were locked before predictions"
    ),
    "perturbation_interpretation": "prediction sensitivity and consistency; not semantic correctness",
}
(REVISION_OUTPUT_DIR / "run_metadata.json").write_text(
    json.dumps(run_metadata, indent=2), encoding="utf-8"
)

# Validate every generated publication figure and its source record.
figure_artifact_names = sorted(set(generated_figure_artifacts))
if not figure_artifact_names:
    raise RuntimeError("No publication figure artifacts were generated.")
figure_manifest_rows = []
for artifact_name in figure_artifact_names:
    artifact_path = REVISION_OUTPUT_DIR / artifact_name
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        raise RuntimeError(f"Generated figure/source artifact is missing or empty: {artifact_path}")
    row = {
        "artifact": artifact_name, "suffix": artifact_path.suffix.lower(),
        "bytes": artifact_path.stat().st_size, "sha256": sha256_file(artifact_path),
        "dpi_x": np.nan, "dpi_y": np.nan, "compression": "not_applicable",
        "image_mode": "not_applicable", "n_frames": np.nan,
        "width_px": np.nan, "height_px": np.nan,
        "width_in": np.nan, "height_in": np.nan,
    }
    if artifact_path.suffix.lower() == ".tif":
        from PIL import Image
        with Image.open(artifact_path) as image:
            dpi = image.info.get("dpi", (np.nan, np.nan))
            row["dpi_x"], row["dpi_y"] = float(dpi[0]), float(dpi[1])
            row["compression"] = str(image.info.get("compression", ""))
            row["image_mode"] = str(image.mode)
            row["n_frames"] = int(getattr(image, "n_frames", 1))
            row["width_px"], row["height_px"] = int(image.width), int(image.height)
        row["width_in"] = row["width_px"] / row["dpi_x"]
        row["height_in"] = row["height_px"] / row["dpi_y"]
        if not (300 <= row["dpi_x"] <= 600 and 300 <= row["dpi_y"] <= 600):
            raise RuntimeError(f"TIFF resolution is outside 300-600 dpi: {artifact_path}")
        if not (2.63 <= row["width_in"] <= 7.5) or row["height_in"] > 8.75:
            raise RuntimeError(f"TIFF physical dimensions are outside PLOS limits: {artifact_path}")
        if row["compression"].lower() not in {"tiff_lzw", "lzw"}:
            raise RuntimeError(f"TIFF is not LZW compressed: {artifact_path}")
        if row["image_mode"] not in {"RGB", "L"} or row["n_frames"] != 1:
            raise RuntimeError(f"TIFF must be single-frame RGB or grayscale: {artifact_path}")
        if artifact_path.stat().st_size >= 10 * 1024 * 1024:
            raise RuntimeError(f"TIFF exceeds the 10 MB PLOS limit: {artifact_path}")
    figure_manifest_rows.append(row)
figure_export_manifest_df = pd.DataFrame(figure_manifest_rows)
figure_export_manifest_df.to_csv(
    REVISION_OUTPUT_DIR / "figure_export_manifest.csv", index=False, lineterminator="\n"
)
CANONICAL_SUPPORTING_FIGURE_ARTIFACTS = [
    *[f"S1_fig.{suffix}" for suffix in ["png", "pdf", "tif"]],
    "S1_fig_source.csv",
    *[f"S2_fig.{suffix}" for suffix in ["png", "pdf", "tif"]],
    "S2_fig_source.csv",
    *[f"S3_fig.{suffix}" for suffix in ["png", "pdf", "tif"]],
    "S3_fig_source.csv",
]
for artifact_name in CANONICAL_FIG1_ARTIFACTS + CANONICAL_SUPPORTING_FIGURE_ARTIFACTS:
    if artifact_name not in set(figure_artifact_names):
        raise RuntimeError(f"Canonical publication figure/source artifact is absent: {artifact_name}")

# Validate in-memory statistical and evidentiary contracts before manifesting files.
def _require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


if PUBLICATION_RUN and not re.fullmatch(r"[0-9a-f]{64}", str(SOURCE_NOTEBOOK_SHA256)):
    raise RuntimeError("Publication mode requires a certified lowercase 64-hex source SHA-256.")
if not FINAL_SYSTEMS_LOCKED or not TEST_OUTCOMES_ACCESSED:
    raise RuntimeError("Final export requires the durable lock and completed corrected-run test access.")
if _sha256_file(PRETEST_LOCK_JSON) != PRETEST_SELECTION_LOCK_SHA256:
    raise RuntimeError("Pre-test lock changed before the final gate.")
final_lock_payload = json.loads(PRETEST_LOCK_JSON.read_text(encoding="utf-8"))
if _sha256_file(PRETEST_SELECTION_CSV) != final_lock_payload["selection_csv_sha256"]:
    raise RuntimeError("Pre-test system-selection CSV changed before the final gate.")
if _sha256_file(PRETEST_COMPARISON_CSV) != final_lock_payload["comparison_csv_sha256"]:
    raise RuntimeError("Pre-test comparison-definition CSV changed before the final gate.")
if set(final_test_predictions) != set(FINAL_SYSTEMS):
    raise RuntimeError("Final reference-prediction coverage differs from FINAL_SYSTEMS.")
if set(final_test_scores) != set(FINAL_SYSTEMS):
    raise RuntimeError("Final score coverage differs from FINAL_SYSTEMS.")
if set(final_fitted_systems) != set(FINAL_SYSTEMS):
    raise RuntimeError("Retained fitted-system coverage differs from FINAL_SYSTEMS.")

n_test = len(test_sent)
for model_name in FINAL_SYSTEMS:
    prediction = np.asarray(final_test_predictions[model_name]).astype(str)
    if len(prediction) != n_test or not set(prediction).issubset(set(LABEL_ORDER)):
        raise RuntimeError(f"Invalid prediction vector for {model_name}.")
    pack = final_test_scores[model_name]
    scores = np.asarray(pack["scores"], dtype=float)
    classes = [str(value) for value in pack["classes"]]
    if scores.shape != (n_test, len(LABEL_ORDER)) or not np.isfinite(scores).all():
        raise RuntimeError(f"Invalid score matrix for {model_name}: {scores.shape}")
    if set(classes) != set(LABEL_ORDER) or not str(pack.get("score_type", "")):
        raise RuntimeError(f"Invalid score classes/semantics for {model_name}.")
    # Preserve each estimator's native class order for exact-score ties.
    # Reordering columns before argmax changes the tie break without changing scores.
    score_prediction = np.asarray(classes)[np.argmax(scores, axis=1)]
    if not np.array_equal(score_prediction, prediction):
        raise RuntimeError(f"Raw-score argmax does not match predictions for {model_name}.")

classical_systems = {
    name for name in FINAL_SYSTEMS if final_fitted_systems[name]["family"] == "classical"
}
deep_systems = set(FINAL_SYSTEMS) - classical_systems
expected_seed_keys = {
    (name, "deterministic") for name in classical_systems
}.union({
    (name, str(seed)) for name in deep_systems for seed in REVISION_DEEP_SEEDS
})
prediction_seed_keys = {
    (str(model), str(seed)) for model, seed in final_test_predictions_by_seed
}
if prediction_seed_keys != expected_seed_keys:
    raise RuntimeError(
        f"All-seed prediction coverage mismatch: expected={expected_seed_keys}, observed={prediction_seed_keys}"
    )
result_key_frame = final_test_results_df[["model_name", "seed"]].astype(str)
if result_key_frame.duplicated().any():
    raise RuntimeError("Duplicate (model_name, seed) final-result rows detected.")
if set(map(tuple, result_key_frame.to_numpy())) != expected_seed_keys:
    raise RuntimeError("Final-result model/seed coverage differs from the prespecified contract.")

_require_columns(
    per_class_test_metrics_df,
    ["model", "class", "reference_seed", "precision", "recall", "F1", "support"],
    "per-class test metrics",
)
if per_class_test_metrics_df.groupby("model").size().to_dict() != {
    name: len(LABEL_ORDER) for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Per-class metrics do not contain exactly five rows per final system.")
if final_test_confusion_long_df.groupby("model").size().to_dict() != {
    name: len(LABEL_ORDER) ** 2 for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Confusion sources do not contain a complete 5 x 5 grid per system.")
if calibration_metrics_df.groupby("model").size().to_dict() != {
    name: 2 for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Calibration metrics require two states per final system.")
if calibration_classwise_ece_df.groupby("model").size().to_dict() != {
    name: 2 * len(LABEL_ORDER) for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Classwise ECE requires every class in both calibration states.")
if bootstrap_long.groupby("model").size().to_dict() != {
    name: BOOTSTRAP_N_REPLICATES for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Final cluster-bootstrap distributions are incomplete.")
if paired_bootstrap_comparison_df["formal_inference_allowed"].sum() != 1:
    raise RuntimeError("Exactly one paired comparison must permit formal inference.")
paired_counts = paired_bootstrap_distributions_df.groupby(
    ["Model A", "Model B", "Comparison family"]
).size()
if len(paired_counts) != len(pretest_comparison_lock_df) or not paired_counts.eq(
    BOOTSTRAP_N_REPLICATES
).all():
    raise RuntimeError("Paired bootstrap distributions are incomplete.")

expected_perturbation_rows_per_model = int(sum(expected_counts.values()))
if context_faithful_perturbation_predictions_df.groupby("model").size().to_dict() != {
    name: expected_perturbation_rows_per_model for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Perturbation item-level model coverage is incomplete.")
if context_faithful_perturbation_summary_df.groupby("model").size().to_dict() != {
    name: len(PERTURBATION_RULES) for name in FINAL_SYSTEMS
}:
    raise RuntimeError("Perturbation summary strata are incomplete.")
if context_faithful_perturbation_summary_df["n_abstracts"].le(0).any():
    raise RuntimeError("A perturbation summary stratum has zero abstract clusters.")
perturbation_distribution_counts = (
    context_faithful_perturbation_cluster_bootstrap_distributions_df
    .groupby(["model", "perturbation_type"]).size()
)
if len(perturbation_distribution_counts) != len(FINAL_SYSTEMS) * len(PERTURBATION_RULES) or not perturbation_distribution_counts.eq(
    PERTURBATION_BOOTSTRAP_N
).all():
    raise RuntimeError("Perturbation cluster-bootstrap distributions are incomplete.")
_require_columns(
    context_faithful_perturbation_predictions_df,
    [
        "model", "benchmark_item_id", "perturbation_type", "pubmed_id",
        "pred_original", "pred_perturbed", "prediction_changed",
        "prediction_consistent", "score_type", "score_class_order",
        "n_operations", "rule_branch",
    ] + [f"score_original__{label}" for label in LABEL_ORDER]
      + [f"score_perturbed__{label}" for label in LABEL_ORDER],
    "perturbation item predictions",
)

for model_name in classical_cv_results:
    fold_frame = classical_cv_results[model_name]
    oof_frame = cv_fold_predictions[model_name]
    if len(fold_frame) != N_FOLDS_CV or set(fold_frame["fold"]) != set(range(1, N_FOLDS_CV + 1)):
        raise RuntimeError(f"Nested grouped-CV folds are incomplete for {model_name}.")
    if len(oof_frame) != len(classical_train_df) or oof_frame["row_index"].duplicated().any():
        raise RuntimeError(f"Nested grouped-CV OOF coverage is incomplete for {model_name}.")
if transformer_truncation_audit_df.duplicated(
    ["model", "phase", "split", "reference_seed"]
).any() or bilstm_truncation_audit_df.duplicated(
    ["model", "phase", "split", "reference_seed"]
).any():
    raise RuntimeError("Truncation-audit keys are duplicated.")

expected_development_models = set(CLASSICAL_MODEL_SPECS).union({
    "S7_CRF_SequenceModel", "S8_BiLSTMStyle_SequentialModel"
}).union(set(transformer_development_predictions))
development_counts = development_item_predictions_long_df.groupby("model").size().to_dict()
if set(development_counts) != expected_development_models or any(
    count != len(dev_sent) for count in development_counts.values()
):
    raise RuntimeError("Item-level development evidence is incomplete.")
for model_name, candidate_scores in transformer_development_scores.items():
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    candidate_classes = transformer_development_score_classes[model_name]
    if candidate_scores.shape != (len(dev_sent), len(LABEL_ORDER)) or not np.isfinite(
        candidate_scores
    ).all() or set(candidate_classes) != set(LABEL_ORDER):
        raise RuntimeError(f"Transformer development scores are invalid for {model_name}.")
if np.asarray(bilstm_development_scores).shape != (len(dev_sent), len(LABEL_ORDER)) or not np.isfinite(
    np.asarray(bilstm_development_scores, dtype=float)
).all() or set(bilstm_development_score_classes) != set(LABEL_ORDER):
    raise RuntimeError("BiLSTM development scores/classes are invalid.")
if len(position_weight_factorial_cells_df) != 4 or len(position_weight_factorial_results) != 7:
    raise RuntimeError("The 2 x 2 position/weight factorial export is incomplete.")
if len(grouped_ablation_bootstrap_distributions_df) != 2 * ABLATION_BOOTSTRAP_N:
    raise RuntimeError("Controlled-ablation bootstrap distributions are incomplete.")
if len(position_weight_factorial_bootstrap_distributions_df) != 7 * ABLATION_BOOTSTRAP_N:
    raise RuntimeError("Factorial bootstrap distributions are incomplete.")
if fixed_ablation_oof_predictions_long_df.empty:
    raise RuntimeError("Fixed-ablation OOF evidence is empty.")

required_outputs = [
    "dataset_integrity.csv", "dataset_split_abstract_overlap.csv",
    "dataset_file_sha256.csv", "dataset_split_class_counts.csv",
    "model_reproducibility_specification.csv", "software_hardware_environment.csv",
    "classical_nested_cv_summary.csv", "full_training_grouped_search_best_params.csv",
    "development_model_results.csv", "development_item_predictions_long.csv",
    "development_scores_for_selection_and_calibration.npz",
    "final_system_selection_audit.csv", "final_system_selection_audit_pretest.csv",
    "comparison_definitions_pretest.csv", "SELECTION_AND_COMPARISON_LOCK.json",
    "transformer_truncation_audit.csv", "bilstm_truncation_audit.csv",
    "environment_snapshot.json", "requirements_snapshot.txt", "compute_hardware_snapshot.json",
    "run_metadata.json", "final_test_item_predictions.csv",
    "final_test_predictions_by_seed_long.csv", "final_test_scores_and_probabilities.npz",
    "per_class_test_precision_recall_f1.csv", "all_final_test_confusion_matrices_long.csv",
    "calibration_and_complementary_metrics.csv", "calibration_classwise_ece.csv",
    "calibration_reliability_bins.csv", "final_test_cluster_bootstrap_summary.csv",
    "final_test_paired_cluster_bootstrap.csv",
    "final_test_paired_cluster_bootstrap_distributions.csv",
    "abstract_cluster_bootstrap_distributions.csv", "deep_model_seed_summary.csv",
    "final_test_results_by_seed.csv", "final_test_class_counts.csv",
    "implementation_specific_efficiency.csv",
    "implementation_specific_efficiency_repeats.csv",
    "context_faithful_perturbation_item_predictions.csv",
    "context_faithful_perturbation_summary.csv",
    "context_faithful_perturbation_cluster_bootstrap_distributions.csv",
    "context_faithful_perturbation_eligibility.csv",
    "controlled_grouped_ablation_results.csv",
    "controlled_grouped_ablation_bootstrap_distributions.csv",
    "fixed_ablation_oof_predictions_long.csv",
    "position_weight_factorial_cells.csv", "position_weight_factorial_results.csv",
    "position_weight_factorial_bootstrap_distributions.csv",
    "figure_export_manifest.csv",
]
required_outputs.extend(sorted(set(dynamic_cv_artifacts)))
required_outputs.extend(figure_artifact_names)
required_outputs = list(dict.fromkeys(required_outputs))


def inspect_artifact(path):
    row_count, schema = None, None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        row_count = 0
        columns = None
        for chunk in pd.read_csv(path, chunksize=100000):
            if columns is None:
                columns = chunk.columns.tolist()
            row_count += len(chunk)
        schema = json.dumps(columns or [], ensure_ascii=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            schema = json.dumps(sorted(archive.files))
            row_count = len(archive.files)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = json.dumps(sorted(payload) if isinstance(payload, dict) else [type(payload).__name__])
        row_count = len(payload) if hasattr(payload, "__len__") else 1
    return row_count, schema


manifest_rows = []
for artifact_name in required_outputs:
    artifact_path = REVISION_OUTPUT_DIR / artifact_name
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        raise RuntimeError(f"Required artifact is missing or empty: {artifact_path}")
    row_count, schema = inspect_artifact(artifact_path)
    if artifact_path.suffix.lower() in {".csv", ".npz", ".json"} and (
        row_count is None or int(row_count) <= 0
    ):
        raise RuntimeError(f"Required structured artifact has no data records: {artifact_path}")
    manifest_rows.append({
        "run_id": RUN_ID, "artifact": artifact_name,
        "relative_path": artifact_name, "bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path), "row_count": row_count,
        "schema": schema, "publication_run": PUBLICATION_RUN,
        "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256 if SOURCE_NOTEBOOK_SHA256 else None,
    })

revision_export_manifest_df = pd.DataFrame(manifest_rows)
revision_export_manifest_path = REVISION_OUTPUT_DIR / "revision_export_manifest.csv"
revision_export_manifest_temporary = revision_export_manifest_path.with_name(
    revision_export_manifest_path.name + ".tmp"
)
revision_export_manifest_df.to_csv(
    revision_export_manifest_temporary, index=False, lineterminator="\n"
)
if (
    not revision_export_manifest_temporary.is_file()
    or revision_export_manifest_temporary.stat().st_size <= 0
):
    raise RuntimeError("Temporary revision manifest was not written.")
os.replace(revision_export_manifest_temporary, revision_export_manifest_path)
if revision_export_manifest_df["sha256"].str.fullmatch(r"[0-9a-f]{64}").ne(True).any():
    raise RuntimeError("One or more artifact hashes are invalid.")
if revision_export_manifest_df["bytes"].le(0).any():
    raise RuntimeError("One or more manifested artifacts are empty.")

if revision_export_manifest_df["artifact"].duplicated().any():
    raise RuntimeError("The revision manifest contains duplicate artifact names.")
if set(revision_export_manifest_df["artifact"]) != set(required_outputs):
    missing = sorted(set(required_outputs) - set(revision_export_manifest_df["artifact"]))
    extra = sorted(set(revision_export_manifest_df["artifact"]) - set(required_outputs))
    raise RuntimeError(
        f"Manifest coverage differs from the required-output contract; "
        f"missing={missing}, extra={extra}"
    )
if set(revision_export_manifest_df["source_notebook_release_id"]) != {
    SOURCE_NOTEBOOK_RELEASE_ID
}:
    raise RuntimeError("Manifest release provenance is inconsistent.")
if set(revision_export_manifest_df["source_notebook_sha256"]) != {
    SOURCE_NOTEBOOK_SHA256
}:
    raise RuntimeError("Manifest source-SHA provenance is inconsistent.")

completion_status = (
    "publication_complete" if PUBLICATION_RUN
    else "smoke_complete_not_for_publication"
)
completion_sentinel_name = (
    "PUBLICATION_RUN_COMPLETE.json" if PUBLICATION_RUN
    else "SMOKE_RUN_COMPLETE_NOT_FOR_PUBLICATION.json"
)
completion_sentinel = {
    "completion_contract_version": "plos_publication_completion_v3",
    "status": completion_status, "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "publication_run": PUBLICATION_RUN, "smoke_run": SMOKE_RUN,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256 if SOURCE_NOTEBOOK_SHA256 else None,
    "revision_export_manifest": revision_export_manifest_path.name,
    "revision_export_manifest_sha256": sha256_file(revision_export_manifest_path),
    "n_manifested_artifacts": int(len(revision_export_manifest_df)),
    "authoritative_data_verified": all(row["verified"] for row in PUBLICATION_FILE_PREFLIGHT),
    "final_systems_locked_before_test": FINAL_SYSTEMS_LOCKED,
    "requested_gpu_type": "G4",
    "actual_gpu_name": PUBLICATION_GPU_PREFLIGHT["device_name"],
    "actual_gpu_vram_gb": PUBLICATION_GPU_PREFLIGHT["device_vram_gb"],
    "compute_capability": PUBLICATION_GPU_PREFLIGHT["compute_capability"],
    "bootstrap_n_replicates": BOOTSTRAP_N_REPLICATES,
    "bootstrap_batch_size": BOOTSTRAP_BATCH_SIZE,
    "time_limit_policy": "no_wall_clock_abort",
}
sentinel_path = REVISION_OUTPUT_DIR / completion_sentinel_name
sentinel_temporary = sentinel_path.with_name(sentinel_path.name + ".tmp")
sentinel_temporary.write_text(
    json.dumps(completion_sentinel, indent=2), encoding="utf-8"
)
if not sentinel_temporary.is_file() or sentinel_temporary.stat().st_size <= 0:
    raise RuntimeError("Temporary completion sentinel was not written.")
os.replace(sentinel_temporary, sentinel_path)
if not sentinel_path.is_file() or sentinel_path.stat().st_size <= 0:
    raise RuntimeError("Completion sentinel was not written.")

sentinel_readback = json.loads(sentinel_path.read_text(encoding="utf-8"))
for key, expected in {
    "completion_contract_version": "plos_publication_completion_v3",
    "status": completion_status,
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "revision_export_manifest_sha256": sha256_file(revision_export_manifest_path),
    "n_manifested_artifacts": int(len(revision_export_manifest_df)),
}.items():
    if sentinel_readback.get(key) != expected:
        raise RuntimeError(
            f"Completion-sentinel read-back mismatch for {key}: "
            f"{sentinel_readback.get(key)!r}"
        )

final_stage_status = {
    "latest_completed_stage": completion_status,
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "n_result_rows": int(len(final_test_results_df)),
    "n_manifested_artifacts": int(len(revision_export_manifest_df)),
    "revision_export_manifest": str(revision_export_manifest_path),
    "revision_export_manifest_sha256": sha256_file(revision_export_manifest_path),
    "completion_sentinel": str(sentinel_path),
}
_atomic_stage_json(final_stage_status, FINAL_STAGE_DIR / "stage_status.json")

display(revision_export_manifest_df)
print("All required computational exports are present and hashed.")
print("Completion sentinel:", sentinel_path)
if not PUBLICATION_RUN:
    print("WARNING: this is a smoke run and cannot supply manuscript results.")

# ============================================================
# TERMINAL PUBLICATION READ-BACK AUDIT
# ============================================================
# This cell performs no analysis and writes no replacement evidence. It
# independently reopens the manifest and sentinel emitted by the fail-closed
# gate and verifies their coverage and hashes from disk.
terminal_manifest_path = REVISION_OUTPUT_DIR / "revision_export_manifest.csv"
terminal_sentinel_path = REVISION_OUTPUT_DIR / "PUBLICATION_RUN_COMPLETE.json"
if not terminal_manifest_path.is_file() or terminal_manifest_path.stat().st_size <= 0:
    raise RuntimeError("Terminal read-back audit cannot find a non-empty manifest.")
if not terminal_sentinel_path.is_file() or terminal_sentinel_path.stat().st_size <= 0:
    raise RuntimeError("Terminal read-back audit cannot find the completion sentinel.")

terminal_manifest = pd.read_csv(terminal_manifest_path)
terminal_sentinel = json.loads(
    terminal_sentinel_path.read_text(encoding="utf-8")
)
if terminal_manifest.empty or terminal_manifest["artifact"].duplicated().any():
    raise RuntimeError("Terminal manifest is empty or contains duplicate artifacts.")
if set(terminal_manifest["run_id"].astype(str)) != {RUN_ID}:
    raise RuntimeError("Terminal manifest run ID does not match the active run.")
if set(terminal_manifest["source_notebook_release_id"].astype(str)) != {
    SOURCE_NOTEBOOK_RELEASE_ID
}:
    raise RuntimeError("Terminal manifest release does not match the active source.")
if set(terminal_manifest["source_notebook_sha256"].astype(str)) != {
    SOURCE_NOTEBOOK_SHA256
}:
    raise RuntimeError("Terminal manifest source SHA does not match the active source.")

terminal_required_categories = {
    "final_test_cluster_bootstrap_summary.csv",
    "final_test_paired_cluster_bootstrap.csv",
    "final_test_paired_cluster_bootstrap_distributions.csv",
    "abstract_cluster_bootstrap_distributions.csv",
    "classical_nested_cv_summary.csv",
    "controlled_grouped_ablation_results.csv",
    "controlled_grouped_ablation_bootstrap_distributions.csv",
    "fixed_ablation_oof_predictions_long.csv",
    "position_weight_factorial_cells.csv",
    "position_weight_factorial_results.csv",
    "position_weight_factorial_bootstrap_distributions.csv",
}
missing_terminal_categories = sorted(
    terminal_required_categories - set(terminal_manifest["artifact"])
)
if missing_terminal_categories:
    raise RuntimeError(
        f"Terminal manifest omits required reviewer-facing analyses: "
        f"{missing_terminal_categories}"
    )

for row in terminal_manifest.itertuples(index=False):
    artifact_path = REVISION_OUTPUT_DIR / row.relative_path
    if not artifact_path.is_file() or artifact_path.stat().st_size != int(row.bytes):
        raise RuntimeError(f"Terminal artifact size mismatch: {artifact_path}")
    if sha256_file(artifact_path) != str(row.sha256):
        raise RuntimeError(f"Terminal artifact hash mismatch: {artifact_path}")

expected_terminal_sentinel = {
    "completion_contract_version": "plos_publication_completion_v3",
    "status": "publication_complete",
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "revision_export_manifest": terminal_manifest_path.name,
    "revision_export_manifest_sha256": sha256_file(terminal_manifest_path),
    "n_manifested_artifacts": int(len(terminal_manifest)),
}
for key, expected_value in expected_terminal_sentinel.items():
    if terminal_sentinel.get(key) != expected_value:
        raise RuntimeError(
            f"Terminal sentinel mismatch for {key}: "
            f"expected={expected_value!r}, observed={terminal_sentinel.get(key)!r}"
        )

print(json.dumps({
    "terminal_audit": "passed",
    "run_id": RUN_ID,
    "source_notebook_release_id": SOURCE_NOTEBOOK_RELEASE_ID,
    "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
    "n_manifested_artifacts": len(terminal_manifest),
    "manifest_sha256": sha256_file(terminal_manifest_path),
    "completion_sentinel": str(terminal_sentinel_path),
}, indent=2))
