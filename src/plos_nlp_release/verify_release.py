"""Verify the lightweight, history-free GitHub release tree.

The verifier is intentionally standard-library only. It checks the repository
manifest, byte counts, SHA-256 values, release identity, selected artifact
schemas, and high-confidence indicators of credentials, personal paths, raw
benchmark text, model weights, or retired manual-review data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any


RUN_ID = "PLOS_R1_publication_20260729T084516Z"
SOURCE_RELEASE = (
    "PLOS_R1_REVIEWER_ALIGNED_G4_FINAL_EXPORT_COMPLETE_CLASSICAL_PARAMS_20260729"
)
SOURCE_NOTEBOOK_SHA256 = (
    "f41a9b1542f4a544eea2ff2085cc096743eb6f98d166a6e2dc0af4e96590ba13"
)
COMPLETION_CONTRACT = "plos_publication_completion_v3"
RELEASE_ARCHIVE_NAME = (
    "PLOS_R1_publication_20260729T084516Z_sanitized_data_code_deposit.zip"
)
RELEASE_ARCHIVE_BYTES = 64_461_196
RELEASE_ARCHIVE_SHA256 = (
    "b6f4a158891784ea5f6f451d33d4e3a46ff42406afa582553a01f2fa381fddf4"
)
PUBLIC_DEPOSIT_MANIFEST_BYTES = 76_452
PUBLIC_DEPOSIT_MANIFEST_SHA256 = (
    "27fdd8bec132208474377209f128f138f0e71fd28d111d554304dfbf79af7e37"
)
OUTER_CHECKSUMS_BYTES = 229
OUTER_CHECKSUMS_SHA256 = (
    "6f160158acba129753954b5aa2caff217c24f627a977d40611181039db8be56a"
)
CC_BY_4_LEGALCODE_SHA256 = (
    "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"
)
MAX_GITHUB_FILE_BYTES = 90 * 1024 * 1024

MANIFEST_COLUMNS = ("path", "bytes", "sha256", "role", "source_origin")
IGNORED_DIRECTORY_NAMES_ANYWHERE = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
IGNORED_ROOT_DIRECTORY_NAMES = {
    ".nox",
    ".tox",
    ".venv",
    "build",
    "dist",
    "htmlcov",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
FORBIDDEN_BASENAMES = {
    "train.txt",
    "dev.txt",
    "test.txt",
    "arial.ttf",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".joblib",
    ".onnx",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".safetensors",
    ".ttf",
}
RAW_TEXT_COLUMNS = {
    "text",
    "sentence",
    "sentence_text",
    "raw_text",
    "source_text",
    "original_text",
    "perturbed_text",
    "previous_sentence",
    "current_sentence",
}
RETIRED_DATA_NAME_RE = re.compile(
    r"(?i)(?:clinician|manual[_ .-]?annotation|human[_ .-]?annotation|adjudicat)"
)
PERSONAL_PATH_PATTERNS = (
    re.compile(r"(?i)(?:[A-Za-z]:[\\/])Users[\\/][^\\/\s\"']+"),
    re.compile(r"(?i)/(?:Users|home)/[^/\s\"']+"),
)
SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github_fine_grained_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "token_bearing_url",
        re.compile(
            r"(?i)https?://[^\s\"'<>]+[?&]"
            r"(?:access_token|api_key|client_secret|password|signature)="
            r"[^&\s\"'<>]+"
        ),
    ),
)
ZENODO_DOI_PATTERN = re.compile(r"\b10\.5281/zenodo\.\d+\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_relative_path(raw: str) -> str:
    if not raw or "\\" in raw:
        raise ValueError(f"manifest path is not canonical POSIX form: {raw!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ValueError(f"manifest path is unsafe: {raw!r}")
    return candidate.as_posix()


def _iter_repository_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, names, filenames in os.walk(root):
        current = Path(directory)
        is_root = current == root
        names[:] = sorted(
            name
            for name in names
            if name not in IGNORED_DIRECTORY_NAMES_ANYWHERE
            and not (is_root and name in IGNORED_ROOT_DIRECTORY_NAMES)
            and not name.casefold().endswith(".egg-info")
        )
        for filename in sorted(filenames):
            path = current / filename
            if path.suffix.lower() not in IGNORED_SUFFIXES:
                paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def _load_manifest(root: Path) -> list[dict[str, str]]:
    manifest_path = root / "release_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("release_manifest.csv is missing")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"release_manifest.csv columns must be {MANIFEST_COLUMNS!r}; "
                f"found {reader.fieldnames!r}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("release_manifest.csv has no data rows")
    return rows


def _scan_text(path: Path, relative: str, errors: list[str]) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{relative}: expected UTF-8 text")
        return
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"{relative}: possible {label}")
    for pattern in PERSONAL_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            errors.append(f"{relative}: personal/local user path {match.group(0)!r}")


def _scan_csv_header(path: Path, relative: str, errors: list[str]) -> None:
    if path.suffix.lower() != ".csv":
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    normalized = {str(item).strip().casefold() for item in header}
    prohibited = sorted(normalized & RAW_TEXT_COLUMNS)
    if prohibited and relative.startswith(("artifacts/", "figures/")):
        errors.append(f"{relative}: raw-text column(s) present: {prohibited}")
    retired = sorted(item for item in normalized if RETIRED_DATA_NAME_RE.search(item))
    if retired and relative.startswith(("artifacts/", "figures/")):
        errors.append(f"{relative}: retired manual-review field(s) present: {retired}")


def _check_csv(
    root: Path,
    relative: str,
    required_columns: set[str],
    expected_rows: int | None,
    errors: list[str],
) -> None:
    path = root / relative
    if not path.is_file():
        errors.append(f"{relative}: required artifact is missing")
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        rows = list(reader)
    missing = sorted(required_columns - columns)
    if missing:
        errors.append(f"{relative}: missing required columns {missing}")
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"{relative}: expected {expected_rows} rows, found {len(rows)}")
    if not rows:
        errors.append(f"{relative}: artifact is empty")


def _check_semantic_contracts(root: Path, errors: list[str]) -> dict[str, Any]:
    sentinel_path = root / "provenance" / "PUBLICATION_RUN_COMPLETE.json"
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"completion sentinel is unreadable: {exc}")
        sentinel = {}
    expected = {
        "completion_contract_version": COMPLETION_CONTRACT,
        "status": "publication_complete",
        "run_id": RUN_ID,
        "source_notebook_release_id": SOURCE_RELEASE,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "authoritative_data_verified": True,
        "final_systems_locked_before_test": True,
        "bootstrap_n_replicates": 10_000,
        "time_limit_policy": "no_wall_clock_abort",
    }
    for key, value in expected.items():
        if sentinel.get(key) != value:
            errors.append(
                f"completion sentinel {key!r}: expected {value!r}, "
                f"found {sentinel.get(key)!r}"
            )

    notebook = root / "analysis" / "certified_source_notebook.ipynb"
    if not notebook.is_file():
        errors.append("analysis/certified_source_notebook.ipynb is missing")
    elif sha256_file(notebook) != SOURCE_NOTEBOOK_SHA256:
        errors.append("certified source notebook SHA-256 mismatch")

    _check_csv(
        root,
        "artifacts/summary/final_test_results_by_seed.csv",
        {"model_name", "seed", "accuracy", "macro_f1"},
        16,
        errors,
    )
    _check_csv(
        root,
        "artifacts/summary/final_test_paired_cluster_bootstrap.csv",
        {"Model A", "Model B", "Delta Macro-F1 (A-B)", "n_cluster_bootstrap"},
        4,
        errors,
    )
    _check_csv(
        root,
        "artifacts/summary/controlled_grouped_ablation_results.csv",
        {
            "feature_change",
            "delta_macro_f1_changed_minus_lexical",
            "n_cluster_bootstrap",
        },
        2,
        errors,
    )
    _check_csv(
        root,
        "artifacts/summary/position_weight_factorial_results.csv",
        {"contrast", "delta_macro_f1", "n_cluster_bootstrap"},
        7,
        errors,
    )
    _check_csv(
        root,
        "artifacts/summary/calibration_and_complementary_metrics.csv",
        {"model", "accuracy", "macro_F1"},
        None,
        errors,
    )
    _check_csv(
        root,
        "artifacts/summary/context_faithful_perturbation_summary.csv",
        {"model", "perturbation_type", "prediction_consistency_rate"},
        None,
        errors,
    )
    _check_csv(
        root,
        "artifacts/summary/per_class_test_precision_recall_f1.csv",
        {"model", "class", "precision", "recall", "F1"},
        None,
        errors,
    )
    return sentinel


def _check_license_contracts(root: Path, errors: list[str]) -> dict[str, Any]:
    if (root / "LICENSE_PENDING.md").exists():
        errors.append("obsolete LICENSE_PENDING.md is still present")

    mit_path = root / "LICENSE"
    cc_path = root / "LICENSE-CONTENT-DATA-CC-BY-4.0.txt"
    scope_path = root / "LICENSES.md"
    required = (mit_path, cc_path, scope_path)
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"{path.name}: required license file is missing or empty")

    mit_text = mit_path.read_text(encoding="utf-8") if mit_path.is_file() else ""
    if not (
        mit_text.startswith("MIT License\n")
        and "Copyright (c) 2026 Tianyi Yu" in mit_text
        and "THE SOFTWARE IS PROVIDED \"AS IS\"" in mit_text
    ):
        errors.append("LICENSE: complete MIT text/copyright notice is not recognized")

    if cc_path.is_file() and sha256_file(cc_path) != CC_BY_4_LEGALCODE_SHA256:
        errors.append("LICENSE-CONTENT-DATA-CC-BY-4.0.txt: legalcode SHA-256 mismatch")

    scope_text = (
        scope_path.read_text(encoding="utf-8") if scope_path.is_file() else ""
    )
    for phrase in (
        "Author-generated code — MIT",
        "Author-generated content and derived outputs — CC BY 4.0",
        "PubMed RCT raw or normalized sentence text",
    ):
        if phrase not in scope_text:
            errors.append(f"LICENSES.md: missing scope phrase {phrase!r}")

    citation_text = (root / "CITATION.cff").read_text(encoding="utf-8")
    if "license: MIT" not in citation_text:
        errors.append("CITATION.cff: MIT software license is not declared")
    if ZENODO_DOI_PATTERN.search(citation_text):
        errors.append(
            "CITATION.cff: Zenodo DOI must not be claimed before GitHub-release "
            "archiving and independent verification"
        )
    readme_text = (root / "README.md").read_text(encoding="utf-8")
    if (
        "[MIT License](LICENSE)" not in readme_text
        or "LICENSE-CONTENT-DATA-CC-BY-4.0.txt" not in readme_text
    ):
        errors.append("README.md: dual-license notice is incomplete")
    checklist_text = (root / "ZENODO_DEPOSIT_CHECKLIST.md").read_text(
        encoding="utf-8"
    )
    release_identity_phrases = (
        "64,461,196",
        RELEASE_ARCHIVE_SHA256,
        "76,452",
        PUBLIC_DEPOSIT_MANIFEST_SHA256,
        OUTER_CHECKSUMS_SHA256,
        "GitHub release asset",
    )
    for relative, text in (
        ("README.md", readme_text),
        ("ZENODO_DEPOSIT_CHECKLIST.md", checklist_text),
    ):
        for phrase in release_identity_phrases:
            if phrase not in text:
                errors.append(
                    f"{relative}: canonical release identity phrase is missing: "
                    f"{phrase!r}"
                )
        if "do not assume" not in text.casefold():
            errors.append(
                f"{relative}: Zenodo manual-release-asset caveat is missing"
            )
    for relative in (
        "README.md",
        "ZENODO_DEPOSIT_CHECKLIST.md",
        "SECURITY_DATA_AUDIT.json",
    ):
        text = (root / relative).read_text(encoding="utf-8").casefold()
        if any(f"{prefix}-license" in text for prefix in ("pre", "pending")):
            errors.append(f"{relative}: obsolete license-status language remains")
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    if 'license = {file = "LICENSE"}' not in pyproject_text:
        errors.append("pyproject.toml: MIT license file is not declared")

    return {
        "code_license": "MIT",
        "content_data_license": "CC-BY-4.0",
        "mit_license_sha256": sha256_file(mit_path) if mit_path.is_file() else None,
        "cc_by_4_legalcode_sha256": (
            sha256_file(cc_path) if cc_path.is_file() else None
        ),
        "scope_notice_sha256": (
            sha256_file(scope_path) if scope_path.is_file() else None
        ),
    }


def verify_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    manifest_rows = _load_manifest(root)
    by_path: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        try:
            relative = _canonical_relative_path(row["path"])
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if relative == "release_manifest.csv":
            errors.append("release_manifest.csv must not self-manifest")
        if relative in by_path:
            errors.append(f"duplicate manifest path: {relative}")
        by_path[relative] = row

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in _iter_repository_files(root)
        if path.relative_to(root).as_posix() != "release_manifest.csv"
    }
    declared_paths = set(by_path)
    for relative in sorted(declared_paths - actual_paths):
        errors.append(f"manifested file is missing: {relative}")
    for relative in sorted(actual_paths - declared_paths):
        errors.append(f"unmanifested file is present: {relative}")

    largest = {"path": None, "bytes": 0}
    for relative in sorted(actual_paths & declared_paths):
        path = root / relative
        size = path.stat().st_size
        if size > largest["bytes"]:
            largest = {"path": relative, "bytes": size}
        row = by_path[relative]
        try:
            declared_bytes = int(row["bytes"])
        except ValueError:
            errors.append(f"{relative}: invalid manifest byte count {row['bytes']!r}")
            declared_bytes = -1
        if size != declared_bytes:
            errors.append(
                f"{relative}: byte count mismatch, expected {declared_bytes}, found {size}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != row["sha256"]:
            errors.append(f"{relative}: SHA-256 mismatch")
        if size > MAX_GITHUB_FILE_BYTES:
            errors.append(f"{relative}: exceeds the 90 MiB GitHub staging limit")
        lowered = path.name.casefold()
        if lowered in FORBIDDEN_BASENAMES or path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: restricted raw-data/model/font file type")
        if RETIRED_DATA_NAME_RE.search(relative):
            errors.append(f"{relative}: retired manual-review artifact name")
        _scan_text(path, relative, errors)
        _scan_csv_header(path, relative, errors)

    sentinel = _check_semantic_contracts(root, errors)
    licenses = _check_license_contracts(root, errors)
    return {
        "contract": "plos_github_release_audit_v1",
        "status": "passed" if not errors else "failed",
        "repository": "terry0809000/beyond-macro-f1-pubmed-rct-public",
        "version": "1.0.0",
        "run_id": RUN_ID,
        "source_release": SOURCE_RELEASE,
        "source_notebook_sha256": SOURCE_NOTEBOOK_SHA256,
        "manifest_rows": len(manifest_rows),
        "verified_files": len(actual_paths & declared_paths),
        "total_manifested_bytes": sum(
            (root / relative).stat().st_size
            for relative in actual_paths & declared_paths
        ),
        "largest_file": largest,
        "completion_status": sentinel.get("status"),
        "licenses": licenses,
        "release_archive": {
            "filename": RELEASE_ARCHIVE_NAME,
            "bytes": RELEASE_ARCHIVE_BYTES,
            "sha256": RELEASE_ARCHIVE_SHA256,
            "members": 119,
            "archive_status": "verified license-inclusive sanitized evidence archive; canonical GitHub release asset",
            "public_deposit_manifest": {
                "filename": "public_deposit_manifest.csv",
                "bytes": PUBLIC_DEPOSIT_MANIFEST_BYTES,
                "sha256": PUBLIC_DEPOSIT_MANIFEST_SHA256,
            },
            "outer_checksums": {
                "filename": "SHA256SUMS",
                "bytes": OUTER_CHECKSUMS_BYTES,
                "sha256": OUTER_CHECKSUMS_SHA256,
            },
        },
        "zenodo_integration": {
            "status": "configured to archive the tagged GitHub repository",
            "doi_status": "pending v1.0.0 archival; no DOI claimed",
            "manual_github_release_asset_inclusion": "not assumed; inspect the generated Zenodo record",
            "note": "Insert only the DOI generated from the archived tagged repository after it resolves; verify Zenodo files independently.",
        },
        "checks": {
            "manifest_paths_bytes_sha256": not any(
                "manifest" in item or "SHA-256 mismatch" in item or "byte count" in item
                for item in errors
            ),
            "no_file_over_90_mib": largest["bytes"] <= MAX_GITHUB_FILE_BYTES,
            "no_high_confidence_secret": not any(
                "possible " in item for item in errors
            ),
            "no_personal_user_path": not any(
                "personal/local user path" in item for item in errors
            ),
            "no_raw_benchmark_text_columns": not any(
                "raw-text column" in item for item in errors
            ),
            "no_retired_manual_review_artifacts": not any(
                "retired manual-review" in item for item in errors
            ),
            "publication_completion_contract": sentinel.get("status")
            == "publication_complete",
            "approved_dual_license_boundary": not any(
                item.startswith(("LICENSE", "CITATION.cff", "README.md", "pyproject.toml"))
                or "license" in item.casefold()
                for item in errors
            ),
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON output path; prefer a path outside the frozen repository.",
    )
    args = parser.parse_args(argv)
    report = verify_repository(args.root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
