"""Smoke tests for the frozen lightweight release."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from plos_nlp_release import verify_repository


ROOT = Path(__file__).resolve().parents[1]


def test_release_contract() -> None:
    report = verify_repository(ROOT)
    assert report["status"] == "passed", report["errors"]
    assert report["completion_status"] == "publication_complete"
    assert report["largest_file"]["bytes"] < 90 * 1024 * 1024
    assert report["licenses"]["code_license"] == "MIT"
    assert report["licenses"]["content_data_license"] == "CC-BY-4.0"


def test_primary_results_cover_all_locked_system_rows() -> None:
    path = ROOT / "artifacts" / "summary" / "final_test_results_by_seed.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert all(float(row["accuracy"]) > 0 for row in rows)
    assert all(float(row["macro_f1"]) > 0 for row in rows)


def test_efficiency_rows_have_positive_model_sizes() -> None:
    path = ROOT / "artifacts" / "summary" / "implementation_specific_efficiency.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    training_rows = [row for row in rows if row["stage"] == "training"]
    assert training_rows
    assert all(float(row["n_parameters_total"]) > 0 for row in training_rows)
    assert all(float(row["estimated_model_size_mb"]) > 0 for row in training_rows)


def test_verifier_ignores_standard_generated_install_metadata(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "release"
    shutil.copytree(
        ROOT,
        copied,
        ignore=shutil.ignore_patterns(
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    generated_files = {
        "src/plos_nlp_release.egg-info/PKG-INFO": "generated metadata\n",
        "build/lib/plos_nlp_release/__init__.py": "generated build copy\n",
        "dist/plos_nlp_release-1.0.0-py3-none-any.whl": "generated wheel\n",
        ".pytest_cache/v/cache/nodeids": "[]\n",
        ".venv/pyvenv.cfg": "generated virtual environment\n",
    }
    for relative, content in generated_files.items():
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    report = verify_repository(copied)
    assert report["status"] == "passed", report["errors"]
    assert report["verified_files"] == report["manifest_rows"]

    concealed = copied / "artifacts" / "build" / "unmanifested.txt"
    concealed.parent.mkdir(parents=True)
    concealed.write_text("must remain visible to the verifier\n", encoding="utf-8")
    report = verify_repository(copied)
    assert report["status"] == "failed"
    assert (
        "unmanifested file is present: artifacts/build/unmanifested.txt"
        in report["errors"]
    )
