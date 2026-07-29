"""Smoke tests for the frozen lightweight release."""

from __future__ import annotations

import csv
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
