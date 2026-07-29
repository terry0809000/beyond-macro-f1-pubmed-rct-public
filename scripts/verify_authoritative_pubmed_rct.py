"""Verify the normalized PubMed RCT 20k files used by the revision.

The expected row counts, abstract counts, and SHA-256 values were established on
2026-07-11 after a row-for-row comparison with the PubMed_20k_RCT files in the
official Franck-Dernoncourt/pubmed-rct repository.  This script deliberately
does not download or rewrite the data; it verifies a caller-supplied directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "train": {
        "rows": 180_040,
        "abstracts": 15_000,
        "sha256": "2b1be3797844d060873d97e445aa4bf05fafbd050d526528364e233c17a7ced0",
    },
    "dev": {
        "rows": 30_212,
        "abstracts": 2_500,
        "sha256": "21019fb5fde4ed4fd67a358ae2a74aadf3b85540aa42ef2fb41d7a86d7aeda82",
    },
    "test": {
        "rows": 30_135,
        "abstracts": 2_500,
        "sha256": "b5dea844fecf520933f67f5dcd216c6cf1250f92946170cd93e55302fbb07d50",
    },
}

ALLOWED_LABELS = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(path: Path) -> tuple[dict[str, object], set[str]]:
    abstract_ids: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number}: expected three tab-separated fields")
            pubmed_id, label, text = parts
            if not pubmed_id or not text:
                raise ValueError(f"{path}:{line_number}: blank identifier or text")
            if label not in ALLOWED_LABELS:
                raise ValueError(f"{path}:{line_number}: unexpected label {label!r}")
            abstract_ids.add(pubmed_id)
            rows += 1
    result: dict[str, object] = {
        "file": str(path.resolve()),
        "rows": rows,
        "abstracts": len(abstract_ids),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    return result, abstract_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Directory containing train.txt, dev.txt, and test.txt")
    parser.add_argument("--csv", dest="csv_path", type=Path, help="Optional CSV report path")
    parser.add_argument("--json", dest="json_path", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    ids_by_split: dict[str, set[str]] = {}
    failures: list[str] = []
    for split, expected in EXPECTED.items():
        path = args.data_dir / f"{split}.txt"
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        observed, abstract_ids = inspect_split(path)
        observed["split"] = split
        observed["expected_rows"] = expected["rows"]
        observed["expected_abstracts"] = expected["abstracts"]
        observed["expected_sha256"] = expected["sha256"]
        observed["verified"] = all(
            observed[key] == expected[key] for key in ("rows", "abstracts", "sha256")
        )
        if not observed["verified"]:
            failures.append(f"{split}: count or SHA-256 mismatch")
        results.append(observed)
        ids_by_split[split] = abstract_ids

    overlaps: dict[str, int] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if left in ids_by_split and right in ids_by_split:
            count = len(ids_by_split[left] & ids_by_split[right])
            overlaps[f"{left}_{right}"] = count
            if count:
                failures.append(f"{left}/{right}: {count} overlapping abstract identifiers")

    payload = {
        "dataset": "PubMed RCT 20k",
        "official_source": (
            "https://github.com/Franck-Dernoncourt/pubmed-rct/tree/"
            "17ed2cb0590decfca0266add0c76f254f67232b4/PubMed_20k_RCT"
        ),
        "official_source_commit": "17ed2cb0590decfca0266add0c76f254f67232b4",
        "verification_date": "2026-07-11",
        "splits": results,
        "abstract_id_overlaps": overlaps,
        "verified": not failures,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.csv_path:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()) if results else ["split"])
            writer.writeheader()
            writer.writerows(results)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
