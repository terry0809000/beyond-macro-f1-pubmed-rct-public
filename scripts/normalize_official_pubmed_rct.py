"""Convert the upstream PubMed RCT block format to the revision's TSV format.

Upstream files use ``###<PMID>`` header lines followed by ``LABEL<TAB>text``
rows and blank separators.  The revision uses one row per sentence with
``PMID<TAB>LABEL<TAB>text``.  Conversion is deterministic and preserves label
and text bytes apart from adding the identifier and final LF line ending.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ALLOWED_LABELS = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}


def convert(source: Path, destination: Path) -> tuple[int, int]:
    current_id: str | None = None
    abstract_ids: set[str] = set()
    rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline=None) as reader, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line_number, raw_line in enumerate(reader, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("###"):
                current_id = line[3:]
                if not current_id:
                    raise ValueError(f"{source}:{line_number}: blank abstract identifier")
                abstract_ids.add(current_id)
                continue
            if current_id is None:
                raise ValueError(f"{source}:{line_number}: sentence precedes the first abstract header")
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"{source}:{line_number}: expected LABEL<TAB>text")
            label, text = parts
            if label not in ALLOWED_LABELS:
                raise ValueError(f"{source}:{line_number}: unexpected label {label!r}")
            if not text:
                raise ValueError(f"{source}:{line_number}: blank sentence text")
            writer.write(f"{current_id}\t{label}\t{text}\n")
            rows += 1
    return rows, len(abstract_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Upstream block-format split file")
    parser.add_argument("destination", type=Path, help="Normalized three-column output file")
    args = parser.parse_args()
    rows, abstracts = convert(args.source, args.destination)
    print(f"wrote={args.destination.resolve()} rows={rows} abstracts={abstracts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
