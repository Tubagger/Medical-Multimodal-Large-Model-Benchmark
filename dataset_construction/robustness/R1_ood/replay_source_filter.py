#!/usr/bin/env python3
"""Verify that the frozen OOD TSV is the exact declared filter of GMAI VAL.

The comparison is streaming and stores only compact per-row digests.  Image
base64 is included in each digest, so a passed result covers both metadata and
image payload identity without materializing either multi-gigabyte TSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ID_MODALITIES = {
    "CT",
    "MRI",
    "X-ray",
    "UltraSound",
    "Fundus Photography",
    "weighted MRI",
    "T1 weighted MRI",
    "T2 weighted MRI",
    "FLAIR MRI",
    "DCE MRI",
}


def row_digest(row: dict[str, str], fieldnames: list[str]) -> str:
    digest = hashlib.sha256()
    for field in fieldnames:
        digest.update(field.encode("utf-8"))
        digest.update(b"\0")
        digest.update((row.get(field) or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def compact_rows(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        fieldnames = list(reader.fieldnames)
        rows = [(row.get("index", ""), row_digest(row, fieldnames)) for row in reader]
    return fieldnames, rows


def replay(raw_tsv: Path, ood_tsv: Path) -> dict:
    csv.field_size_limit(sys.maxsize)
    ood_fields, expected = compact_rows(ood_tsv)
    actual: list[tuple[str, str]] = []
    raw_rows = 0
    missing_modality = 0
    with raw_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in {raw_tsv}")
        if list(reader.fieldnames) != ood_fields:
            raise ValueError("Raw and OOD TSV schemas differ")
        for row in reader:
            raw_rows += 1
            modality = row.get("modality", "")
            if not modality:
                missing_modality += 1
                continue
            if modality not in ID_MODALITIES:
                actual.append((row.get("index", ""), row_digest(row, ood_fields)))

    mismatch_positions = [index for index, (left, right) in enumerate(zip(expected, actual)) if left != right]
    report = {
        "status": "passed" if expected == actual else "failed",
        "raw_rows": raw_rows,
        "missing_modality_rows_removed": missing_modality,
        "ood_rows_expected": len(expected),
        "ood_rows_replayed": len(actual),
        "id_modalities_removed": sorted(ID_MODALITIES),
        "first_mismatch_positions": mismatch_positions[:20],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-tsv", required=True, type=Path)
    parser.add_argument("--ood-tsv", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = replay(args.raw_tsv.resolve(), args.ood_tsv.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
