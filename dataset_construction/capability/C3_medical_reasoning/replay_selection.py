#!/usr/bin/env python3
"""Replay the upstream seed-42 task-balanced selection without retaining images.

The raw TSV is large because every image is base64-encoded.  This audit reads
it sequentially and keeps only non-image columns in memory, then applies the
same pandas sampling operations as the legacy selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd


SEED = 42
SAMPLES_PER_TASK = 15


def read_raw_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "image" not in reader.fieldnames:
            raise ValueError("Expected a TSV header containing an image column")
        fields = [field for field in reader.fieldnames if field != "image"]
        rows = [{field: row[field] for field in fields} for row in reader]
    return fields, rows


def read_balanced_metadata(path: Path, fields: list[str]) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{field: row[field] for field in fields} for row in reader]


def replay(raw_tsv: Path, balanced_tsv: Path) -> dict:
    fields, raw_rows = read_raw_metadata(raw_tsv)
    frame = pd.DataFrame(raw_rows)
    selected_groups = []
    for _, group in sorted(
        frame.groupby("clinical VQA task"), key=lambda item: len(item[1]), reverse=True
    ):
        if len(group) >= SAMPLES_PER_TASK:
            selected = group.sample(n=SAMPLES_PER_TASK, random_state=SEED)
        else:
            selected = group
        selected_groups.append(selected)
    expected = (
        pd.concat(selected_groups)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
        .to_dict("records")
    )
    observed = read_balanced_metadata(balanced_tsv, fields)
    mismatches: list[dict] = []
    if len(expected) != len(observed):
        mismatches.append({"type": "row_count", "expected": len(expected), "observed": len(observed)})
    for row_index, (left, right) in enumerate(zip(expected, observed)):
        for field in fields:
            left_value = "" if pd.isna(left[field]) else str(left[field])
            if left_value != right[field]:
                mismatches.append(
                    {
                        "type": "field",
                        "row": row_index,
                        "field": field,
                        "expected": left_value,
                        "observed": right[field],
                    }
                )
                break
    return {
        "schema_version": 1,
        "task_id": "C.3",
        "seed": SEED,
        "raw_row_count": len(raw_rows),
        "balanced_row_count": len(observed),
        "task_counts": {
            str(task): int(count)
            for task, count in frame["clinical VQA task"].value_counts().sort_index().items()
        },
        "metadata_mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
        "status": "passed" if not mismatches else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-tsv", required=True, type=Path)
    parser.add_argument("--balanced-tsv", required=True, type=Path)
    args = parser.parse_args()
    report = replay(args.raw_tsv.resolve(), args.balanced_tsv.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
