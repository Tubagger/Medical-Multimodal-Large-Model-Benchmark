#!/usr/bin/env python3
"""Validate a dataset produced by the verified C.3 builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from build import EXPECTED_TASKS, LETTERS, SAMPLES_PER_TASK, normalized_text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def parse_options(value: object) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    output: dict[str, str] = {}
    for line in value.splitlines():
        match = re.fullmatch(r"([A-E])\. (.+)", line)
        if match is None or match.group(1) in output:
            return None
        output[match.group(1)] = match.group(2).strip()
    return output


def validate(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((dataset_dir / "labels.json").read_text(encoding="utf-8"))
    provenance = [
        json.loads(line)
        for line in (dataset_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = EXPECTED_TASKS * SAMPLES_PER_TASK
    errors: list[str] = []
    check(manifest.get("task_id") == "C.3", "wrong task ID", errors)
    check(manifest.get("upstream_sampling_seed") == 42, "wrong upstream seed", errors)
    check(manifest.get("synthetic_options_added") == 232, "wrong synthetic option count", errors)
    check(manifest.get("rows_with_synthetic_options") == 197, "wrong completed-row count", errors)
    check(len(labels) == expected_count, f"labels count is {len(labels)}, expected {expected_count}", errors)
    check(len(provenance) == expected_count, "provenance count mismatch", errors)
    check(len({str(row.get("id")) for row in labels}) == expected_count, "duplicate IDs", errors)
    check(len({str(row.get("source_index")) for row in provenance}) == expected_count, "duplicate source indices", errors)

    by_id = {str(row.get("id")): row for row in provenance}
    task_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    option_count_distribution: Counter[int] = Counter()
    answer_counts: Counter[str] = Counter()
    image_hashes: list[str] = []
    expected_checksum_lines: set[str] = set()

    required_text_fields = [
        "question",
        "category",
        "clinical_vqa_task",
        "perceptual_granularity",
        "modality",
        "original_task",
    ]
    for row in labels:
        output_id = str(row.get("id"))
        prov = by_id.get(output_id)
        if prov is None:
            errors.append(f"missing provenance for ID {output_id}")
            continue
        for field in required_text_fields:
            check(isinstance(row.get(field), str) and bool(row[field].strip()), f"missing {field} for ID {output_id}", errors)
        task_counts[row.get("clinical_vqa_task")] += 1
        modality_counts[row.get("modality")] += 1

        options = parse_options(row.get("options"))
        check(options is not None, f"malformed options for ID {output_id}", errors)
        if options is None:
            continue
        check(len(options) == 5, f"ID {output_id} does not have five options", errors)
        check(list(options) == list(LETTERS[: len(options)]), f"non-contiguous option letters for ID {output_id}", errors)
        check(row.get("option_count") == len(options), f"option count field mismatch for ID {output_id}", errors)
        normalized_options = [normalized_text(value) for value in options.values()]
        check(len(set(normalized_options)) == len(normalized_options), f"duplicate options for ID {output_id}", errors)
        answer = row.get("answer")
        check(answer in options, f"answer is absent from options for ID {output_id}", errors)
        if answer in options:
            check(
                normalized_text(options[answer]) == normalized_text(row.get("category", "")),
                f"answer/category mismatch for ID {output_id}",
                errors,
            )
        original_letters = prov.get("original_option_letters")
        synthetic_letters = prov.get("synthetic_option_letters")
        check(isinstance(original_letters, list), f"missing original option provenance for ID {output_id}", errors)
        check(isinstance(synthetic_letters, list), f"missing synthetic option provenance for ID {output_id}", errors)
        if isinstance(original_letters, list) and isinstance(synthetic_letters, list):
            check(
                original_letters + synthetic_letters == list(options),
                f"option provenance mismatch for ID {output_id}",
                errors,
            )
        option_count_distribution[len(options)] += 1
        answer_counts[answer] += 1

        image_path = dataset_dir / "images" / str(row.get("image", ""))
        check(image_path.is_file(), f"missing image for ID {output_id}", errors)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                size = image.size
                format_name = image.format
                image.verify()
            check(size == (prov.get("width"), prov.get("height")), f"image dimension mismatch for ID {output_id}", errors)
            check(format_name == prov.get("image_format"), f"image format mismatch for ID {output_id}", errors)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"invalid image for ID {output_id}: {exc}")
        digest = sha256_file(image_path)
        image_hashes.append(digest)
        check(digest == prov.get("sha256"), f"checksum mismatch for ID {output_id}", errors)
        expected_checksum_lines.add(f"{digest}  images/{row['image']}")

    check(len(task_counts) == EXPECTED_TASKS, "task count mismatch", errors)
    check(all(count == SAMPLES_PER_TASK for count in task_counts.values()), "task balance mismatch", errors)
    check(len(set(image_hashes)) == expected_count, "duplicate image content", errors)
    image_files = [path for path in (dataset_dir / "images").iterdir() if path.is_file()]
    check(len(image_files) == expected_count, "unexpected image file count", errors)
    actual_checksum_lines = set((dataset_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    check(actual_checksum_lines == expected_checksum_lines, "checksums.sha256 mismatch", errors)

    report = {
        "schema_version": 1,
        "task_id": "C.3",
        "dataset": "GMAI",
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels),
        "task_counts": dict(sorted(task_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "option_count_distribution": {
            str(count): frequency for count, frequency in sorted(option_count_distribution.items())
        },
        "answer_counts": dict(answer_counts),
        "unique_source_images": len(set(image_hashes)),
        "synthetic_options": sum(
            len(row.get("synthetic_option_letters", [])) for row in provenance
        ),
        "rows_with_synthetic_options": sum(
            bool(row.get("synthetic_option_letters", [])) for row in provenance
        ),
        "missing_optional_metadata_counts": {
            "department": sum(not row.get("department", "").strip() for row in labels)
        },
        "errors": errors,
    }
    (dataset_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
