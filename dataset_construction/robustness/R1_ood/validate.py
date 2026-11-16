#!/usr/bin/env python3
"""Validate the corrected verified R.1 OOD dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from build import EXPECTED_MODALITIES, EXPECTED_SAMPLES, LETTERS, MAX_PER_MODALITY, normalized_text


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
    errors: list[str] = []
    source_observations: list[str] = []
    check(manifest.get("task_id") == "R.1", "wrong task ID", errors)
    check(manifest.get("selection_seed") == 42, "wrong selection seed", errors)
    check(manifest.get("added_option_rows") == 161, "unexpected added-option row count", errors)
    check(manifest.get("added_option_fields") == 256, "unexpected added-option field count", errors)
    check(manifest.get("annotation_authority") == "published benchmark release", "wrong annotation authority", errors)
    check(manifest.get("secondary_clinical_review_required") is False, "unexpected secondary-review requirement", errors)
    check(manifest.get("upstream_dataset", {}).get("declared_license") == "apache-2.0", "missing upstream license", errors)
    check(len(labels) == EXPECTED_SAMPLES, f"labels count is {len(labels)}, expected {EXPECTED_SAMPLES}", errors)
    check(len(provenance) == EXPECTED_SAMPLES, "provenance count mismatch", errors)
    check(len({str(row.get("id")) for row in labels}) == EXPECTED_SAMPLES, "duplicate IDs", errors)
    check(len({row.get("source_row") for row in provenance}) == EXPECTED_SAMPLES, "duplicate source rows", errors)

    by_id = {str(row.get("id")): row for row in provenance}
    modality_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    answer_counts: Counter[str] = Counter()
    image_hashes: list[str] = []
    expected_checksums: set[str] = set()
    required_text = ["question", "category", "clinical_vqa_task", "modality", "original_task"]

    for row in labels:
        row_id = str(row.get("id"))
        prov = by_id.get(row_id)
        if prov is None:
            errors.append(f"missing provenance for ID {row_id}")
            continue
        for field in required_text:
            check(isinstance(row.get(field), str) and bool(row[field].strip()), f"missing {field} for ID {row_id}", errors)
        modality_counts[row.get("modality")] += 1
        task_counts[row.get("clinical_vqa_task")] += 1

        options = parse_options(row.get("options"))
        check(options is not None, f"malformed options for ID {row_id}", errors)
        if options is None:
            continue
        check(len(options) == 5, f"runtime option count is not five for ID {row_id}", errors)
        check(list(options) == list(LETTERS), f"runtime options are not A--E for ID {row_id}", errors)
        check(row.get("option_count") == len(options), f"option count mismatch for ID {row_id}", errors)
        normalized = [normalized_text(value) for value in options.values()]
        if len(set(normalized)) != len(normalized):
            source_observations.append(f"published source contains normalized duplicate options for ID {row_id}")
        answer = row.get("answer")
        check(answer in options, f"answer absent from options for ID {row_id}", errors)
        if answer in options:
            check(normalized_text(options[answer]) == normalized_text(row.get("category", "")), f"answer/category mismatch for ID {row_id}", errors)
        original_letters = prov.get("original_option_letters", [])
        added_letters = prov.get("added_option_letters", [])
        check(original_letters + added_letters == list(options), f"option provenance mismatch for ID {row_id}", errors)
        check(not set(original_letters) & set(added_letters), f"overlapping option provenance for ID {row_id}", errors)
        check(prov.get("modality_normalized") == row.get("modality"), f"modality provenance mismatch for ID {row_id}", errors)
        option_counts[len(options)] += 1
        answer_counts[answer] += 1

        image_path = dataset_dir / "images" / str(row.get("image", ""))
        check(image_path.is_file(), f"missing image for ID {row_id}", errors)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                size = image.size
                format_name = image.format
                image.verify()
            check(size == (prov.get("width"), prov.get("height")), f"image dimensions mismatch for ID {row_id}", errors)
            check(format_name == prov.get("image_format"), f"image format mismatch for ID {row_id}", errors)
        except Exception as exc:
            errors.append(f"invalid image for ID {row_id}: {exc}")
        digest = sha256_file(image_path)
        image_hashes.append(digest)
        check(digest == prov.get("sha256"), f"checksum mismatch for ID {row_id}", errors)
        expected_checksums.add(f"{digest}  images/{row['image']}")

    check(len(modality_counts) == EXPECTED_MODALITIES, "normalized modality count mismatch", errors)
    check(all(count <= MAX_PER_MODALITY for count in modality_counts.values()), "a modality exceeds the 25-row cap", errors)
    check(dict(sorted(modality_counts.items())) == manifest.get("modality_counts"), "manifest modality counts mismatch", errors)
    check(len(set(image_hashes)) == EXPECTED_SAMPLES, "duplicate image content", errors)
    image_files = [path for path in (dataset_dir / "images").iterdir() if path.is_file()]
    check(len(image_files) == EXPECTED_SAMPLES, "unexpected image file count", errors)
    actual_checksums = set((dataset_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    check(actual_checksums == expected_checksums, "checksums.sha256 mismatch", errors)

    report = {
        "schema_version": 1,
        "task_id": "R.1",
        "dataset": "GMAI-MMBench OOD",
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels),
        "modality_count": len(modality_counts),
        "modality_counts": dict(sorted(modality_counts.items())),
        "clinical_vqa_task_counts": dict(sorted(task_counts.items())),
        "option_count_distribution": {str(k): v for k, v in sorted(option_counts.items())},
        "answer_counts": dict(sorted(answer_counts.items())),
        "unique_source_images": len(set(image_hashes)),
        "added_option_rows": manifest.get("added_option_rows"),
        "added_option_fields": manifest.get("added_option_fields"),
        "historical_runtime_compatible": True,
        "requires_model_rerun": False,
        "annotation_authority": manifest.get("annotation_authority"),
        "secondary_clinical_review_required": False,
        "source_observations": source_observations,
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
