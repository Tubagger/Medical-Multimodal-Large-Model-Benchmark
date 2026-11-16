#!/usr/bin/env python3
"""Validate a subset produced by the verified C.1 builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from build import DATASETS, LETTERS, SAMPLES_PER_CLASS, display_label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    config = DATASETS[dataset]
    labels = json.loads((dataset_dir / "labels.json").read_text(encoding="utf-8"))
    provenance = [
        json.loads(line)
        for line in (dataset_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = len(config["classes"]) * SAMPLES_PER_CLASS
    errors: list[str] = []

    fail(manifest.get("seed") == 42, "wrong random seed", errors)
    fail(
        manifest.get("random_method") == "single random.Random(42) instance per dataset build",
        "wrong random method",
        errors,
    )
    fail(len(labels) == expected_count, f"labels count is {len(labels)}, expected {expected_count}", errors)
    fail(len(provenance) == expected_count, "provenance count mismatch", errors)
    fail(len({row["id"] for row in labels}) == expected_count, "duplicate output IDs", errors)
    fail(len({row["image"] for row in labels}) == expected_count, "duplicate output image names", errors)
    fail(len({str(row["source_id"]) for row in provenance}) == expected_count, "duplicate source IDs", errors)

    by_id = {row["id"]: row for row in provenance}
    class_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    image_hashes: list[str] = []
    patient_ids: list[str] = []

    for row in labels:
        prov = by_id.get(row["id"])
        if prov is None:
            errors.append(f"missing provenance for {row['id']}")
            continue
        source_labels = prov["source_labels"]
        source_path_pattern = config["source_path_pattern"]
        if source_path_pattern:
            fail(
                re.search(source_path_pattern, str(prov.get("original_path", ""))) is not None,
                f"source split mismatch: {row['id']}",
                errors,
            )
        positives = [name for name, value in source_labels.items() if value == 1]
        fail(len(positives) == 1, f"non-single-positive source row: {row['id']}", errors)
        fail(all(value in (0, 1) for value in source_labels.values()), f"uncertain source label: {row['id']}", errors)
        fail(set(source_labels) == set(config["classes"]), f"class schema mismatch: {row['id']}", errors)
        selected = prov["selected_label"]
        class_counts[selected] += 1

        option_lines = row["options"].splitlines()
        fail(len(option_lines) == 5, f"not five options: {row['id']}", errors)
        option_map = {}
        for line in option_lines:
            match = re.fullmatch(r"([A-E])\. (.+)", line)
            if match:
                option_map[match.group(1)] = match.group(2)
        fail(len(option_map) == 5, f"malformed options: {row['id']}", errors)
        fail(row["answer"] in LETTERS, f"invalid answer: {row['id']}", errors)
        if row["answer"] in option_map:
            fail(
                option_map[row["answer"]] == display_label(dataset, selected),
                f"answer/label mismatch: {row['id']}",
                errors,
            )
        answer_counts[row["answer"]] += 1

        image_path = dataset_dir / "images" / row["image"]
        fail(image_path.is_file(), f"missing image: {row['image']}", errors)
        if image_path.is_file():
            digest = sha256_file(image_path)
            image_hashes.append(digest)
            fail(digest == prov["sha256"], f"checksum mismatch: {row['image']}", errors)
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(f"invalid image {row['image']}: {exc}")
        if prov.get("patient_id") is not None:
            patient_ids.append(prov["patient_id"])

    fail(all(class_counts[name] == SAMPLES_PER_CLASS for name in config["classes"]), "class balance mismatch", errors)
    fail(len(set(image_hashes)) == len(image_hashes), "duplicate image contents", errors)
    expected_answers = expected_count // 5
    fail(all(answer_counts[letter] == expected_answers for letter in LETTERS), "answer-position imbalance", errors)
    if config["patient_pattern"]:
        fail(len(patient_ids) == expected_count, "missing patient identifiers", errors)
        fail(len(set(patient_ids)) == expected_count, "duplicate patients", errors)

    checksum_lines = set((dataset_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    expected_lines = {
        f"{prov['sha256']}  images/{row['image']}"
        for row, prov in ((row, by_id[row["id"]]) for row in labels if row["id"] in by_id)
    }
    fail(checksum_lines == expected_lines, "checksums.sha256 does not match provenance", errors)

    report = {
        "schema_version": 1,
        "task_id": "C.1",
        "dataset": dataset,
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels),
        "class_counts": dict(class_counts),
        "answer_counts": dict(answer_counts),
        "unique_patients": len(set(patient_ids)) if patient_ids else None,
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
