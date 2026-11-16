#!/usr/bin/env python3
"""Validate a dataset produced by the verified C.2 builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from build import CLASSES, SAMPLES_PER_CLASS, valid_bbox


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(dataset_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((dataset_dir / "labels.json").read_text(encoding="utf-8"))
    provenance = [
        json.loads(line)
        for line in (dataset_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = len(CLASSES) * SAMPLES_PER_CLASS
    errors: list[str] = []
    check(manifest.get("task_id") == "C.2", "wrong task ID", errors)
    check(manifest.get("seed") == 42, "wrong random seed", errors)
    check(len(labels) == expected_count, f"labels count is {len(labels)}, expected {expected_count}", errors)
    check(len(provenance) == expected_count, "provenance count mismatch", errors)
    check([row.get("id") for row in labels] == list(range(1, expected_count + 1)), "IDs are not sequential", errors)
    check(len({row.get("source_id") for row in provenance}) == expected_count, "duplicate source IDs", errors)
    check(len({row.get("original_path") for row in provenance}) == expected_count, "duplicate original paths", errors)

    by_id = {row.get("id"): row for row in provenance}
    class_counts: Counter[str] = Counter()
    image_hashes: list[str] = []
    expected_checksum_lines: set[str] = set()
    for row in labels:
        output_id = row.get("id")
        prov = by_id.get(output_id)
        if prov is None:
            errors.append(f"missing provenance for ID {output_id}")
            continue
        lesion = row.get("lesion")
        class_counts[lesion] += 1
        check(lesion in CLASSES, f"invalid class for ID {output_id}", errors)
        check(valid_bbox(row.get("bbox")), f"invalid normalized bbox for ID {output_id}", errors)
        check(row.get("bbox") == prov.get("source_bbox"), f"bbox/source mismatch for ID {output_id}", errors)
        check(lesion == prov.get("source_lesion"), f"class/source mismatch for ID {output_id}", errors)
        check(row.get("original_path") == prov.get("original_path"), f"path/source mismatch for ID {output_id}", errors)
        check(
            (row.get("width"), row.get("height"))
            == (prov.get("source_width"), prov.get("source_height")),
            f"dimension/source mismatch for ID {output_id}",
            errors,
        )

        image_path = dataset_dir / "images" / str(row.get("image", ""))
        check(image_path.is_file(), f"missing image for ID {output_id}", errors)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                size = image.size
                image.verify()
            check(size == (row.get("width"), row.get("height")), f"actual dimension mismatch for ID {output_id}", errors)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"invalid image for ID {output_id}: {exc}")
        digest = sha256_file(image_path)
        image_hashes.append(digest)
        check(digest == prov.get("sha256"), f"checksum mismatch for ID {output_id}", errors)
        expected_checksum_lines.add(f"{digest}  images/{row['image']}")

    check(all(class_counts[name] == SAMPLES_PER_CLASS for name in CLASSES), "class balance mismatch", errors)
    check(len(set(image_hashes)) == len(image_hashes), "duplicate image contents", errors)
    actual_checksum_lines = set((dataset_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    check(actual_checksum_lines == expected_checksum_lines, "checksums.sha256 mismatch", errors)
    image_files = [path for path in (dataset_dir / "images").iterdir() if path.is_file()]
    check(len(image_files) == expected_count, "unexpected image file count", errors)

    report = {
        "schema_version": 1,
        "task_id": "C.2",
        "dataset": "DeepLesion",
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels),
        "class_counts": dict(class_counts),
        "unique_source_images": len(set(image_hashes)),
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
