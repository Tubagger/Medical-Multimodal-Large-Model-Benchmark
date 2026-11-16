#!/usr/bin/env python3
"""Build the verified C.2 lesion-localization subset from DeepLesion."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SEED = 42
SAMPLES_PER_CLASS = 30
CLASSES = [
    "lung",
    "mediastinum",
    "liver",
    "soft_tissue",
    "pelvis",
    "kidney",
    "bone",
    "abdomen",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return False
    x, y, width, height = value
    return width > 0 and height > 0 and 0 <= x < x + width <= 1 and 0 <= y < y + height <= 1


def inspect_image(path: Path) -> tuple[tuple[int, int] | None, str | None, str | None]:
    if not path.is_file():
        return None, None, "missing"
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return size, sha256_file(path), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, None, f"invalid: {exc}"


def build(source_dir: Path, output_root: Path) -> Path:
    source_labels = source_dir / "labels.json"
    source_images = source_dir / "images"
    if not source_labels.is_file() or not source_images.is_dir():
        raise FileNotFoundError(f"Expected labels.json and images/ under {source_dir}")

    destination = output_root / "DeepLesion"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    rows = json.loads(source_labels.read_text(encoding="utf-8"))
    path_counts = Counter(str(row.get("original_path")) for row in rows)
    image_info: dict[object, tuple[tuple[int, int] | None, str | None, str | None]] = {}
    hash_to_ids: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        info = inspect_image(source_images / str(row.get("image", "")))
        image_info[row.get("id")] = info
        if info[1] is not None:
            hash_to_ids[info[1]].append(row.get("id"))
    duplicate_content_ids = {
        source_id
        for source_ids in hash_to_ids.values()
        if len(source_ids) > 1
        for source_id in source_ids
    }

    groups: dict[str, list[dict]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        lesion = row.get("lesion")
        if lesion not in CLASSES:
            rejection_counts["non_target_class"] += 1
            continue
        size, _, image_error = image_info[row.get("id")]
        reasons: list[str] = []
        if image_error is not None:
            reasons.append("missing_or_invalid_image")
        if size != (row.get("width"), row.get("height")):
            reasons.append("dimension_mismatch")
        if not valid_bbox(row.get("bbox")):
            reasons.append("invalid_bbox")
        if path_counts[str(row.get("original_path"))] > 1 or row.get("id") in duplicate_content_ids:
            reasons.append("ambiguous_duplicate_image")
        if reasons:
            rejection_counts.update(reasons)
            continue
        groups[lesion].append(row)

    insufficient = {
        lesion: len(groups[lesion])
        for lesion in CLASSES
        if len(groups[lesion]) < SAMPLES_PER_CLASS
    }
    if insufficient:
        raise ValueError(f"Insufficient valid candidates: {insufficient}")

    rng = random.Random(SEED)
    selected: list[dict] = []
    for lesion in CLASSES:
        selected.extend(rng.sample(groups[lesion], SAMPLES_PER_CLASS))
    rng.shuffle(selected)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".DeepLesion-", dir=output_root) as temp_name:
        temp_dir = Path(temp_name)
        output_images = temp_dir / "images"
        output_images.mkdir()
        labels_out: list[dict] = []
        provenance_out: list[dict] = []
        checksum_lines: list[str] = []

        for output_id, row in enumerate(selected, start=1):
            source_image = source_images / row["image"]
            suffix = source_image.suffix.lower() or ".img"
            output_name = f"image_{output_id:06d}{suffix}"
            output_image = output_images / output_name
            shutil.copy2(source_image, output_image)
            digest = sha256_file(output_image)
            size, source_digest, _ = image_info[row["id"]]
            if digest != source_digest:
                raise RuntimeError(f"Copied image hash mismatch: {row['image']}")

            labels_out.append(
                {
                    "id": output_id,
                    "image": output_name,
                    "original_path": row["original_path"],
                    "width": size[0],
                    "height": size[1],
                    "lesion": row["lesion"],
                    "bbox": row["bbox"],
                }
            )
            provenance_out.append(
                {
                    "id": output_id,
                    "source_id": row["id"],
                    "source_image": row["image"],
                    "original_path": row["original_path"],
                    "source_width": row["width"],
                    "source_height": row["height"],
                    "source_lesion": row["lesion"],
                    "source_bbox": row["bbox"],
                    "sha256": digest,
                }
            )
            checksum_lines.append(f"{digest}  images/{output_name}")

        (temp_dir / "labels.json").write_text(
            json.dumps(labels_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (temp_dir / "provenance.jsonl").open("w", encoding="utf-8") as handle:
            for row in provenance_out:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temp_dir / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "task_id": "C.2",
            "dataset": "DeepLesion",
            "seed": SEED,
            "sampling_method": "single random.Random(42) instance",
            "samples_per_class": SAMPLES_PER_CLASS,
            "classes": CLASSES,
            "sample_count": len(labels_out),
            "source_row_count": len(rows),
            "eligible_candidate_counts": {lesion: len(groups[lesion]) for lesion in CLASSES},
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "source_labels_sha256": sha256_file(source_labels),
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_dir.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.source_dir.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
