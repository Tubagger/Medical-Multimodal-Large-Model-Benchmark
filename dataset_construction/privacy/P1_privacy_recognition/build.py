#!/usr/bin/env python3
"""Materialize the frozen, accepted P.1 privacy-recognition cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED = 260
TYPE_LABEL = {"clean": 0, "phi": 1, "noise": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(labels_path: Path, image_dir: Path, balanced_labels_path: Path, output_root: Path) -> Path:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    balanced = json.loads(balanced_labels_path.read_text(encoding="utf-8"))
    if len(labels) != EXPECTED or len(balanced) != EXPECTED:
        raise ValueError("P.1 requires 260 released and 260 balanced-source rows")
    if len({str(row["id"]) for row in labels}) != EXPECTED:
        raise ValueError("P.1 contains duplicate IDs")
    if len({str(row["id"]) for row in balanced}) != EXPECTED:
        raise ValueError("P.1 balanced source contains duplicate IDs")
    balanced_by_id = {str(row["id"]): row for row in balanced}

    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".P1-GMAI-", dir=output_root) as temp_name:
        temp = Path(temp_name)
        images_out = temp / "images"
        images_out.mkdir()
        provenance = []
        checksums = []
        for rank, row in enumerate(labels):
            condition = str(row["type"])
            if row["label"] != TYPE_LABEL.get(condition):
                raise ValueError(f"ID {row['id']}: type/label mismatch")
            source = balanced_by_id.get(str(row["source_id"]))
            if source is None or source.get("image") != row.get("original_image"):
                raise ValueError(f"ID {row['id']}: source provenance mismatch")
            source_classes = [name for name, value in source.get("labels", {}).items() if value == 1]
            if len(source_classes) != 1:
                raise ValueError(f"ID {row['id']}: expected one CheXpert source class")
            image_name = Path(row["image"]).name
            if image_name != row["image"] or not image_name.startswith(condition + "_"):
                raise ValueError(f"ID {row['id']}: unsafe or inconsistent image name")
            image_path = image_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                fmt, (width, height) = image.format, image.size
                image.verify()
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                yellow_pixels = sum(r >= 180 and g >= 180 and b <= 100 for r, g, b in rgb.getdata())
            digest = sha256(image_path)
            shutil.copyfile(image_path, images_out / image_name)
            checksums.append(f"{digest}  images/{image_name}")
            provenance.append({
                "rank": rank,
                "id": row["id"],
                "source_id": row["source_id"],
                "source_image": row["original_image"],
                "source_class": source_classes[0],
                "condition": condition,
                "label": row["label"],
                "image": image_name,
                "sha256": digest,
                "format": fmt,
                "width": width,
                "height": height,
                "synthetic_yellow_pixel_count": yellow_pixels,
            })

        manifest = {
            "schema_version": 1,
            "task_id": "P.1",
            "dataset": "CheXpert synthetic privacy-recognition cohort",
            "sample_count": EXPECTED,
            "class_counts": {str(k): v for k, v in sorted(Counter(row["label"] for row in labels).items())},
            "condition_counts": dict(sorted(Counter(row["type"] for row in labels).items())),
            "source_disease_count": 13,
            "source_samples_per_disease": 20,
            "selection_seed": 42,
            "overlay_seed": 42,
            "historical_runtime_compatible": True,
            "requires_model_rerun": False,
            "accepted_protocol_observations": [
                "clean denotes no benchmark-added overlay, while source radiographs may retain acquisition markers",
                "synthetic PHI and noise overlay lengths differ",
                "synthetic identifiers are benchmark fields rather than real patient records",
            ],
        }
        (temp / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (temp / "provenance.jsonl").open("w", encoding="utf-8") as handle:
            for record in provenance:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        (temp / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--balanced-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.labels.resolve(), args.image_dir.resolve(), args.balanced_labels.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
