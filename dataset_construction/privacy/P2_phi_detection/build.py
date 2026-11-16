#!/usr/bin/env python3
"""Materialize the frozen 260-item P.2 PHI-detection cohort."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_type(value: str) -> str:
    if value.startswith("MRN-"):
        return "MRN"
    return value.split(":", 1)[0]


def build(labels_path: Path, image_dir: Path, balanced_labels_path: Path, output_root: Path) -> Path:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    balanced = json.loads(balanced_labels_path.read_text(encoding="utf-8"))
    if len(labels) != EXPECTED or len(balanced) != EXPECTED:
        raise ValueError("P.2 requires 260 labels and 260 balanced source rows")
    if len({str(row["id"]) for row in labels}) != EXPECTED:
        raise ValueError("P.2 labels contain duplicate IDs")
    balanced_by_id = {str(row["id"]): row for row in balanced}

    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".P2-GMAI-", dir=output_root) as temp_name:
        temp = Path(temp_name)
        images_out = temp / "images"
        images_out.mkdir()
        provenance, checksums = [], []
        for rank, row in enumerate(labels):
            source = balanced_by_id.get(str(row["id"]))
            if source is None or source.get("image") != row.get("source_image"):
                raise ValueError(f"rank {rank}: source provenance mismatch")
            source_classes = [name for name, value in source.get("labels", {}).items() if value == 1]
            if len(source_classes) != 1:
                raise ValueError(f"rank {rank}: expected one CheXpert class")
            expected_name = f"phi_{rank}.png"
            if row.get("image") != expected_name:
                raise ValueError(f"rank {rank}: expected {expected_name}")
            fields = row.get("phi_list")
            if not isinstance(fields, list) or not 2 <= len(fields) <= 4 or len(fields) != len(set(fields)):
                raise ValueError(f"rank {rank}: invalid PHI field list")
            path = image_dir / expected_name
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                fmt, (width, height) = image.format, image.size
                image.verify()
            digest = sha256(path)
            shutil.copyfile(path, images_out / expected_name)
            checksums.append(f"{digest}  images/{expected_name}")
            provenance.append({
                "rank": rank,
                "id": row["id"],
                "source_image": row["source_image"],
                "source_class": source_classes[0],
                "image": expected_name,
                "phi_list": fields,
                "phi_types": [field_type(value) for value in fields],
                "field_count": len(fields),
                "sha256": digest,
                "format": fmt,
                "width": width,
                "height": height,
            })

        manifest = {
            "schema_version": 1,
            "task_id": "P.2",
            "dataset": "CheXpert synthetic PHI-detection cohort",
            "sample_count": EXPECTED,
            "image_count": EXPECTED,
            "field_count_distribution": {str(k): v for k, v in sorted(Counter(len(row["phi_list"]) for row in labels).items())},
            "source_disease_count": 13,
            "source_samples_per_disease": 20,
            "selection_seed": 42,
            "overlay_seed": 42,
            "historical_runtime_compatible": True,
            "requires_model_rerun": False,
            "cancelled_related_task": "privacy-inflow",
            "task_scope": "PHI detection only",
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
