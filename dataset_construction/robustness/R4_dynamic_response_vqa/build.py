#!/usr/bin/env python3
"""Materialize the frozen 200-item R.4 base cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image


EXPECTED = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(labels_path: Path, image_dir: Path, output_root: Path) -> Path:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if len(labels) != EXPECTED or len({row["id"] for row in labels}) != EXPECTED:
        raise ValueError("R.4 must contain exactly 200 unique IDs")
    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".R5-GMAI-", dir=output_root) as temp_name:
        temp = Path(temp_name)
        images_out = temp / "images"
        images_out.mkdir()
        provenance = []
        checksums = []
        for rank, row in enumerate(labels):
            name = Path(row["image"]).name
            if name != row["image"]:
                raise ValueError(f"Unsafe image name for ID {row['id']}")
            source = image_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                fmt, (width, height) = str(image.format), image.size
                image.verify()
            digest = sha256(source)
            shutil.copyfile(source, images_out / name)
            checksums.append(f"{digest}  images/{name}")
            provenance.append({
                "rank": rank, "id": row["id"], "image": name,
                "sha256": digest, "format": fmt, "width": width, "height": height,
            })
        manifest = {
            "schema_version": 1,
            "task_id": "R.4",
            "dataset": "PMC-VQA dynamic response-robustness base cohort",
            "sample_count": EXPECTED,
            "selection_seed": 42,
            "dynamic_branches": [
                "answer-flip", "question-negation", "option-expansion", "narrative-distraction"
            ],
            "dynamic_protocol_preserved": True,
            "question_negation_answer_type": "multi-label set",
            "historical_runtime_compatible": True,
            "requires_model_rerun": False,
        }
        (temp / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (temp / "provenance.jsonl").open("w", encoding="utf-8") as handle:
            for row in provenance:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temp / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.labels.resolve(), args.image_dir.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
