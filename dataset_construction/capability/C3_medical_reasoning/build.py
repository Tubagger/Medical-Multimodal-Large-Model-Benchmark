#!/usr/bin/env python3
"""Build the verified C.3 dataset with a uniform five-option MCQ protocol.

The frozen input was selected upstream with seed 42 as 15 rows for each of 18
clinical VQA tasks.  The original balanced TSV remains the source of images,
questions, answers, metadata, and original options.  A completed TSV supplies
only the option fields that were blank in the original artifact.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


SEED = 42
EXPECTED_TASKS = 18
SAMPLES_PER_TASK = 15
LETTERS = "ABCDE"
FORMAT_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def read_rows(tsv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(sys.maxsize)
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSV has no header")
        return list(reader.fieldnames), list(reader)


def image_format(payload: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(payload)) as image:
        image_format_name = image.format
        size = image.size
        image.verify()
    if image_format_name not in FORMAT_SUFFIX:
        raise ValueError(f"Unsupported image format: {image_format_name}")
    return image_format_name, size


def build(source_tsv: Path, completed_tsv: Path, output_root: Path) -> Path:
    required_columns = {
        "index",
        "question",
        "A",
        "B",
        "C",
        "D",
        "E",
        "answer",
        "image",
        "category",
        "clinical VQA task",
        "department",
        "perceptual granularity",
        "modality",
        "original task",
    }
    fieldnames, rows = read_rows(source_tsv)
    missing_columns = required_columns - set(fieldnames)
    if missing_columns:
        raise ValueError(f"Missing TSV columns: {sorted(missing_columns)}")
    completed_fieldnames, completed_rows = read_rows(completed_tsv)
    missing_completed_columns = required_columns - set(completed_fieldnames)
    if missing_completed_columns:
        raise ValueError(f"Missing completed-TSV columns: {sorted(missing_completed_columns)}")
    if len(completed_rows) != len(rows):
        raise ValueError("Original and completed TSV row counts differ")

    task_counts = Counter(row["clinical VQA task"].strip() for row in rows)
    if len(task_counts) != EXPECTED_TASKS or any(count != SAMPLES_PER_TASK for count in task_counts.values()):
        raise ValueError(f"Input is not {EXPECTED_TASKS} tasks x {SAMPLES_PER_TASK} rows: {task_counts}")

    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".GMAI-", dir=output_root) as temp_name:
        temp_dir = Path(temp_name)
        output_images = temp_dir / "images"
        output_images.mkdir()
        labels_out: list[dict] = []
        provenance_out: list[dict] = []
        checksum_lines: list[str] = []
        image_hashes: set[str] = set()

        for source_row, (row, completed_row) in enumerate(zip(rows, completed_rows)):
            source_id_text = row["index"].strip()
            completed_id_text = completed_row["index"].strip()
            if completed_id_text != source_id_text:
                raise ValueError(f"Original/completed row ID mismatch at source row {source_row}")
            for field in required_columns - set(LETTERS) - {"image"}:
                if row[field].strip() != completed_row[field].strip():
                    raise ValueError(
                        f"Completed TSV changed non-option field {field!r} at source row {source_row}"
                    )
            if row["image"] != completed_row["image"]:
                raise ValueError(f"Completed TSV changed image payload at source row {source_row}")
            try:
                source_id: int | str = int(source_id_text)
            except ValueError:
                source_id = source_id_text
            # Preserve the frozen prompt text byte-for-byte so historical logs
            # can be matched to the rebuilt benchmark without prompt drift.
            question = row["question"]
            answer = row["answer"].strip()
            category = row["category"].strip()
            original_options = {
                letter: row[letter].strip() for letter in LETTERS if row[letter].strip()
            }
            options = {
                letter: completed_row[letter].strip()
                for letter in LETTERS
                if completed_row[letter].strip()
            }
            for letter, value in original_options.items():
                if options.get(letter) != value:
                    raise ValueError(
                        f"Completed TSV changed original option {letter} at source row {source_row}"
                    )
            if answer not in options:
                raise ValueError(f"Correct answer option is missing at source row {source_row}")
            if list(options) != list(LETTERS):
                raise ValueError(f"Completed row does not contain five options at source row {source_row}")
            normalized_options = [normalized_text(value) for value in options.values()]
            if len(set(normalized_options)) != len(normalized_options):
                raise ValueError(f"Duplicate options at source row {source_row}")
            if normalized_text(options[answer]) != normalized_text(category):
                raise ValueError(f"Answer/category mismatch at source row {source_row}")

            try:
                image_payload = base64.b64decode(row["image"], validate=True)
            except Exception as exc:
                raise ValueError(f"Invalid base64 image at source row {source_row}: {exc}") from exc
            format_name, (width, height) = image_format(image_payload)
            digest = sha256_bytes(image_payload)
            if digest in image_hashes:
                raise ValueError(f"Duplicate image content at source row {source_row}")
            image_hashes.add(digest)
            image_name = f"image_{source_row:06d}{FORMAT_SUFFIX[format_name]}"
            image_path = output_images / image_name
            image_path.write_bytes(image_payload)

            option_text = "\n".join(f"{letter}. {value}" for letter, value in options.items())
            labels_out.append(
                {
                    "id": source_id,
                    "image": image_name,
                    "question": question,
                    "answer": answer,
                    "options": option_text,
                    "option_count": len(options),
                    "category": category,
                    "clinical_vqa_task": row["clinical VQA task"].strip(),
                    "department": row["department"].strip(),
                    "perceptual_granularity": row["perceptual granularity"].strip(),
                    "modality": row["modality"].strip(),
                    "original_task": row["original task"].strip(),
                }
            )
            provenance_out.append(
                {
                    "id": source_id,
                    "source_row": source_row,
                    "source_index": source_id_text,
                    "source_artifact": completed_tsv.name,
                    "original_option_letters": list(original_options),
                    "synthetic_option_letters": [
                        letter for letter in options if letter not in original_options
                    ],
                    "image_format": format_name,
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
            )
            checksum_lines.append(f"{digest}  images/{image_name}")

        (temp_dir / "labels.json").write_text(
            json.dumps(labels_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (temp_dir / "provenance.jsonl").open("w", encoding="utf-8") as handle:
            for row in provenance_out:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temp_dir / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        option_count_distribution = Counter(row["option_count"] for row in labels_out)
        missing_metadata_counts = {
            field: sum(not row[field].strip() for row in rows)
            for field in ["department", "perceptual granularity", "modality", "original task"]
        }
        synthetic_options_added = sum(
            len(row["synthetic_option_letters"]) for row in provenance_out
        )
        rows_with_synthetic_options = sum(
            bool(row["synthetic_option_letters"]) for row in provenance_out
        )
        manifest = {
            "schema_version": 1,
            "task_id": "C.3",
            "dataset": "GMAI",
            "upstream_sampling_seed": SEED,
            "source_artifact": completed_tsv.name,
            "source_artifact_role": "uniform five-option TSV derived from the frozen seed-42 balanced TSV",
            "original_source_artifact": source_tsv.name,
            "sample_count": len(labels_out),
            "task_count": len(task_counts),
            "samples_per_task": SAMPLES_PER_TASK,
            "task_counts": dict(sorted(task_counts.items())),
            "option_count_distribution": {
                str(count): frequency for count, frequency in sorted(option_count_distribution.items())
            },
            "synthetic_options_added": synthetic_options_added,
            "rows_with_synthetic_options": rows_with_synthetic_options,
            "missing_source_metadata_counts": missing_metadata_counts,
            "source_tsv_sha256": sha256_file(source_tsv),
            "completed_tsv_sha256": sha256_file(completed_tsv),
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_dir.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tsv", required=True, type=Path)
    parser.add_argument("--completed-tsv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.source_tsv.resolve(), args.completed_tsv.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
