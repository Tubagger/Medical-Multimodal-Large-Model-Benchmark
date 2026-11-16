#!/usr/bin/env python3
"""Build the corrected R.1 OOD subset from the frozen GMAI OOD TSV.

The historical 219-row artifact cannot be reproduced by the documented
``at most 20 samples per modality`` rule: six modalities contain 25 rows.
This builder normalizes whitespace in modality names and samples at most 25
rows per normalized modality with the original NumPy/Pandas random-state
semantics.  Final five-option inputs are read from the frozen runtime labels;
their original and added option letters remain explicit in provenance.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SEED = 42
MAX_PER_MODALITY = 25
EXPECTED_MODALITIES = 15
EXPECTED_SAMPLES = 219
LETTERS = "ABCDE"
FORMAT_SUFFIX = {"PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tiff"}
REQUIRED_COLUMNS = {
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


def normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def normalized_modality(value: str) -> str:
    return " ".join(value.split())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_runtime_options(value: object) -> dict[str, str]:
    if not isinstance(value, str):
        raise ValueError("Runtime options are not a string")
    output: dict[str, str] = {}
    for line in value.splitlines():
        match = re.fullmatch(r"([A-E])\. (.+)", line)
        if match is None or match.group(1) in output:
            raise ValueError(f"Malformed runtime options: {value!r}")
        output[match.group(1)] = match.group(2).strip()
    return output


def image_format(payload: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(payload)) as image:
        format_name = image.format
        size = image.size
        image.verify()
    if format_name not in FORMAT_SUFFIX:
        raise ValueError(f"Unsupported image format: {format_name}")
    return format_name, size


def open_reader(path: Path):
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None:
        handle.close()
        raise ValueError("TSV has no header")
    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        handle.close()
        raise ValueError(f"Missing TSV columns: {sorted(missing)}")
    return handle, reader


def select_source_rows(source_tsv: Path) -> tuple[list[int], dict[str, int], int]:
    """Replay pandas ``groupby(...).sample(random_state=42)`` without images."""
    csv.field_size_limit(sys.maxsize)
    groups: dict[str, list[int]] = defaultdict(list)
    source_rows = 0
    handle, reader = open_reader(source_tsv)
    try:
        for source_row, row in enumerate(reader):
            modality = normalized_modality(row["modality"])
            if not modality:
                raise ValueError(f"Missing modality at source row {source_row}")
            groups[modality].append(source_row)
            source_rows += 1
    finally:
        handle.close()

    if len(groups) != EXPECTED_MODALITIES:
        raise ValueError(f"Expected {EXPECTED_MODALITIES} normalized modalities, found {len(groups)}")

    selected: list[int] = []
    selected_counts: dict[str, int] = {}
    for modality in sorted(groups):
        candidates = groups[modality]
        sample_size = min(MAX_PER_MODALITY, len(candidates))
        # pandas DataFrame.sample(random_state=<int>) creates the same legacy
        # RandomState independently for every group in the historical lambda.
        offsets = np.random.RandomState(SEED).choice(len(candidates), size=sample_size, replace=False)
        selected.extend(candidates[int(offset)] for offset in offsets)
        selected_counts[modality] = sample_size

    if len(selected) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} selected rows, found {len(selected)}")
    return selected, selected_counts, source_rows


def build(source_tsv: Path, runtime_labels_path: Path, output_root: Path, source_sha256: str) -> Path:
    selected_rows, modality_counts, source_rows = select_source_rows(source_tsv)
    output_rank = {source_row: rank for rank, source_row in enumerate(selected_rows)}
    if len(output_rank) != len(selected_rows):
        raise ValueError("Sampling selected a source row more than once")

    runtime_labels = json.loads(runtime_labels_path.read_text(encoding="utf-8"))
    if not isinstance(runtime_labels, list) or len(runtime_labels) != EXPECTED_SAMPLES:
        raise ValueError(f"Runtime labels must contain {EXPECTED_SAMPLES} rows")
    runtime_by_id = {str(row.get("id")): row for row in runtime_labels}
    if len(runtime_by_id) != EXPECTED_SAMPLES:
        raise ValueError("Runtime labels contain duplicate IDs")

    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".GMAI-R1-", dir=output_root) as temp_name:
        temp_dir = Path(temp_name)
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        labels: list[dict | None] = [None] * len(selected_rows)
        provenance: list[dict | None] = [None] * len(selected_rows)
        checksum_lines: list[str | None] = [None] * len(selected_rows)
        image_hashes: set[str] = set()

        handle, reader = open_reader(source_tsv)
        try:
            for source_row, row in enumerate(reader):
                rank = output_rank.get(source_row)
                if rank is None:
                    continue
                source_id_text = row["index"].strip()
                try:
                    source_id: int | str = int(source_id_text)
                except ValueError:
                    source_id = source_id_text
                question = row["question"].strip()
                answer = row["answer"].strip()
                category = row["category"].strip()
                modality_original = row["modality"]
                modality = normalized_modality(modality_original)
                options = {letter: row[letter].strip() for letter in LETTERS if row[letter].strip()}
                if not question:
                    raise ValueError(f"Missing question at source row {source_row}")
                if answer not in options:
                    raise ValueError(f"Answer is absent from original options at source row {source_row}")
                if not 2 <= len(options) <= 5:
                    raise ValueError(f"Invalid option count at source row {source_row}: {len(options)}")
                normalized_options = [normalized_text(value) for value in options.values()]
                if len(set(normalized_options)) != len(normalized_options):
                    raise ValueError(f"Duplicate original options at source row {source_row}")
                if normalized_text(options[answer]) != normalized_text(category):
                    raise ValueError(f"Answer/category mismatch at source row {source_row}")

                runtime_row = runtime_by_id.get(source_id_text)
                if runtime_row is None:
                    raise ValueError(f"Selected ID {source_id_text} is absent from runtime labels")
                if str(runtime_labels[rank].get("id")) != source_id_text:
                    raise ValueError(f"Runtime order differs from seed-42 cap-25 selection at rank {rank}")
                if runtime_row.get("question", "").strip() != question:
                    raise ValueError(f"Runtime question mismatch for ID {source_id_text}")
                if runtime_row.get("answer", "").strip() != answer:
                    raise ValueError(f"Runtime answer mismatch for ID {source_id_text}")
                final_options = parse_runtime_options(runtime_row.get("options"))
                if list(final_options) != list(LETTERS):
                    raise ValueError(f"Runtime options are not A--E for ID {source_id_text}")
                for letter, value in options.items():
                    if final_options.get(letter) != value:
                        raise ValueError(f"Runtime changed original option {letter} for ID {source_id_text}")
                synthetic_letters = [letter for letter in LETTERS if letter not in options]

                try:
                    payload = base64.b64decode(row["image"], validate=True)
                except Exception as exc:
                    raise ValueError(f"Invalid base64 image at source row {source_row}: {exc}") from exc
                format_name, (width, height) = image_format(payload)
                digest = sha256_bytes(payload)
                if digest in image_hashes:
                    raise ValueError(f"Duplicate selected image content at source row {source_row}")
                image_hashes.add(digest)
                image_name = f"image_{rank:06d}{FORMAT_SUFFIX[format_name]}"
                (image_dir / image_name).write_bytes(payload)
                runtime_image = runtime_labels_path.parent / "images" / str(runtime_row.get("image", ""))
                if not runtime_image.is_file() or sha256_file(runtime_image) != digest:
                    raise ValueError(f"Runtime image mismatch for ID {source_id_text}")

                labels[rank] = {
                    "id": source_id,
                    "image": image_name,
                    "question": question,
                    "answer": answer,
                    "options": "\n".join(f"{letter}. {value}" for letter, value in final_options.items()),
                    "option_count": len(final_options),
                    "category": category,
                    "clinical_vqa_task": row["clinical VQA task"].strip(),
                    "department": row["department"].strip(),
                    "perceptual_granularity": row["perceptual granularity"].strip(),
                    "modality": modality,
                    "original_task": row["original task"].strip(),
                }
                provenance[rank] = {
                    "id": source_id,
                    "source_row": source_row,
                    "source_index": source_id_text,
                    "source_artifact": source_tsv.name,
                    "modality_original": modality_original,
                    "modality_normalized": modality,
                    "original_option_letters": list(options),
                    "added_option_letters": synthetic_letters,
                    "image_format": format_name,
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
                checksum_lines[rank] = f"{digest}  images/{image_name}"
        finally:
            handle.close()

        if any(row is None for row in labels) or any(row is None for row in provenance):
            raise ValueError("Not every selected source row was materialized")
        labels_out = [row for row in labels if row is not None]
        provenance_out = [row for row in provenance if row is not None]
        task_counts = Counter(row["clinical_vqa_task"] for row in labels_out)
        option_counts = Counter(row["option_count"] for row in labels_out)
        added_option_rows = sum(bool(row["added_option_letters"]) for row in provenance_out)
        added_option_fields = sum(len(row["added_option_letters"]) for row in provenance_out)
        normalized_duplicate_option_ids = [
            row["id"]
            for row in labels_out
            if len({normalized_text(value) for value in parse_runtime_options(row["options"]).values()})
            != len(LETTERS)
        ]
        manifest = {
            "schema_version": 1,
            "task_id": "R.1",
            "dataset": "GMAI-MMBench OOD",
            "upstream_dataset": {
                "name": "GMAI-MMBench",
                "split": "VAL",
                "repository": "https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench",
                "declared_license": "apache-2.0",
                "access_note": "The upstream repository is gated and requires acceptance of its access conditions.",
            },
            "source_artifact": source_tsv.name,
            "source_artifact_role": "frozen OOD-filtered TSV derived from the GMAI-MMBench validation split",
            "source_artifact_sha256": source_sha256,
            "source_row_count": source_rows,
            "selection_seed": SEED,
            "selection_rule": "normalize modality whitespace, then sample min(25, modality_count) with an independent numpy.random.RandomState(42) per modality",
            "sample_count": len(labels_out),
            "modality_count": len(modality_counts),
            "modality_counts": modality_counts,
            "clinical_vqa_task_counts": dict(sorted(task_counts.items())),
            "option_count_distribution": {str(k): v for k, v in sorted(option_counts.items())},
            "runtime_labels_artifact": str(runtime_labels_path),
            "runtime_labels_sha256": sha256_file(runtime_labels_path),
            "added_option_rows": added_option_rows,
            "added_option_fields": added_option_fields,
            "normalized_duplicate_option_ids": normalized_duplicate_option_ids,
            "annotation_authority": "published benchmark release",
            "secondary_clinical_review_required": False,
            "historical_runtime_compatible": True,
        }
        (temp_dir / "labels.json").write_text(
            json.dumps(labels_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (temp_dir / "provenance.jsonl").open("w", encoding="utf-8") as output:
            for row in provenance_out:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temp_dir / "checksums.sha256").write_text(
            "\n".join(line for line in checksum_lines if line is not None) + "\n", encoding="utf-8"
        )
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_dir.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tsv", required=True, type=Path)
    parser.add_argument("--runtime-labels", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        raise SystemExit("--source-sha256 must be a lowercase SHA-256 digest")
    print(
        build(
            args.source_tsv.resolve(),
            args.runtime_labels.resolve(),
            args.output_root.resolve(),
            args.source_sha256,
        )
    )


if __name__ == "__main__":
    main()
