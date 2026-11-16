#!/usr/bin/env python3
"""Audit F.2 image integrity, duplication, and saved-log image mappings.

This script is deliberately read-only with respect to the dataset.  It writes
only reproducible audit artifacts (JSON, CSV, SHA256 and contact sheets).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


BRANCHES = (
    "bias-vqa-race",
    "bias-vqa-language",
    "bias-vqa-emotion",
    "bias-vqa-cognitive",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | (pixels[offset + col] > pixels[offset + col + 1])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def resolve_image(image_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    candidate = image_dir / path.name
    return candidate


def make_contact_sheets(rows: list[dict], destination: Path, columns: int = 7) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h, label_h = 150, 150, 22
    page_size = columns * columns
    pages = math.ceil(len(rows) / page_size)
    for page in range(pages):
        batch = rows[page * page_size : (page + 1) * page_size]
        canvas = Image.new("RGB", (columns * thumb_w, columns * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, row in enumerate(batch):
            x = (slot % columns) * thumb_w
            y = (slot // columns) * (thumb_h + label_h)
            with Image.open(row["absolute_path"]) as source:
                tile = ImageOps.fit(source.convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill="white")
            draw.text((x + 4, y + thumb_h + 4), f"id={row['case_id']}  {row['filename']}", fill="black")
        canvas.save(destination / f"page_{page + 1:02d}.jpg", quality=88, optimize=True)
    return pages


def audit_logs(log_root: Path, expected: dict[int, str]) -> dict:
    result = {
        "log_root": str(log_root),
        "model_directories": 0,
        "fully_valid_models": 0,
        "image_references_checked": 0,
        "problems": [],
    }
    if not log_root.exists():
        result["status"] = "not_found"
        return result

    for model_dir in sorted(path for path in log_root.iterdir() if path.is_dir()):
        result["model_directories"] += 1
        model_ok = True
        mappings: dict[str, dict[int, str]] = {}
        for branch in BRANCHES:
            path = model_dir / f"{branch}.json"
            if not path.exists():
                result["problems"].append([model_dir.name, branch, "missing_file"])
                mappings[branch] = {}
                model_ok = False
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            samples = payload.get("per_sample_results", [])
            mapping = {
                int(sample["id"]): sample.get("content", {}).get("image_path")
                for sample in samples
            }
            mappings[branch] = mapping
            if len(mapping) != len(expected):
                result["problems"].append([model_dir.name, branch, "row_count", len(mapping)])
                model_ok = False
            for case_id, image_path in mapping.items():
                result["image_references_checked"] += 1
                actual = Path(image_path).name if image_path else None
                if expected.get(case_id) != actual:
                    result["problems"].append(
                        [model_dir.name, branch, "mapping", case_id, actual, expected.get(case_id)]
                    )
                    model_ok = False

        if all(mappings.get(branch) for branch in BRANCHES):
            common_ids = set.intersection(*(set(mappings[branch]) for branch in BRANCHES))
            for case_id in common_ids:
                filenames = {Path(mappings[branch][case_id]).name for branch in BRANCHES}
                if len(filenames) != 1:
                    result["problems"].append(
                        [model_dir.name, "cross_branch", case_id, sorted(filenames)]
                    )
                    model_ok = False
        if model_ok:
            result["fully_valid_models"] += 1

    result["problem_count"] = len(result["problems"])
    result["status"] = "pass" if not result["problems"] else "fail"
    return result


def main() -> None:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=Path,
        default=repository / "data/fairness/F2_response_adaptive_vqa/bias_vqa.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=repository / "data/fairness/F2_response_adaptive_vqa/images",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=repository / "logs/fairness/F2",
        help="Historical directory name containing the F.2 logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "run_state/data_checks/F2",
    )
    parser.add_argument("--skip-contact-sheets", action="store_true")
    args = parser.parse_args()

    annotations = json.loads(args.annotation.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_fields = {
        "case_id",
        "category",
        "clinical_summary",
        "image",
        "patient_self-report",
        "question",
        "third-person_narration",
        "vital_sign",
    }

    inventory: list[dict] = []
    errors: list[list] = []
    references = []
    ids = []
    for item in annotations:
        case_id = int(item["case_id"])
        filename = Path(item["image"]).name
        path = resolve_image(args.image_dir, item["image"])
        ids.append(case_id)
        references.append(filename)
        missing = sorted(required_fields - set(item))
        if missing:
            errors.append([case_id, "missing_annotation_fields", missing])
        if not path.exists():
            errors.append([case_id, "missing_image", str(path)])
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                metadata_keys = sorted(image.info)
                record = {
                    "case_id": case_id,
                    "filename": filename,
                    "absolute_path": str(path.resolve()),
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "sha256": sha256(path),
                    "dhash64": f"{dhash(image):016x}",
                    "metadata_keys": metadata_keys,
                    "bytes": path.stat().st_size,
                    "category": item.get("category", ""),
                }
                inventory.append(record)
        except Exception as exc:  # Keep auditing remaining files.
            errors.append([case_id, "decode_error", repr(exc)])

    actual_files = sorted(path.name for path in args.image_dir.iterdir() if path.is_file())
    missing_references = sorted(set(references) - set(actual_files))
    extra_files = sorted(set(actual_files) - set(references))
    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    duplicate_references = sorted(name for name, count in Counter(references).items() if count > 1)

    exact_groups: dict[str, list[int]] = {}
    for row in inventory:
        exact_groups.setdefault(row["sha256"], []).append(row["case_id"])
    exact_duplicates = [group for group in exact_groups.values() if len(group) > 1]

    perceptual_pairs = []
    for left_index, left in enumerate(inventory):
        left_hash = int(left["dhash64"], 16)
        for right in inventory[left_index + 1 :]:
            distance = hamming(left_hash, int(right["dhash64"], 16))
            if distance <= 5:
                perceptual_pairs.append([left["case_id"], right["case_id"], distance])

    log_audit = audit_logs(args.log_root, {int(i): name for i, name in zip(ids, references)})

    public_inventory = [{k: v for k, v in row.items() if k != "absolute_path"} for row in inventory]
    with (args.output_dir / "image_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_inventory[0]))
        writer.writeheader()
        writer.writerows(public_inventory)
    with (args.output_dir / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for row in inventory:
            handle.write(f"{row['sha256']}  {row['filename']}\n")

    pages = 0
    if not args.skip_contact_sheets:
        pages = make_contact_sheets(inventory, args.output_dir / "contact_sheets")

    formats = Counter(row["format"] for row in inventory)
    modes = Counter(row["mode"] for row in inventory)
    dimensions = Counter(f"{row['width']}x{row['height']}" for row in inventory)
    metadata_nonempty = [row["case_id"] for row in inventory if row["metadata_keys"]]
    summary = {
        "audit_scope": "F.2 source-image integrity and saved-log image mapping",
        "annotation": str(args.annotation),
        "image_directory": str(args.image_dir),
        "annotation_rows": len(annotations),
        "unique_case_ids": len(set(ids)),
        "unique_image_references": len(set(references)),
        "image_files": len(actual_files),
        "decoded_images": len(inventory),
        "duplicate_case_ids": duplicate_ids,
        "duplicate_image_references": duplicate_references,
        "missing_references": missing_references,
        "extra_files": extra_files,
        "formats": dict(formats),
        "modes": dict(modes),
        "dimensions": dict(dimensions),
        "images_with_png_metadata": metadata_nonempty,
        "exact_sha256_duplicate_groups": exact_duplicates,
        "dhash_pairs_hamming_le_5": perceptual_pairs,
        "contact_sheet_pages": pages,
        "annotation_or_decode_errors": errors,
        "saved_log_mapping": log_audit,
    }
    summary["machine_integrity_status"] = (
        "pass"
        if not any(
            (
                errors,
                duplicate_ids,
                duplicate_references,
                missing_references,
                extra_files,
                exact_duplicates,
                perceptual_pairs,
                log_audit.get("problems", []),
            )
        )
        else "fail"
    )
    (args.output_dir / "image_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
