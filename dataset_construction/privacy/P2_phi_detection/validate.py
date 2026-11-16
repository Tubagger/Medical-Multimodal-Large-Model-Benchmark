#!/usr/bin/env python3
"""Validate and exactly replay the accepted P.2 PHI-detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops


CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
]
FIELD_PATTERNS = [
    re.compile(r"^Name: \S.+$"), re.compile(r"^Hospital: \S.+$"),
    re.compile(r"^DOB: \d{4}-\d{2}-\d{2}$"), re.compile(r"^MRN-\d{6}$"),
    re.compile(r"^ID: \d{3}-\d{2}-\d{4}$"), re.compile(r"^CT Date: \d{4}-\d{2}-\d{2}$"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_selection(source_labels: Path) -> list[dict]:
    rows = json.loads(source_labels.read_text(encoding="utf-8"))
    pools = defaultdict(list)
    for row in rows:
        for name in CLASSES:
            if row.get("labels", {}).get(name, 0) == 1:
                pools[name].append(row)
    random.seed(42)
    selected, seen = [], set()
    for name in CLASSES:
        for row in random.sample(pools[name], min(20, len(pools[name]))):
            if row["id"] not in seen:
                selected.append(row)
                seen.add(row["id"])
    return selected


def load_constructor(path: Path):
    spec = importlib.util.spec_from_file_location("p2_original_constructor", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(dataset: Path, balanced_labels: Path, balanced_images: Path,
             source_labels: Path, constructor: Path) -> dict:
    labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    provenance = [json.loads(line) for line in (dataset / "provenance.jsonl").read_text(encoding="utf-8").splitlines() if line]
    balanced = json.loads(balanced_labels.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: dict[str, str] = {}

    def check(condition: bool, name: str, detail: str = "") -> None:
        checks[name] = "passed" if condition else "failed"
        if not condition:
            errors.append(f"{name}: {detail}" if detail else name)

    check(len(labels) == len(provenance) == len(balanced) == 260, "sample_count")
    check(len({str(row["id"]) for row in labels}) == 260, "unique_ids")
    check(len({row["image"] for row in labels}) == 260, "unique_images")
    check(all(row["image"] == f"phi_{index}.png" for index, row in enumerate(labels)), "image_order")
    check(Counter(len(row["phi_list"]) for row in labels) == Counter({2: 75, 3: 87, 4: 98}), "field_count_distribution")
    check(all(len(row["phi_list"]) == len(set(row["phi_list"])) for row in labels), "unique_fields_per_image")
    check(all(any(pattern.fullmatch(value) for pattern in FIELD_PATTERNS)
              for row in labels for value in row["phi_list"]), "field_format")

    replayed = replay_selection(source_labels)
    check(replayed == balanced, "seed42_source_selection_replay")
    positive_counts = Counter()
    for row in balanced:
        for name in CLASSES:
            positive_counts[name] += row.get("labels", {}).get(name, 0) == 1
    check(all(positive_counts[name] == 20 for name in CLASSES), "source_disease_balance")
    check(all(sum(value == 1 for value in row.get("labels", {}).values()) == 1 for row in balanced), "single_label_sources")

    by_id = {str(row["id"]): row for row in provenance}
    checksum_lines = set((dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    observed_lines = set()
    changed_images = central_boundary_images = 0
    central_boundary_ids = []
    for row in labels:
        path = dataset / "images" / row["image"]
        record = by_id.get(str(row["id"]))
        if not path.is_file() or record is None:
            errors.append(f"missing image/provenance for ID {row['id']}")
            continue
        digest = sha256(path)
        observed_lines.add(f"{digest}  images/{row['image']}")
        if digest != record["sha256"] or record["phi_list"] != row["phi_list"]:
            errors.append(f"provenance mismatch for ID {row['id']}")
        source_path = balanced_images / row["source_image"]
        with Image.open(source_path) as source, Image.open(path) as released:
            source_rgb, released_rgb = source.convert("RGB"), released.convert("RGB")
            difference = ImageChops.difference(source_rgb, released_rgb)
            changed_images += difference.getbbox() is not None
            x1, y1 = int(source.width * .25), int(source.height * .25)
            x2, y2 = x1 + int(source.width * .5), y1 + int(source.height * .5)
            if difference.crop((x1, y1, x2, y2)).getbbox() is not None:
                central_boundary_images += 1
                central_boundary_ids.append(row["id"])
    check(observed_lines == checksum_lines, "checksum_manifest")
    check(changed_images == 260, "all_images_modified")

    module = load_constructor(constructor)
    module.set_seed(42)
    replay_dir = dataset.parent / ".validation-replay-images"
    replay_json = dataset.parent / ".validation-replay-labels.json"
    if replay_dir.exists() or replay_json.exists():
        raise FileExistsError("stale P.2 validation replay artifacts")
    try:
        module.process_folder(str(balanced_labels), str(balanced_images), str(replay_dir), str(replay_json))
        replay_labels = json.loads(replay_json.read_text(encoding="utf-8"))
        check(replay_labels == labels, "overlay_metadata_replay")
        byte_equal = pixel_equal = 0
        for row in labels:
            generated, released = replay_dir / row["image"], dataset / "images" / row["image"]
            byte_equal += sha256(generated) == sha256(released)
            with Image.open(generated) as first, Image.open(released) as second:
                pixel_equal += first.size == second.size and ImageChops.difference(first.convert("RGB"), second.convert("RGB")).getbbox() is None
        check(byte_equal == 260, "overlay_byte_replay", f"{byte_equal}/260")
        check(pixel_equal == 260, "overlay_pixel_replay", f"{pixel_equal}/260")
    finally:
        if replay_dir.exists():
            import shutil
            shutil.rmtree(replay_dir)
        if replay_json.exists():
            replay_json.unlink()

    report = {
        "schema_version": 1,
        "task_id": "P.2",
        "task_scope": "PHI detection only",
        "cancelled_related_task": "privacy-inflow",
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels),
        "image_count": len(list((dataset / "images").glob("*"))),
        "field_count_distribution": dict(sorted(Counter(len(row["phi_list"]) for row in labels).items())),
        "checks": checks,
        "accepted_observations": {
            "central_boundary_antialiasing_images": central_boundary_images,
            "central_boundary_ids": central_boundary_ids,
            "mrn_strings_have_no_colon": sum(value.startswith("MRN-") for row in labels for value in row["phi_list"]),
        },
        "errors": errors,
    }
    (dataset / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--balanced-labels", type=Path, required=True)
    parser.add_argument("--balanced-images", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--constructor", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.dataset.resolve(), args.balanced_labels.resolve(), args.balanced_images.resolve(),
                      args.source_labels.resolve(), args.constructor.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
