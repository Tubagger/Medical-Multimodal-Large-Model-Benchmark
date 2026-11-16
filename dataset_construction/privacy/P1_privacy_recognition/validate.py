#!/usr/bin/env python3
"""Validate P.1 structure, source replay, rendering, and frozen payloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops


CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
]
TYPE_LABEL = {"clean": 0, "phi": 1, "noise": 2}


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
    spec = importlib.util.spec_from_file_location("p1_original_constructor", path)
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
    check(Counter(row["label"] for row in labels) == Counter({0: 87, 1: 87, 2: 86}), "class_balance")
    check(Counter(row["type"] for row in labels) == Counter({"clean": 87, "phi": 87, "noise": 86}), "condition_balance")
    check(all(row["label"] == TYPE_LABEL.get(row["type"]) for row in labels), "type_label_alignment")
    check(all(row["image"].startswith(row["type"] + "_") for row in labels), "filename_condition_alignment")

    replayed = replay_selection(source_labels)
    check(replayed == balanced, "seed42_source_selection_replay")
    positive_counts = Counter()
    for row in balanced:
        positives = [name for name in CLASSES if row.get("labels", {}).get(name, 0) == 1]
        for name in positives:
            positive_counts[name] += 1
    check(all(positive_counts[name] == 20 for name in CLASSES), "source_disease_balance")
    check(all(sum(value == 1 for value in row.get("labels", {}).values()) == 1 for row in balanced), "single_label_sources")

    by_id = {str(row["id"]): row for row in provenance}
    checksum_lines = set((dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    observed_checksum_lines = set()
    clean_unchanged = phi_changed = noise_changed = 0
    yellow_by_type = defaultdict(list)
    for row in labels:
        path = dataset / "images" / row["image"]
        record = by_id.get(str(row["id"]))
        if not path.is_file() or record is None:
            errors.append(f"missing image/provenance for ID {row['id']}")
            continue
        digest = sha256(path)
        observed_checksum_lines.add(f"{digest}  images/{row['image']}")
        if digest != record["sha256"]:
            errors.append(f"checksum mismatch for ID {row['id']}")
        with Image.open(path) as image:
            image.load()
            yellow = sum(r >= 180 and g >= 180 and b <= 100 for r, g, b in image.convert("RGB").getdata())
            if image.format != record["format"] or image.size != (record["width"], record["height"]):
                errors.append(f"metadata mismatch for ID {row['id']}")
        yellow_by_type[row["type"]].append(yellow)
        source_path = balanced_images / row["original_image"]
        with Image.open(source_path) as source_image, Image.open(path) as released_image:
            source_rgb, released_rgb = source_image.convert("RGB"), released_image.convert("RGB")
            changed = source_rgb.size != released_rgb.size or ImageChops.difference(source_rgb, released_rgb).getbbox() is not None
        if row["type"] == "clean": clean_unchanged += not changed
        elif row["type"] == "phi": phi_changed += changed
        else: noise_changed += changed
    check(observed_checksum_lines == checksum_lines, "checksum_manifest")
    check(clean_unchanged == 87, "clean_matches_source")
    check(phi_changed == 87, "all_phi_images_modified")
    check(noise_changed == 86, "all_noise_images_modified")
    check(all(value == 0 for value in yellow_by_type["clean"]), "clean_has_no_synthetic_yellow")
    check(all(value > 0 for value in yellow_by_type["phi"]), "phi_overlay_present")
    check(all(value > 0 for value in yellow_by_type["noise"]), "noise_overlay_present")

    module = load_constructor(constructor)
    module.set_seed(42)
    rendered, replay_labels = module.process_data(balanced, str(balanced_images))
    check(replay_labels == labels, "overlay_metadata_replay")
    pixel_equal = 0
    for image, name in rendered:
        with Image.open(dataset / "images" / name) as released:
            released_rgb = released.convert("RGB")
            pixel_equal += image.size == released_rgb.size and ImageChops.difference(image, released_rgb).getbbox() is None
    check(pixel_equal == 260, "overlay_pixel_replay", f"{pixel_equal}/260")

    values = []
    for row in labels:
        values.append((by_id[str(row["id"])]["synthetic_yellow_pixel_count"], row["label"]))
    best_accuracy, best_threshold = 0.0, None
    for threshold in range(max(value for value, _ in values) + 2):
        predicted = [0 if value == 0 else (1 if value >= threshold else 2) for value, _ in values]
        accuracy = sum(prediction == truth for prediction, (_, truth) in zip(predicted, values)) / len(values)
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, threshold

    report = {
        "schema_version": 1,
        "task_id": "P.1",
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels),
        "image_count": len(list((dataset / "images").glob("*"))),
        "class_counts": dict(sorted(Counter(row["label"] for row in labels).items())),
        "condition_counts": dict(sorted(Counter(row["type"] for row in labels).items())),
        "source_disease_count": len(CLASSES),
        "checks": checks,
        "accepted_observations": {
            "best_accuracy_from_yellow_pixel_count_only": best_accuracy,
            "best_yellow_pixel_threshold": best_threshold,
            "clean_definition": "no benchmark-added overlay",
            "phi_fields": "synthetic benchmark identifiers",
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
