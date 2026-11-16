#!/usr/bin/env python3
"""Build deterministic, strictly single-label C.1 subsets.

The source directory for each dataset must contain ``labels.json`` and an
``images`` directory.  The builder never modifies the source dataset and
refuses to overwrite an existing output subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path


SEED = 42
SAMPLES_PER_CLASS = 10
LETTERS = "ABCDE"
DEFAULT_EXCLUSIONS = Path(__file__).with_name("manual_exclusions.json")

DATASETS = {
    "CheXpert": {
        "classes": [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
        ],
        "question": "Considering the chest X-ray image, which finding corresponds most accurately?",
        "display": {"No Finding": "no abnormal finding"},
        "patient_pattern": r"/(patient\d+)/",
        "source_path_pattern": None,
    },
    "MRI": {
        "classes": ["glioma", "meningioma", "pituitary", "notumor"],
        "question": "Considering the MRI image, which diagnosis corresponds most accurately?",
        "display": {"notumor": "no tumor"},
        "patient_pattern": None,
        "source_path_pattern": r"^Testing/",
    },
    "OCT": {
        "classes": ["NORMAL", "CNV", "DME", "DRUSEN"],
        "question": "Considering the retinal OCT image, which diagnosis corresponds most accurately?",
        "display": {
            "NORMAL": "normal",
            "CNV": "choroidal neovascularization",
            "DME": "diabetic macular edema",
            "DRUSEN": "drusen",
        },
        "patient_pattern": r"(?:NORMAL|CNV|DME|DRUSEN)-([0-9]+)-",
        "source_path_pattern": r"/test/",
    },
    "ISIC": {
        "classes": [
            "actinic keratosis",
            "basal cell carcinoma",
            "dermatofibroma",
            "melanoma",
            "nevus",
            "pigmented benign keratosis",
            "seborrheic keratosis",
            "squamous cell carcinoma",
            "vascular lesion",
        ],
        "question": "Considering the dermoscopic image, which diagnosis corresponds most accurately?",
        "display": {},
        "patient_pattern": None,
        "source_path_pattern": r"^Train/",
    },
    "MURA": {
        "classes": [
            "Abnormal_ELBOW",
            "Abnormal_FINGER",
            "Abnormal_FOREARM",
            "Abnormal_HAND",
            "Abnormal_HUMERUS",
            "Abnormal_SHOULDER",
            "Abnormal_WRIST",
            "No Finding",
        ],
        "question": "Considering the musculoskeletal X-ray image, which finding corresponds most accurately?",
        "display": {"No Finding": "no abnormal finding"},
        "patient_pattern": r"/(patient\d+)/",
        "source_path_pattern": r"^valid/",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_label(row: dict, expected_classes: list[str]) -> str | None:
    labels = row.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(expected_classes):
        return None
    if any(value not in (0, 1) for value in labels.values()):
        return None
    positives = [name for name, value in labels.items() if value == 1]
    return positives[0] if len(positives) == 1 else None


def patient_id(row: dict, pattern: str | None) -> str | None:
    if not pattern:
        return None
    match = re.search(pattern, str(row.get("original_path", "")))
    return match.group(1) if match else None


def display_label(dataset: str, label: str) -> str:
    return DATASETS[dataset]["display"].get(label, label)


def build_options(
    dataset: str,
    correct: str,
    class_index: int,
    item_index: int,
    rng: random.Random,
) -> tuple[str, str]:
    config = DATASETS[dataset]
    candidates = [display_label(dataset, label) for label in config["classes"]]
    if len(candidates) < 5:
        candidates.append("other")
    correct_text = display_label(dataset, correct)
    distractors = [value for value in candidates if value != correct_text]
    chosen = rng.sample(distractors, 4)
    answer_position = (class_index * SAMPLES_PER_CLASS + item_index) % 5
    chosen.insert(answer_position, correct_text)
    options = "\n".join(f"{LETTERS[index]}. {value}" for index, value in enumerate(chosen))
    return options, LETTERS[answer_position]


def load_exclusions(path: Path, dataset: str) -> tuple[set[str], list[dict]]:
    if not path.is_file():
        return set(), []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get(dataset, [])
    ids = {str(record["source_id"]) for record in records}
    if len(ids) != len(records):
        raise ValueError(f"Duplicate exclusion IDs for {dataset} in {path}")
    return ids, records


def build_dataset(dataset: str, source_dir: Path, output_root: Path, exclusion_file: Path) -> Path:
    config = DATASETS[dataset]
    source_labels = source_dir / "labels.json"
    source_images = source_dir / "images"
    if not source_labels.is_file() or not source_images.is_dir():
        raise FileNotFoundError(f"Expected labels.json and images/ under {source_dir}")

    destination = output_root / dataset
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")

    rows = json.loads(source_labels.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    excluded_ids, exclusion_records = load_exclusions(exclusion_file, dataset)
    groups: dict[str, list[dict]] = defaultdict(list)
    rejected = 0
    outside_source_scope = 0
    for row in rows:
        if str(row.get("id")) in excluded_ids:
            continue
        source_path_pattern = config["source_path_pattern"]
        if source_path_pattern and not re.search(source_path_pattern, str(row.get("original_path", ""))):
            outside_source_scope += 1
            continue
        label = positive_label(row, config["classes"])
        if label is None:
            rejected += 1
            continue
        groups[label].append(row)

    missing = {
        label: len(groups[label])
        for label in config["classes"]
        if len(groups[label]) < SAMPLES_PER_CLASS
    }
    if missing:
        raise ValueError(f"Insufficient strictly clean candidates: {missing}")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=output_root) as temp_name:
        temp_dir = Path(temp_name)
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        labels_out: list[dict] = []
        provenance_out: list[dict] = []
        checksums: list[str] = []
        used_patients: set[str] = set()
        used_source_ids: set[str] = set()
        used_image_hashes: set[str] = set()

        output_index = 0
        for class_index, label in enumerate(config["classes"]):
            candidates = list(groups[label])
            rng.shuffle(candidates)
            selected: list[dict] = []
            for row in candidates:
                source_id = str(row.get("id"))
                pid = patient_id(row, config["patient_pattern"])
                if source_id in used_source_ids:
                    continue
                if config["patient_pattern"] and pid is None:
                    continue
                if pid is not None and pid in used_patients:
                    continue
                source_image = source_images / str(row.get("image", ""))
                if not source_image.is_file():
                    continue
                source_digest = sha256_file(source_image)
                if source_digest in used_image_hashes:
                    continue
                selected.append(row)
                used_source_ids.add(source_id)
                used_image_hashes.add(source_digest)
                if pid is not None:
                    used_patients.add(pid)
                if len(selected) == SAMPLES_PER_CLASS:
                    break
            if len(selected) != SAMPLES_PER_CLASS:
                raise ValueError(f"Could not select {SAMPLES_PER_CLASS} unique rows for {dataset}/{label}")

            for item_index, row in enumerate(selected):
                source_image = source_images / row["image"]
                suffix = source_image.suffix.lower() or ".img"
                output_name = f"{dataset.lower()}_{output_index:04d}{suffix}"
                output_image = image_dir / output_name
                shutil.copy2(source_image, output_image)
                digest = sha256_file(output_image)
                options, answer = build_options(dataset, label, class_index, item_index, rng)
                sample_id = f"{dataset}:{row['id']}"
                pid = patient_id(row, config["patient_pattern"])

                labels_out.append(
                    {
                        "id": sample_id,
                        "image": output_name,
                        "dataset": dataset,
                        "question": config["question"],
                        "options": options,
                        "answer": answer,
                        "label": display_label(dataset, label),
                    }
                )
                provenance_out.append(
                    {
                        "id": sample_id,
                        "source_id": row["id"],
                        "source_image": row["image"],
                        "original_path": row.get("original_path"),
                        "patient_id": pid,
                        "source_labels": row["labels"],
                        "selected_label": label,
                        "sha256": digest,
                    }
                )
                checksums.append(f"{digest}  images/{output_name}")
                output_index += 1

        (temp_dir / "labels.json").write_text(
            json.dumps(labels_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (temp_dir / "provenance.jsonl").open("w", encoding="utf-8") as handle:
            for row in provenance_out:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (temp_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "task_id": "C.1",
            "dataset": dataset,
            "seed": SEED,
            "random_method": "single random.Random(42) instance per dataset build",
            "samples_per_class": SAMPLES_PER_CLASS,
            "classes": config["classes"],
            "sample_count": len(labels_out),
            "strictly_clean_candidate_count": sum(len(groups[label]) for label in config["classes"]),
            "rejected_source_rows": rejected,
            "outside_source_scope_rows": outside_source_scope,
            "source_path_pattern": config["source_path_pattern"],
            "manual_exclusions": exclusion_records,
            "manual_exclusions_sha256": sha256_file(exclusion_file) if exclusion_file.is_file() else None,
            "source_labels_sha256": sha256_file(source_labels),
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_dir.rename(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exclude-file", type=Path, default=DEFAULT_EXCLUSIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_dataset(
        args.dataset,
        args.source_dir.resolve(),
        args.output_root.resolve(),
        args.exclude_file.resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()
