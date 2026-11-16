#!/usr/bin/env python3
"""Build corrected R.2/R.3 artifacts without model-performance filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image


EXPECTED = 200
LETTERS = "ABCDE"
REPLACEMENT_POOL = [
    "brain", "retina", "chest", "lung", "heart", "liver", "kidney",
    "stomach", "bladder", "uterus", "prostate", "femur", "humerus",
    "skin", "hand", "shoulder", "elbow", "wrist", "finger", "forearm",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_options(text: str) -> dict[str, str]:
    pairs = re.findall(r"(?m)^([A-E])\.\s*([^\n]+)$", text)
    if [letter for letter, _ in pairs] != list(LETTERS):
        raise ValueError(f"Malformed five-option block: {text!r}")
    return dict(pairs)


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in LETTERS)


def replace_duplicate(options: dict[str, str], protected: set[str]) -> None:
    while len(set(options.values())) != len(options):
        duplicate = next(value for value in options.values() if list(options.values()).count(value) > 1)
        letters = [letter for letter in LETTERS if options[letter] == duplicate]
        protected_letters = [letter for letter in letters if letter in protected]
        if len(protected_letters) > 1:
            raise ValueError(f"Protected duplicate {duplicate}")
        keep = protected_letters[0] if protected_letters else letters[0]
        for replace_letter in [letter for letter in letters if letter != keep]:
            used = set(options.values())
            options[replace_letter] = next(value for value in REPLACEMENT_POOL if value not in used)


def corrected_label(sample_id: int, corrections: dict) -> str | None:
    if sample_id in set(corrections["isic"]["ids"]):
        return corrections["isic"]["corrected_label"]
    return corrections["mura"]["labels_by_id"].get(str(sample_id))


def correct_r2(row: dict, corrections: dict) -> dict:
    result = dict(row)
    sample_id = int(row["id"])
    label = corrected_label(sample_id, corrections)
    options = parse_options(row["options"])
    answer = str(row["answer"])
    legacy_label = options[answer]
    if label is not None:
        expected = "isic" if sample_id < 80 else "mura"
        if legacy_label != expected:
            raise ValueError(f"Unexpected R.2 label for ID {sample_id}: {legacy_label}")
        options[answer] = label
        replace_duplicate(options, {answer})
        result["question"] = corrections["neutral_question"]
        result["options"] = format_options(options)
        result["legacy_ground_truth_label"] = legacy_label
        result["correction"] = "source_dataset_name_replaced_with_anatomical_label"
    result["ground_truth_answer"] = answer
    result["ground_truth_label"] = options[answer]
    return result


def correct_r3(row: dict, corrections: dict) -> dict:
    result = dict(row)
    sample_id = int(row["id"])
    options = parse_options(row["options"])
    original_label = str(row["original_answer"])
    target_label = str(row["target_answer"])
    original_letters = [letter for letter, value in options.items() if value == original_label]
    if len(original_letters) != 1:
        raise ValueError(f"Cannot locate R.3 original answer for ID {sample_id}")
    original_letter = original_letters[0]
    target_letter = str(row["answer"])
    if options[target_letter] != target_label:
        raise ValueError(f"Cannot locate R.3 target answer for ID {sample_id}")

    label = corrected_label(sample_id, corrections)
    if label is not None:
        options[original_letter] = label
        replacement_target = corrections["r3_retarget"].get(str(sample_id))
        if replacement_target:
            options[target_letter] = replacement_target
            target_label = replacement_target
            result["attack_image_regenerated"] = True
        else:
            result["attack_image_regenerated"] = False
        replace_duplicate(options, {original_letter, target_letter})
        result["question"] = corrections["neutral_question"]
        result["legacy_original_answer"] = original_label
        result["correction"] = "source_dataset_name_replaced_with_anatomical_label"

    result["options"] = format_options(options)
    result["answer"] = original_letter
    result["original_answer"] = options[original_letter]
    result["ground_truth_answer"] = original_letter
    result["ground_truth_label"] = options[original_letter]
    result["attack_target_answer"] = target_letter
    result["attack_target_label"] = target_label
    result["legacy_attack_target_answer"] = row["answer"]
    return result


def replay_source_rows(source_root: Path, archived_originals: Path) -> dict[int, dict]:
    recovered: dict[int, dict] = {}
    specs = [
        ("ISIC", 40, lambda row: all(row.get("labels", {}).get(name, 0) != 1 for name in [
            "melanoma", "basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis"
        ])),
        ("MURA", 120, lambda row: row.get("labels", {}).get("No Finding", 0) == 1 and all(
            value != 1 for name, value in row.get("labels", {}).items() if name.startswith("Abnormal_")
        )),
    ]
    for dataset, start, predicate in specs:
        dataset_root = source_root / dataset
        rows = read_json(dataset_root / "labels.json")
        selected = random.Random(42).sample([row for row in rows if predicate(row)], 40)
        for offset, source in enumerate(selected):
            sample_id = start + offset
            source_image = dataset_root / "images" / source["image"]
            archive_image = archived_originals / f"image_{sample_id}.jpg"
            if sha256(source_image) != sha256(archive_image):
                raise ValueError(f"Source replay payload mismatch for ID {sample_id}")
            recovered[sample_id] = {
                "source_dataset": dataset,
                "source_id": source["id"],
                "source_image": source["image"],
                "original_path": source.get("original_path"),
                "source_labels": source.get("labels", {}),
                "source_payload_sha256": sha256(source_image),
            }
    return recovered


def image_info(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as image:
        fmt, (width, height) = str(image.format), image.size
        image.verify()
    return fmt, width, height


def materialize_task(
    task_id: str,
    labels: list[dict],
    clean_dir: Path,
    attack_dir: Path,
    output_root: Path,
    provenance_source: dict[int, dict],
    replacement_attacks: Path | None,
    required_replacements: set[int],
) -> Path:
    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    missing_replacements: list[int] = []
    with tempfile.TemporaryDirectory(prefix=f".{task_id}-GMAI-", dir=output_root) as temp_name:
        temp = Path(temp_name)
        clean_out = temp / "clean" / "images"
        attack_out = temp / "attacked" / "images"
        clean_out.mkdir(parents=True)
        attack_out.mkdir(parents=True)
        provenance = []
        checksums = []
        for rank, row in enumerate(labels):
            sample_id = int(row["id"])
            name = str(row["image"])
            clean = clean_dir / name
            attack = attack_dir / name
            if sample_id in required_replacements:
                replacement = replacement_attacks / name if replacement_attacks else None
                if replacement is None or not replacement.is_file():
                    missing_replacements.append(sample_id)
                    attack = None
                else:
                    attack = replacement
            if not clean.is_file():
                raise FileNotFoundError(clean)
            clean_format, width, height = image_info(clean)
            clean_hash = sha256(clean)
            shutil.copyfile(clean, clean_out / name)
            checksums.append(f"{clean_hash}  clean/images/{name}")
            attack_hash = None
            attack_format = None
            if attack is not None:
                if not attack.is_file():
                    raise FileNotFoundError(attack)
                attack_format, attack_width, attack_height = image_info(attack)
                if (attack_width, attack_height) != (width, height):
                    raise ValueError(f"Pair dimensions differ for ID {sample_id}")
                attack_hash = sha256(attack)
                if attack_hash == clean_hash:
                    raise ValueError(f"Identical clean/attack pair for ID {sample_id}")
                shutil.copyfile(attack, attack_out / name)
                checksums.append(f"{attack_hash}  attacked/images/{name}")
            provenance.append({
                "rank": rank,
                "id": sample_id,
                "image": name,
                "clean_sha256": clean_hash,
                "attack_sha256": attack_hash,
                "clean_format": clean_format,
                "attack_format": attack_format,
                "width": width,
                "height": height,
                **provenance_source.get(sample_id, {}),
            })

        manifest = {
            "schema_version": 1,
            "task_id": task_id,
            "sample_count": len(labels),
            "selection": "40 label-filtered samples from each of five source datasets, sampled with seed 42; no model outputs used",
            "model_performance_filtering": False,
            "semantic_corrections": {"ISIC_to_skin": 40, "MURA_to_anatomical_region": 40},
            "historical_runtime_compatible": False,
            "requires_model_rerun": True,
            "missing_regenerated_attack_ids": missing_replacements,
            "status": "passed" if not missing_replacements else "pending_attack_regeneration",
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
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archived-originals", type=Path, required=True)
    parser.add_argument("--r2-output", type=Path, required=True)
    parser.add_argument("--r3-output", type=Path, required=True)
    parser.add_argument("--replacement-attacks", type=Path)
    args = parser.parse_args()
    corrections = read_json(Path(__file__).with_name("corrections.json"))
    legacy = args.legacy_root.resolve()
    r2_labels = [correct_r2(row, corrections) for row in read_json(legacy / "adv_untarget.json")]
    r3_labels = [correct_r3(row, corrections) for row in read_json(legacy / "adv_target_hard.json")]
    if len(r2_labels) != EXPECTED or len(r3_labels) != EXPECTED:
        raise ValueError("Expected 200 rows for both R.2 and R.3")
    source_rows = replay_source_rows(args.source_root.resolve(), args.archived_originals.resolve())
    r2 = materialize_task(
        "R.2", r2_labels, legacy / "images/clean", legacy / "images/untarget",
        args.r2_output.resolve(), source_rows, None, set(),
    )
    r3 = materialize_task(
        "R.3", r3_labels, legacy / "images/clean", legacy / "images/target",
        args.r3_output.resolve(), source_rows,
        args.replacement_attacks.resolve() if args.replacement_attacks else None,
        {int(value) for value in corrections["r3_retarget"]},
    )
    print(r2)
    print(r3)


if __name__ == "__main__":
    main()
