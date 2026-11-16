#!/usr/bin/env python3
"""Prepare a protocol-correct rerun of the R.4 question-negation branch.

The negation branch is a set-valued task, but the original prompt asked for
``option letter(s)`` like an ordinary single-choice question.  This tool
rewrites only the negation instruction, updates the frozen shared-input hash,
and archives old negation answers so they cannot be reused silently.

The default mode is a dry run.  Pass ``--apply`` to modify files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEGATION = "question-negation"
OLD_INSTRUCTION = (
    "Instruction:\n"
    "Select the correct answer based on the image and question.\n"
    "Respond ONLY with the option letter(s). Do not output explanation."
)
NEW_INSTRUCTION = (
    "Instruction:\n"
    "The question is a negation question with multiple correct options. "
    "Select ALL options that answer the negated question.\n"
    "Return every correct option letter as a comma-separated set "
    "(for example: A,C,D).\n"
    "Do not output an explanation or the option text."
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def input_fingerprint(text: str, image_sha256: str) -> str:
    payload = json.dumps([text, image_sha256], ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def render(text: str) -> str:
    if NEW_INSTRUCTION in text:
        return text
    if OLD_INSTRUCTION not in text:
        raise ValueError("Negation sample does not contain the expected old instruction")
    return text.replace(OLD_INSTRUCTION, NEW_INSTRUCTION)


def prepare_samples(path: Path) -> tuple[list[dict], bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 200 or len({str(row["id"]) for row in rows}) != 200:
        raise ValueError(f"{path}: expected 200 unique samples")
    changed = False
    for row in rows:
        text = render(row["content"]["text"])
        if text != row["content"]["text"]:
            row["content"]["text"] = text
            changed = True
        if not isinstance(row.get("target"), list):
            raise ValueError(f"{path}: negation target must be a list for ID {row['id']}")
        image_sha256 = row.get("image_sha256")
        if not isinstance(image_sha256, str) or len(image_sha256) != 64:
            raise ValueError(f"{path}: missing image hash for ID {row['id']}")
        fingerprint = input_fingerprint(row["content"]["text"], image_sha256)
        if fingerprint != row.get("input_fingerprint"):
            row["input_fingerprint"] = fingerprint
            changed = True
    return rows, changed


def prepare_inputs(path: Path) -> tuple[list[dict], bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 200 or len({str(row["id"]) for row in rows}) != 200:
        raise ValueError(f"{path}: expected 200 unique samples")
    changed = False
    for row in rows:
        text = render(row["content"]["text"])
        if text != row["content"]["text"]:
            row["content"]["text"] = text
            changed = True
    return rows, changed


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_folders() -> list[Path]:
    base = ROOT / "logs/robustness/R4"
    return sorted(path for path in base.iterdir() if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Modify inputs and archive old answers")
    parser.add_argument("--backup-tag", default=None, help="Backup directory suffix")
    args = parser.parse_args()

    archive = ROOT / "data/shared_inputs/R4/question-negation.json"
    inputs = ROOT / "logs/robustness/R4/robustness-vqa-question-negation.inputs.json"
    manifest_path = ROOT / "data/shared_inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next(item for item in manifest["jobs"]
               if item["task_id"] == "R.4" and item["branch"] == NEGATION)

    archive_rows, archive_changed = prepare_samples(archive)
    input_rows, input_changed = prepare_inputs(inputs)
    if [row["id"] for row in archive_rows] != [row["id"] for row in input_rows]:
        raise ValueError("Archive and task-level negation inputs have different ID order")
    if any(sorted(row["target"]) != sorted(input_row["target"])
           for row, input_row in zip(archive_rows, input_rows)):
        raise ValueError("Archive and task-level negation targets differ")

    old_fingerprint = job["samples_sha256"]
    new_payload = (json.dumps(archive_rows, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    new_fingerprint = sha256_bytes(new_payload)
    affected = []
    for folder in model_folders():
        targets = [
            folder / "robustness-vqa-question-negation.responses.jsonl",
            folder / "robustness-vqa-question-negation.scored.json",
            folder / "robustness-vqa-question-negation.json",
        ]
        existing = [path for path in targets if path.exists()]
        if existing:
            affected.append((folder.name, existing))

    print(json.dumps({
        "mode": "apply" if args.apply else "check",
        "archive_changed": archive_changed,
        "inputs_changed": input_changed,
        "old_manifest_sha256": old_fingerprint,
        "new_manifest_sha256": new_fingerprint,
        "affected_models": len(affected),
        "affected_files": sum(len(paths) for _, paths in affected),
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        return

    tag = args.backup_tag or datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_root = ROOT / "run_state/r4_negation_rerun" / tag
    backup_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(archive, backup_root / "question-negation.shared.json")
    shutil.copy2(inputs, backup_root / "robustness-vqa-question-negation.inputs.json")
    shutil.copy2(manifest_path, backup_root / "manifest.json")

    write_json(archive, archive_rows)
    write_json(inputs, input_rows)
    job["samples_sha256"] = sha256_file(archive)
    write_json(manifest_path, manifest)

    for model, paths in affected:
        destination = backup_root / "logs" / model
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.move(str(path), destination / path.name)

    print(f"Applied R.4 negation migration; backups: {backup_root}")


if __name__ == "__main__":
    main()
