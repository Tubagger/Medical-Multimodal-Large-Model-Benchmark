#!/usr/bin/env python3
"""Validate corrected R.2/R.3 artifacts and write validation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(dataset: Path) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    provenance = [json.loads(line) for line in (dataset / "provenance.jsonl").read_text(encoding="utf-8").splitlines() if line]
    errors = []
    pending = [int(value) for value in manifest.get("missing_regenerated_attack_ids", [])]
    if len(labels) != 200: errors.append("label count is not 200")
    if len(provenance) != 200: errors.append("provenance count is not 200")
    if len({int(row["id"]) for row in labels}) != 200: errors.append("duplicate IDs")
    by_id = {int(row["id"]): row for row in provenance}
    for row in labels:
        sample_id = int(row["id"])
        options = dict(re.findall(r"(?m)^([A-E])\.\s*([^\n]+)$", row["options"]))
        if len(options) != 5 or len(set(options.values())) != 5:
            errors.append(f"malformed or duplicate options for ID {sample_id}")
            continue
        if options.get(row.get("ground_truth_answer")) != row.get("ground_truth_label"):
            errors.append(f"ground-truth mapping mismatch for ID {sample_id}")
        if "isic" in options.values() or "mura" in options.values():
            errors.append(f"legacy source name remains in ID {sample_id}")
        if manifest["task_id"] == "R.3":
            if options.get(row.get("attack_target_answer")) != row.get("attack_target_label"):
                errors.append(f"attack-target mapping mismatch for ID {sample_id}")
            if row.get("ground_truth_answer") == row.get("attack_target_answer"):
                errors.append(f"ground truth equals attack target for ID {sample_id}")
        prov = by_id.get(sample_id)
        if prov is None:
            errors.append(f"missing provenance for ID {sample_id}")
            continue
        for condition, hash_key in (("clean", "clean_sha256"), ("attacked", "attack_sha256")):
            path = dataset / condition / "images" / row["image"]
            expected = prov.get(hash_key)
            if sample_id in pending and condition == "attacked":
                if path.exists() or expected is not None:
                    errors.append(f"pending attack unexpectedly materialized for ID {sample_id}")
                continue
            if not path.is_file():
                errors.append(f"missing {condition} image for ID {sample_id}")
                continue
            if sha256(path) != expected:
                errors.append(f"checksum mismatch for {condition} ID {sample_id}")
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"invalid image for {condition} ID {sample_id}: {exc}")
    status = "failed" if errors else ("pending_attack_regeneration" if pending else "passed")
    report = {
        "schema_version": 1,
        "task_id": manifest["task_id"],
        "status": status,
        "sample_count": len(labels),
        "clean_image_count": len(list((dataset / "clean/images").glob("*"))),
        "attack_image_count": len(list((dataset / "attacked/images").glob("*"))),
        "semantic_corrections": manifest["semantic_corrections"],
        "model_performance_filtering": False,
        "requires_model_rerun": True,
        "pending_attack_ids": pending,
        "errors": errors,
    }
    (dataset / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    args = parser.parse_args()
    failed = False
    for dataset in args.datasets:
        report = validate(dataset.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        failed |= report["status"] == "failed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
