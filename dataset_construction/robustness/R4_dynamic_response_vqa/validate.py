#!/usr/bin/env python3
"""Validate the frozen R.4 base cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(dataset: Path) -> dict:
    labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    provenance = [json.loads(line) for line in (dataset / "provenance.jsonl").read_text(encoding="utf-8").splitlines() if line]
    errors = []
    if len(labels) != 200: errors.append("wrong label count")
    if len(provenance) != 200: errors.append("wrong provenance count")
    if len({row["id"] for row in labels}) != 200: errors.append("duplicate IDs")
    if len({row["image"] for row in labels}) != 200: errors.append("duplicate images")
    by_id = {row["id"]: row for row in provenance}
    checksums = set()
    for row in labels:
        if list(row["options"]) != list("ABCD") or row["answer"] not in row["options"]:
            errors.append(f"invalid options for ID {row['id']}")
        path = dataset / "images" / row["image"]
        prov = by_id.get(row["id"])
        if not path.is_file() or prov is None:
            errors.append(f"missing image/provenance for ID {row['id']}")
            continue
        digest = sha256(path)
        if digest != prov["sha256"]: errors.append(f"checksum mismatch for ID {row['id']}")
        checksums.add(f"{digest}  images/{row['image']}")
        try:
            with Image.open(path) as image:
                size, fmt = image.size, image.format
                image.verify()
            if size != (prov["width"], prov["height"]) or fmt != prov["format"]:
                errors.append(f"image metadata mismatch for ID {row['id']}")
        except Exception as exc:
            errors.append(f"invalid image for ID {row['id']}: {exc}")
    if checksums != set((dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines()):
        errors.append("checksums file mismatch")
    report = {
        "schema_version": 1, "task_id": "R.4",
        "status": "passed" if not errors else "failed",
        "sample_count": len(labels), "image_count": len(list((dataset / "images").glob("*"))),
        "dynamic_protocol_preserved": manifest.get("dynamic_protocol_preserved"),
        "question_negation_answer_type": manifest.get("question_negation_answer_type"),
        "errors": errors,
    }
    (dataset / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__":
    main()
