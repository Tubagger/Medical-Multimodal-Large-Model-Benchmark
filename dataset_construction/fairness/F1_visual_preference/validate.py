#!/usr/bin/env python3
"""Validate the structure and provenance of the F.1 v2 candidate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from build import PROMPT


EXPECTED_GROUPS = {"brain", "isic", "lung", "mura", "retina"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(dataset_dir: Path, pairing_path: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((dataset_dir / "labels.json").read_text(encoding="utf-8"))
    provenance = [
        json.loads(line)
        for line in (dataset_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    embedded_pairing = dataset_dir / "pairing_manifest.json"
    check(embedded_pairing.is_file(), "missing embedded pairing manifest", errors)
    if embedded_pairing.is_file():
        check(sha256(embedded_pairing) == sha256(pairing_path), "embedded pairing manifest mismatch", errors)
    check(manifest.get("status") == "passed", "wrong status", errors)
    check(manifest.get("review_required") is False, "unexpected review requirement", errors)
    check(manifest.get("clinical_ground_truth_required") is False, "wrong clinical-ground-truth policy", errors)
    check(len(pairing.get("pairs", [])) == 100, "pairing count is not 100", errors)
    check(len(labels) == 200, "sample count is not 200", errors)
    check(len(provenance) == 200, "provenance count is not 200", errors)
    check(len({row.get("id") for row in labels}) == 200, "duplicate sample IDs", errors)
    check(all(row.get("answer") == "C" for row in labels), "non-C answer found", errors)
    check(all(row.get("question") == PROMPT for row in labels), "prompt mismatch", errors)

    pairing_by_id = {row["pair_id"]: row for row in pairing["pairs"]}
    labels_by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        labels_by_pair[row.get("pair_id")].append(row)
    check(set(labels_by_pair) == set(pairing_by_id), "label/pair manifest mismatch", errors)
    used_source_ids = []
    for pair_id, pair in pairing_by_id.items():
        check(pair["source_a_id"] != pair["source_b_id"], f"same source in {pair_id}", errors)
        used_source_ids.extend([pair["source_a_id"], pair["source_b_id"]])
        rows = sorted(labels_by_pair.get(pair_id, []), key=lambda row: row.get("order", ""))
        check(len(rows) == 2, f"{pair_id} does not have two orders", errors)
        if len(rows) == 2:
            ab, ba = rows
            check(ab.get("order") == "ab" and ba.get("order") == "ba", f"wrong orders for {pair_id}", errors)
            check(
                (ab.get("source_left_id"), ab.get("source_right_id"))
                == (pair["source_a_id"], pair["source_b_id"]),
                f"AB mapping mismatch for {pair_id}", errors,
            )
            check(
                (ba.get("source_left_id"), ba.get("source_right_id"))
                == (pair["source_b_id"], pair["source_a_id"]),
                f"BA mapping mismatch for {pair_id}", errors,
            )
    check(len(used_source_ids) == len(set(used_source_ids)) == 200, "source images are reused", errors)
    check(set(row.get("group") for row in labels) == EXPECTED_GROUPS, "source group mismatch", errors)
    group_counts = Counter(row.get("group") for row in labels)
    check(all(group_counts[group] == 40 for group in EXPECTED_GROUPS), "group imbalance", errors)

    prov_by_id = {row["id"]: row for row in provenance}
    expected_checksums = set()
    image_hashes = []
    for row in labels:
        sample_id = row["id"]
        prov = prov_by_id.get(sample_id)
        check(prov is not None, f"missing provenance for {sample_id}", errors)
        image_path = dataset_dir / "images" / row["image"]
        check(image_path.is_file(), f"missing image for {sample_id}", errors)
        if not image_path.is_file() or prov is None:
            continue
        try:
            with Image.open(image_path) as image:
                check(image.format == "PNG", f"wrong image format for {sample_id}", errors)
                check(image.height == 592, f"wrong image height for {sample_id}", errors)
                image.verify()
        except Exception as exc:
            errors.append(f"invalid image for {sample_id}: {exc}")
        digest = sha256(image_path)
        image_hashes.append(digest)
        check(digest == prov.get("composite_sha256"), f"image checksum mismatch for {sample_id}", errors)
        expected_checksums.add(f"{digest}  images/{row['image']}")
    check(len(set(image_hashes)) == 200, "duplicate rendered composites", errors)
    actual_checksums = set((dataset_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    check(actual_checksums == expected_checksums, "checksums.sha256 mismatch", errors)
    image_files = list((dataset_dir / "images").glob("*.png"))
    check(len(image_files) == 200, "unexpected image count", errors)

    report = {
        "schema_version": 2,
        "task_id": "F.1-v2",
        "status": "passed" if not errors else "failed",
        "base_pair_count": len(pairing_by_id),
        "sample_count": len(labels),
        "source_image_count": len(set(used_source_ids)),
        "group_sample_counts": dict(sorted(group_counts.items())),
        "clinical_ground_truth_required": False,
        "review_required": False,
        "errors": errors,
    }
    (dataset_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--pairing", type=Path, default=Path(__file__).with_name("pairing_manifest.json"))
    args = parser.parse_args()
    report = validate(args.dataset_dir.resolve(), args.pairing.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
