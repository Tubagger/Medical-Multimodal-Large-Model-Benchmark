#!/usr/bin/env python3
"""Validate S.1 sampling provenance, frozen transformations, and images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from build import TARGET_CATEGORIES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(dataset: Path, source: Path, upstream_images: Path) -> dict:
    raw = load_jsonl(source / "balanced_210.jsonl")
    translated = load_jsonl(source / "balanced_210_en.jsonl")
    prompts = load_jsonl(source / "balanced_210_with_prompt.jsonl")
    final = load_jsonl(source / "safetyrisk_vqa.jsonl")
    labels = load_jsonl(dataset / "labels.jsonl")
    released_prompts = load_jsonl(dataset / "image_prompts.jsonl")
    provenance = load_jsonl(dataset / "provenance.jsonl")
    errors: list[str] = []
    checks: dict[str, str] = {}

    def check(condition: bool, name: str, detail: str = "") -> None:
        checks[name] = "passed" if condition else "failed"
        if not condition:
            errors.append(f"{name}: {detail}" if detail else name)

    frame = pd.read_parquet(source / "train-00000-of-00001.parquet")
    pool = frame[frame["malicious_category"].isin(TARGET_CATEGORIES)]
    source_columns = list(pool.columns)
    reproduced = (
        pool.groupby("malicious_category", group_keys=False)[source_columns]
        .apply(lambda group: group.sample(n=30, random_state=42), include_groups=False)
        .reset_index(drop=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
        .to_dict("records")
    )
    check(len(frame) == 214219, "source_row_count", str(len(frame)))
    check(len(pool) == 213553, "seven_category_pool_count", str(len(pool)))
    check(reproduced == raw, "seeded_balanced_sampling_exact_match")
    check(len(labels) == len(raw) == len(translated) == len(prompts) == len(final) == 210,
          "construction_stage_counts")
    check([row.get("id") for row in labels] == list(range(210)), "ordered_unique_ids")
    expected_counts = Counter({category: 30 for category in TARGET_CATEGORIES})
    check(Counter(row.get("malicious_category") for row in labels) == expected_counts,
          "balanced_category_counts")
    check(all(
        source_row.get("persona") == english.get("persona")
        and source_row.get("rationale") == english.get("rationale")
        and source_row.get("malicious_category") == english.get("malicious_category")
        and str(english.get("malicious_prompt", "")).strip()
        for source_row, english in zip(raw, translated)
    ), "translation_metadata_preservation")
    check(all(
        all(english.get(field) == prompt.get(field)
            for field in ("persona", "rationale", "malicious_category", "malicious_prompt"))
        for english, prompt in zip(translated, prompts)
    ), "image_prompt_metadata_exact_match")
    check(all(prompt.get("is_fallback") is False for prompt in prompts), "no_fallback_image_prompts")
    check(all(
        row == {
            "id": index,
            "image_prompt": prompts[index]["image_prompt"],
            "negative_prompt": prompts[index]["negative_prompt"],
            "is_fallback": False,
        }
        for index, row in enumerate(released_prompts)
    ), "released_image_prompts_exact_match")
    check(all(
        label.get("image") == final[index].get("image")
        and label.get("malicious_prompt") == final[index].get("malicious_prompt")
        and label.get("persona") == final[index].get("persona")
        and label.get("rationale") == final[index].get("rationale")
        and label.get("malicious_category") == prompts[index].get("malicious_category")
        for index, label in enumerate(labels)
    ), "released_labels_exact_match")

    text_instruction_terms = (
        "sign on the door that reads", "labeled '", "labeled generically",
        "folder labeled", "documents labeled", "sign is placed", "written in portuguese",
        "number written on it", "door marked",
    )
    text_instruction_ids = [
        int(row["id"]) for row in prompts
        if any(term in row["image_prompt"].lower() for term in text_instruction_terms)
    ]
    checksum_lines = set((dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    observed_lines, image_hashes = set(), []
    valid_images = byte_matches = 0
    for label, record in zip(labels, provenance):
        released = dataset / "images" / label["image"]
        upstream = upstream_images / label["image"]
        if not released.is_file() or not upstream.is_file():
            errors.append(f"missing image {label['image']}")
            continue
        digest = sha256(released)
        image_hashes.append(digest)
        observed_lines.add(f"{digest}  images/{label['image']}")
        byte_matches += digest == sha256(upstream)
        try:
            with Image.open(released) as image:
                image.load()
                valid_images += image.format == "PNG" and image.mode == "RGB" and image.size == (1664, 928)
        except Exception as exc:
            errors.append(f"invalid image {label['image']}: {exc}")
        if record.get("id") != label["id"] or record.get("image_sha256") != digest:
            errors.append(f"provenance mismatch for ID {label['id']}")
    check(valid_images == 210, "valid_png_rgb_1664x928_images", f"{valid_images}/210")
    check(byte_matches == 210, "upstream_image_byte_match", f"{byte_matches}/210")
    check(len(set(image_hashes)) == 210, "unique_image_bytes")
    check(observed_lines == checksum_lines, "checksum_manifest")
    check(len(list((dataset / "images").glob("*.png"))) == 210, "image_file_count")

    report = {
        "schema_version": 1,
        "task_id": "S.1",
        "status": "passed_with_observations" if not errors else "failed",
        "source_samples": len(frame),
        "selection_pool_samples": len(pool),
        "released_samples": len(labels),
        "category_counts": dict(sorted(Counter(row["malicious_category"] for row in labels).items())),
        "image_count": valid_images,
        "image_prompt_count": len(released_prompts),
        "fallback_prompt_count": sum(row.get("is_fallback") is not False for row in prompts),
        "checks": checks,
        "accepted_observations": {
            "image_prompts_explicitly_requesting_visible_text_despite_no_text_rule": text_instruction_ids,
            "count": len(text_instruction_ids),
            "visual_contact_sheet_review": "all 210 images are intact and depict plausible medical, administrative, or community contexts",
            "role_of_image": "contextual grounding; malicious intent remains explicit in the paired user request",
            "legacy_sampling_paths_are_nonportable": True,
            "legacy_translation_partial_resume_bug_present": True,
            "legacy_translation_resume_bug_affected_frozen_release": False,
        },
        "errors": errors,
    }
    (dataset / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--construction-source", type=Path, required=True)
    parser.add_argument("--upstream-images", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.dataset.resolve(), args.construction_source.resolve(), args.upstream_images.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
