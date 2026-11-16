#!/usr/bin/env python3
"""Validate provenance, structure, and frozen assets for P.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import openpyxl
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category(value: str) -> str:
    value = value.replace("\u202f", " ").replace("‑", "-")
    return " ".join(value.casefold().split())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(dataset: Path, construction_source: Path, upstream_images: Path) -> dict:
    labels = json.loads((dataset / "labels.json").read_text(encoding="utf-8"))
    base = json.loads((construction_source / "privacy.json").read_text(encoding="utf-8"))
    expanded = json.loads((construction_source / "privacy_243.json").read_text(encoding="utf-8"))
    prompts = load_jsonl(construction_source / "privacy_vqa_243_with_prompt.jsonl")
    patient_info = load_jsonl(construction_source / "patient_info_generate.jsonl")
    phi_step1 = load_jsonl(construction_source / "step1_load_or_generate_phi.jsonl")
    phi_step2 = load_jsonl(construction_source / "step2_embed_extra_info.jsonl")
    provenance = load_jsonl(dataset / "provenance.jsonl")
    errors: list[str] = []
    checks: dict[str, str] = {}

    def check(condition: bool, name: str, detail: str = "") -> None:
        checks[name] = "passed" if condition else "failed"
        if not condition:
            errors.append(f"{name}: {detail}" if detail else name)

    workbook = openpyxl.load_workbook(construction_source / "DAS_Privacy.xlsx", read_only=True, data_only=True)
    sheet = workbook["Privacy"]
    excel_rows = [
        {"question": row[0], "category": row[1], "diagnosis": row[2]}
        for row in sheet.iter_rows(min_row=2, values_only=True)
    ]
    check(len(excel_rows) == len(base) == 81, "das_source_count")
    check(all(
        source.get("question") == excel.get("question")
        and source.get("category") == excel.get("category")
        and source.get("diagnosis") == excel.get("diagnosis")
        for source, excel in zip(base, excel_rows)
    ), "das_excel_json_exact_match")
    check(len(labels) == len(expanded) == len(prompts) == len(provenance) == 243, "expanded_sample_count")
    check([row.get("id") for row in labels] == list(range(243)), "ordered_unique_ids")
    check(all(
        {key: label.get(key) for key in ("id", "question", "category", "diagnosis")}
        == {key: row.get(key) for key in ("id", "question", "category", "diagnosis")}
        for label, row in zip(labels, expanded)
    ), "expanded_label_exact_match")
    placeholder_ids = {row["id"] for row in expanded if "{GENERATED PHI}" in row["question"]}
    cache_id_sets = [{row["id"] for row in rows} for rows in (patient_info, phi_step1, phi_step2)]
    check(len(placeholder_ids) == 33 and all(ids == placeholder_ids for ids in cache_id_sets),
          "generated_phi_cache_coverage")
    resolved_phi = {row["id"]: row["revised_phi"] for row in phi_step2}
    check(all(
        label.get("resolved_question")
        == row["question"].replace("{GENERATED PHI}", resolved_phi.get(row["id"], ""))
        and "{GENERATED PHI}" not in label.get("resolved_question", "")
        for label, row in zip(labels, expanded)
    ), "resolved_runtime_questions")
    check(all(prompt.get("is_fallback") is False for prompt in prompts), "no_fallback_prompts")
    check(all(str(prompt.get("image_prompt", "")).strip() and str(prompt.get("negative_prompt", "")).strip()
              for prompt in prompts), "nonempty_image_prompts")
    check(all(
        all(prompt.get(key) == row.get(key) for key in ("id", "question", "category", "diagnosis"))
        for prompt, row in zip(prompts, expanded)
    ), "prompt_metadata_exact_match")

    exact_category_triplets = normalized_category_triplets = 0
    for source_index, source in enumerate(base):
        triplet = expanded[source_index * 3:source_index * 3 + 3]
        exact_category_triplets += len({row["category"] for row in triplet}) == 1
        normalized_category_triplets += len({normalize_category(row["category"]) for row in triplet}) == 1
        original = dict(triplet[0])
        original["id"] = source_index
        if original != source:
            errors.append(f"DAS original mismatch at source index {source_index}")
    check(exact_category_triplets == 78, "recorded_typographic_category_variants", f"{exact_category_triplets}/81")
    check(normalized_category_triplets == 81, "normalized_category_preservation")

    checksum_lines = set((dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines())
    observed_lines = set()
    byte_matches = valid_images = 0
    for label, record in zip(labels, provenance):
        image_name = label["image"]
        released = dataset / "images" / image_name
        upstream = upstream_images / image_name
        if not released.is_file() or not upstream.is_file():
            errors.append(f"missing image {image_name}")
            continue
        digest = sha256(released)
        observed_lines.add(f"{digest}  images/{image_name}")
        byte_matches += digest == sha256(upstream)
        try:
            with Image.open(released) as image:
                image.load()
                valid_images += image.format == "PNG" and image.mode == "RGB" and image.size == (1024, 1024)
        except Exception as exc:
            errors.append(f"invalid image {image_name}: {exc}")
        if record.get("id") != label["id"] or record.get("image_sha256") != digest:
            errors.append(f"provenance mismatch for ID {label['id']}")
    check(valid_images == 243, "valid_png_rgb_1024_images", f"{valid_images}/243")
    check(byte_matches == 243, "upstream_image_byte_match", f"{byte_matches}/243")
    check(observed_lines == checksum_lines, "checksum_manifest")
    check(len(list((dataset / "images").glob("*.png"))) == 243, "image_file_count")

    report = {
        "schema_version": 1,
        "task_id": "P.4",
        "status": "passed_with_observations" if not errors else "failed",
        "base_das_samples": len(base),
        "expanded_samples": len(labels),
        "image_count": valid_images,
        "prompt_count": len(prompts),
        "fallback_prompt_count": sum(prompt.get("is_fallback") is not False for prompt in prompts),
        "generated_phi_template_count": len(placeholder_ids),
        "normalized_category_counts": dict(sorted(Counter(normalize_category(row["category"]) for row in labels).items())),
        "checks": checks,
        "accepted_observations": {
            "triplets_with_typographic_category_variants": 3,
            "semantic_image_alignment_review": "out_of_scope_for_construction_code_audit",
            "legacy_fallback_resume_bug_affected_frozen_release": False,
        },
        "errors": errors,
    }
    (dataset / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

