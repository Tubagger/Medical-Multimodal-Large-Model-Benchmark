#!/usr/bin/env python3
"""Materialize the frozen 243-item P.4 privacy-modified VQA release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED_BASE = 81
EXPECTED_SAMPLES = 243


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_frozen_inputs(source_root: Path, image_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    base = load_json(source_root / "privacy.json")
    expanded = load_json(source_root / "privacy_243.json")
    prompts = load_jsonl(source_root / "privacy_vqa_243_with_prompt.jsonl")
    if len(base) != EXPECTED_BASE or len(expanded) != EXPECTED_SAMPLES or len(prompts) != EXPECTED_SAMPLES:
        raise ValueError("P.4 requires 81 DAS sources and 243 expanded samples/prompts")
    expected_ids = list(range(EXPECTED_SAMPLES))
    if [row.get("id") for row in expanded] != expected_ids or [row.get("id") for row in prompts] != expected_ids:
        raise ValueError("P.4 IDs must be unique and ordered from 0 through 242")
    for index, row in enumerate(base):
        original = dict(expanded[index * 3])
        original["id"] = index
        if original != row:
            raise ValueError(f"DAS source mismatch at source row {index}")
    metadata = ("id", "question", "category", "diagnosis")
    for row, prompt in zip(expanded, prompts):
        if any(row.get(key) != prompt.get(key) for key in metadata):
            raise ValueError(f"prompt metadata mismatch for ID {row['id']}")
        if prompt.get("is_fallback") is not False:
            raise ValueError(f"fallback image prompt for ID {row['id']}")
        if not str(prompt.get("image_prompt", "")).strip() or not str(prompt.get("negative_prompt", "")).strip():
            raise ValueError(f"empty image prompt for ID {row['id']}")
        image = image_dir / f"img_{row['id']}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
    return base, expanded, prompts


def build(source_root: Path, image_dir: Path, output_root: Path) -> Path:
    base, expanded, prompts = validate_frozen_inputs(source_root, image_dir)
    patient_info = load_jsonl(source_root / "patient_info_generate.jsonl")
    phi_step1 = load_jsonl(source_root / "step1_load_or_generate_phi.jsonl")
    phi_step2 = load_jsonl(source_root / "step2_embed_extra_info.jsonl")
    placeholder_ids = {row["id"] for row in expanded if "{GENERATED PHI}" in row["question"]}
    cache_id_sets = [{row["id"] for row in rows} for rows in (patient_info, phi_step1, phi_step2)]
    if len(placeholder_ids) != 33 or any(ids != placeholder_ids for ids in cache_id_sets):
        raise ValueError("P.4 generated-PHI caches do not exactly cover the 33 template rows")
    resolved_phi = {row["id"]: row["revised_phi"] for row in phi_step2}
    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".P4-GMAI-", dir=output_root) as temp_name:
        temp = Path(temp_name)
        images_out = temp / "images"
        images_out.mkdir()
        labels, provenance, checksums = [], [], []
        prompt_by_id = {row["id"]: row for row in prompts}
        for row in expanded:
            sample_id = row["id"]
            image_name = f"img_{sample_id}.png"
            source_image = image_dir / image_name
            destination_image = images_out / image_name
            shutil.copyfile(source_image, destination_image)
            with Image.open(destination_image) as image:
                image.load()
                width, height = image.size
                image_format, image_mode = image.format, image.mode
            digest = sha256(destination_image)
            prompt = prompt_by_id[sample_id]
            label = {
                "id": sample_id,
                "image": image_name,
                "question": row["question"],
                "category": row["category"],
                "diagnosis": row.get("diagnosis"),
                "resolved_question": row["question"].replace(
                    "{GENERATED PHI}", resolved_phi.get(sample_id, "")
                ),
            }
            labels.append(label)
            provenance.append({
                "id": sample_id,
                "das_source_index": sample_id // 3,
                "variant": ("das_original", "qwen3_expansion_1", "qwen3_expansion_2")[sample_id % 3],
                "image": image_name,
                "category": row["category"],
                "diagnosis": row.get("diagnosis"),
                "image_prompt": prompt["image_prompt"],
                "negative_prompt": prompt["negative_prompt"],
                "image_prompt_sha256": hashlib.sha256(prompt["image_prompt"].encode()).hexdigest(),
                "resolved_question_sha256": hashlib.sha256(label["resolved_question"].encode()).hexdigest(),
                "generated_phi_cache_used": sample_id in placeholder_ids,
                "image_sha256": digest,
                "format": image_format,
                "mode": image_mode,
                "width": width,
                "height": height,
            })
            checksums.append(f"{digest}  images/{image_name}")

        manifest = {
            "schema_version": 1,
            "task_id": "P.4",
            "name": "privacy_modified_medical_vqa",
            "status": "passed_with_observations",
            "base_das_samples": len(base),
            "expanded_samples": len(labels),
            "image_count": len(labels),
            "expansion_model": "Qwen3-32B",
            "image_model": "Qwen-Image-2512",
            "image_seed_rule": "seed equals sample ID",
            "image_size": [1024, 1024],
            "image_steps": 50,
            "true_cfg_scale": 4.0,
            "generated_phi_template_count": len(placeholder_ids),
            "category_counts": dict(sorted(Counter(row["category"] for row in labels).items())),
            "historical_runtime_compatible": True,
            "requires_model_rerun": False,
            "source_artifacts": {
                name: sha256(source_root / name)
                for name in (
                    "DAS_Privacy.xlsx", "privacy.json", "privacy_243.json",
                    "privacy_vqa_243_with_prompt.jsonl", "patient_info_generate.jsonl",
                    "step1_load_or_generate_phi.jsonl", "step2_embed_extra_info.jsonl",
                )
            },
        }
        (temp / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp / "image_prompts.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts), encoding="utf-8"
        )
        (temp / "provenance.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in provenance), encoding="utf-8"
        )
        (temp / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source_root.resolve(), args.image_dir.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()

