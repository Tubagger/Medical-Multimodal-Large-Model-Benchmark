#!/usr/bin/env python3
"""Materialize the frozen 210-item S.1 medical-risk request release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED_SAMPLES = 210
TARGET_CATEGORIES = (
    "Corruption and Conflicts of Interest",
    "System Manipulation and Fraud",
    "Prejudice, Discrimination, and Hate Speech",
    "Harassment, Violence, and Abuse",
    "Misinformation and Public Health Sabotage",
    "Unethical and Inappropriate Medical Requests",
    "Data Privacy Violations and Security Exploits",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_inputs(source: Path, images: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    raw = load_jsonl(source / "balanced_210.jsonl")
    translated = load_jsonl(source / "balanced_210_en.jsonl")
    prompts = load_jsonl(source / "balanced_210_with_prompt.jsonl")
    final = load_jsonl(source / "safetyrisk_vqa.jsonl")
    if not all(len(rows) == EXPECTED_SAMPLES for rows in (raw, translated, prompts, final)):
        raise ValueError("S.1 requires 210 rows at every frozen construction stage")
    if Counter(row["malicious_category"] for row in raw) != Counter({key: 30 for key in TARGET_CATEGORIES}):
        raise ValueError("S.1 requires exactly 30 samples from each of seven risk categories")
    if [row.get("id") for row in prompts] != [str(i) for i in range(EXPECTED_SAMPLES)]:
        raise ValueError("image-prompt IDs must be ordered strings 0 through 209")
    if [row.get("id") for row in final] != list(range(EXPECTED_SAMPLES)):
        raise ValueError("final IDs must be ordered integers 0 through 209")
    for index, (source_row, english, prompt, released) in enumerate(zip(raw, translated, prompts, final)):
        for field in ("persona", "malicious_category", "rationale"):
            if source_row.get(field) != english.get(field):
                raise ValueError(f"translation changed {field} at row {index}")
        for field in ("persona", "malicious_category", "malicious_prompt", "rationale"):
            if english.get(field) != prompt.get(field):
                raise ValueError(f"prompt metadata mismatch at row {index}: {field}")
        for field in ("persona", "malicious_prompt", "rationale"):
            if prompt.get(field) != released.get(field):
                raise ValueError(f"final metadata mismatch at row {index}: {field}")
        if prompt.get("is_fallback") is not False:
            raise ValueError(f"fallback image prompt at row {index}")
        if not str(prompt.get("image_prompt", "")).strip() or not str(prompt.get("negative_prompt", "")).strip():
            raise ValueError(f"empty image prompt at row {index}")
        expected_image = f"img_{index}.png"
        if released.get("image") != expected_image or not (images / expected_image).is_file():
            raise ValueError(f"missing or misnamed image at row {index}")
    return raw, translated, prompts, final


def build(source: Path, images: Path, output_root: Path) -> Path:
    raw, translated, prompts, final = validate_inputs(source, images)
    destination = output_root / "GMAI"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".S1-GMAI-", dir=output_root) as temp_name:
        temp = Path(temp_name)
        image_output = temp / "images"
        image_output.mkdir()
        labels, prompt_rows, provenance, checksums = [], [], [], []
        for index, (original, english, prompt, released) in enumerate(zip(raw, translated, prompts, final)):
            image_name = released["image"]
            source_image = images / image_name
            target_image = image_output / image_name
            shutil.copyfile(source_image, target_image)
            with Image.open(target_image) as image:
                image.load()
                size, mode, image_format = image.size, image.mode, image.format
            digest = sha256(target_image)
            label = {
                "id": index,
                "image": image_name,
                "malicious_category": prompt["malicious_category"],
                "malicious_prompt": english["malicious_prompt"],
                "persona": english["persona"],
                "rationale": english["rationale"],
            }
            labels.append(label)
            prompt_rows.append({
                "id": index,
                "image_prompt": prompt["image_prompt"],
                "negative_prompt": prompt["negative_prompt"],
                "is_fallback": prompt["is_fallback"],
            })
            provenance.append({
                "id": index,
                "source_row_after_seeded_sampling": index,
                "category": label["malicious_category"],
                "original_prompt_sha256": hashlib.sha256(original["malicious_prompt"].encode()).hexdigest(),
                "english_prompt_sha256": hashlib.sha256(label["malicious_prompt"].encode()).hexdigest(),
                "image_prompt_sha256": hashlib.sha256(prompt["image_prompt"].encode()).hexdigest(),
                "image_sha256": digest,
                "image_format": image_format,
                "image_mode": mode,
                "image_size": list(size),
            })
            checksums.append(f"{digest}  images/{image_name}")

        manifest = {
            "schema_version": 1,
            "task_id": "S.1",
            "name": "medical_risk_requests",
            "status": "passed_with_observations",
            "source_dataset": "Medical Malice frozen parquet",
            "source_sample_count": 214219,
            "selection_pool_count": 213553,
            "sample_count": EXPECTED_SAMPLES,
            "category_count": len(TARGET_CATEGORIES),
            "samples_per_category": 30,
            "sampling_seed": 42,
            "translation_model": "Qwen3-32B",
            "image_prompt_model": "Qwen3-32B",
            "image_model": "Qwen-Image-2512",
            "image_seed_rule": "seed equals sequential sample ID",
            "image_size": [1664, 928],
            "image_steps": 50,
            "true_cfg_scale": 4.0,
            "historical_runtime_compatible": True,
            "requires_model_rerun": False,
            "category_counts": dict(sorted(Counter(row["malicious_category"] for row in labels).items())),
            "source_artifacts": {
                name: sha256(source / name)
                for name in (
                    "train-00000-of-00001.parquet", "balanced_210.jsonl",
                    "balanced_210_en.jsonl", "balanced_210_with_prompt.jsonl",
                    "safetyrisk_vqa.jsonl",
                )
            },
        }
        (temp / "labels.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8"
        )
        (temp / "image_prompts.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_rows), encoding="utf-8"
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
