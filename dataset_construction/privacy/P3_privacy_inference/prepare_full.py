#!/usr/bin/env python3
"""Freeze complete P.3 v2 text and Qwen image jobs without running a model."""

import argparse
import hashlib
import json
from pathlib import Path

from build_p3 import DEFAULT_DEFINITION, build_rows, image_prompt, load_json, validate_definition

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[2] / 'data/privacy/P3_privacy_inference'


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)


def freeze(path, content):
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError(f'Frozen text differs: {path}; use a new version directory')
    path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    output = parser.parse_args().output_dir.resolve()
    definition = load_json(DEFAULT_DEFINITION)
    validate_definition(definition)
    images, instances = build_rows(definition, 'full')
    assert len(images) == 120 and len(instances) == 240
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in images:
        jobs.append({
            **row,
            'render_backend': 'Qwen-Image-2512',
            'seed': int.from_bytes(hashlib.sha256(row['base_id'].encode()).digest()[:4], 'big'),
            'image_prompt': image_prompt({}, {}, {'visual': row['visual_specification']}),
            # Text is intentional in medical records; do not forbid all image text.
            'negative_prompt': 'low quality, blurry, distorted anatomy, watermark, real personal identifiers, unrelated diagnostic clues',
        })
    labels = [{
        'id': row['instance_id'], 'base_id': row['base_id'],
        'image': row['image_filename'], 'question': row['question'],
        'category_id': row['category_id'], 'attribute_id': row['attribute_id'],
        'attribute_name': row['attribute_name'],
        'evidence_condition': row['evidence_condition'],
        'request_context': row['request_context'],
        'expected_behavior': row['expected_behavior'],
        'review_status': 'candidate_not_clinically_validated',
    } for row in instances]
    freeze(output / 'labels.jsonl', jsonl(labels))
    freeze(output / 'image_jobs.jsonl', jsonl(jobs))
    manifest = {
        'task_id': 'P.3', 'version': 'v2-full-qwen2512',
        'status': 'experimental_candidate_text_frozen_images_pending',
        'image_model': 'Qwen-Image-2512', 'base_image_count': 120,
        'question_count': 240, 'category_count': 6, 'attribute_count': 60,
        'per_category_images': 20, 'per_category_questions': 40,
        'source_definition_sha256': hashlib.sha256(DEFAULT_DEFINITION.read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256((output / 'labels.jsonl').read_bytes()).hexdigest(),
        'image_jobs_sha256': hashlib.sha256((output / 'image_jobs.jsonl').read_bytes()).hexdigest(),
        'generation': {'width': 1664, 'height': 928, 'steps': 50, 'true_cfg_scale': 4.0},
        'clinical_review': 'pending', 'judge_enabled': False,
        'old_p3_results_compatible': False,
    }
    freeze(output / 'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
