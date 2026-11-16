#!/usr/bin/env python3
"""Generate all 120 P.3 images locally with resumable per-image provenance."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parents[2] / 'data/privacy/P3_privacy_inference'
DEFAULT_MODEL = Path(os.environ.get('QWEN_IMAGE_MODEL', 'models/Qwen-Image-2512'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_inputs(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert digest(root / 'labels.jsonl') == manifest['labels_sha256'], 'Labels changed'
    assert digest(root / 'image_jobs.jsonl') == manifest['image_jobs_sha256'], 'Image jobs changed'
    jobs = load_rows(root / 'image_jobs.jsonl')
    labels = load_rows(root / 'labels.jsonl')
    assert len(jobs) == len({j['base_id'] for j in jobs}) == 120
    assert len(labels) == len({r['id'] for r in labels}) == 240
    for job in jobs:
        pair = [row for row in labels if row['base_id'] == job['base_id']]
        assert len(pair) == 2
        assert {row['request_context'] for row in pair} == {'unauthorized_privacy', 'authorized_utility'}
        assert all(row['image'] == job['image_filename'] for row in pair)
    return manifest, jobs


def completed(root, job, manifest):
    image = root / 'images' / job['image_filename']
    record = root / 'image_metadata' / (job['base_id'] + '.json')
    if not image.exists() and not record.exists():
        return False
    if not image.exists() or not record.exists():
        raise ValueError(f'Incomplete image/metadata pair: {job["base_id"]}; inspect before resuming')
    metadata = json.loads(record.read_text())
    job_hash = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    assert metadata['job_sha256'] == job_hash, f'Job changed: {image}'
    assert metadata['generation'] == manifest['generation'], f'Parameters changed: {image}'
    assert metadata['image_sha256'] == digest(image), f'Image changed: {image}'
    with Image.open(image) as im:
        im.load()
        assert im.mode == 'RGB'
        assert im.size == (manifest['generation']['width'], manifest['generation']['height'])
    return True


def validate_images(root):
    manifest, jobs = validate_inputs(root)
    missing = [job['base_id'] for job in jobs if not completed(root, job, manifest)]
    if missing:
        raise ValueError(f'{len(missing)} images still missing: {missing[:5]}')
    hashes = [digest(root / 'images' / job['image_filename']) for job in jobs]
    assert len(set(hashes)) == 120, 'Duplicate generated image bytes'
    report = {
        'status': 'technical_integrity_passed_clinical_review_pending',
        'image_count': 120, 'question_count': 240, 'image_model': 'Qwen-Image-2512',
        'labels_sha256': manifest['labels_sha256'], 'image_jobs_sha256': manifest['image_jobs_sha256'],
        'clinical_review': 'pending', 'judge_enabled': False,
    }
    (root / 'image_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    root = args.data_dir.resolve()
    manifest, jobs = validate_inputs(root)
    if args.validate_only:
        validate_images(root)
        return
    assert (args.model_path / 'model_index.json').is_file(), args.model_path
    if args.dry_run:
        print(json.dumps({'model': str(args.model_path), 'jobs': len(jobs),
                          'questions': 240, 'generation': manifest['generation'],
                          'output': str(root / 'images'), 'model_loaded': False}, indent=2))
        return
    pending = [j for j in jobs if not completed(root, j, manifest)]
    if not pending:
        validate_images(root)
        return

    import torch
    import diffusers
    from diffusers import DiffusionPipeline
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing an unintended CPU image-generation run')
    pipe = DiffusionPipeline.from_pretrained(
        str(args.model_path.resolve()), torch_dtype=torch.bfloat16,
        device_map='balanced', local_files_only=True,
    )
    (root / 'images').mkdir(exist_ok=True)
    (root / 'image_metadata').mkdir(exist_ok=True)
    params = manifest['generation']
    for index, job in enumerate(pending, 1):
        print(f'[{datetime.now().isoformat(timespec="seconds")}] Generating {index}/{len(pending)} {job["base_id"]}', flush=True)
        result = pipe(
            prompt=job['image_prompt'], negative_prompt=job['negative_prompt'],
            width=params['width'], height=params['height'],
            num_inference_steps=params['steps'], true_cfg_scale=params['true_cfg_scale'],
            generator=torch.Generator(device='cuda').manual_seed(job['seed']),
        )
        image = result.images[0].convert('RGB')
        destination = root / 'images' / job['image_filename']
        temporary = destination.with_suffix('.tmp.png')
        image.save(temporary)
        os.replace(temporary, destination)
        metadata = {
            'base_id': job['base_id'], 'model_path': str(args.model_path.resolve()),
            'model_index_sha256': digest(args.model_path / 'model_index.json'),
            'image_sha256': digest(destination),
            'job_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
            'seed': job['seed'], 'generation': params,
            'torch_version': torch.__version__, 'diffusers_version': diffusers.__version__,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'clinical_review': 'pending',
        }
        (root / 'image_metadata' / (job['base_id'] + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    validate_images(root)


if __name__ == '__main__':
    main()
