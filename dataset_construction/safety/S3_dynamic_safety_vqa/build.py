#!/usr/bin/env python3
"""Freeze existing S.3 base inputs and rewrites without generating model output."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = int(HERE.parents[1].name.rsplit('_v', 1)[1])
DEFAULT_OUTPUT = ROOT / f'data_v{VERSION}/safety/S3_dynamic_safety_vqa/GMAI'
BRANCHES = ('risk-minimization', 'authority', 'urgency',
            'consequence-obfuscation', 'diagnostic-anchoring')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    labels = rows(dataset / 'labels.jsonl')
    if len(labels) != 210 or {str(r['id']) for r in labels} != {str(i) for i in range(210)}:
        raise ValueError('S.3 requires the frozen 210 unique sample IDs')
    if digest(dataset / 'labels.jsonl') != manifest['source_annotation_sha256']:
        raise ValueError('S.3 source annotations changed')
    provenance = {str(r['id']): r for r in rows(dataset / 'provenance.jsonl')}
    checksums = []
    for row in labels:
        image = dataset / 'images' / row['image']
        expected = provenance[str(row['id'])]['image_sha256']
        if digest(image) != expected:
            raise ValueError(f'Image changed: {image}')
        with Image.open(image) as im:
            im.verify()
        checksums.append(f'{expected}  images/{row["image"]}')
    if len({r['image'] for r in labels}) != 210 or len(list((dataset / 'images').iterdir())) != 210:
        raise ValueError('S.3 image coverage differs')
    if checksums != (dataset / 'checksums.sha256').read_text().splitlines():
        raise ValueError('S.3 checksums differ')
    cache_files = list((dataset / 'rewrites').glob('*/*.jsonl')) + list((dataset / 'shared_rewrites').glob('*.jsonl'))
    rewrite_errors = 0
    for path in cache_files:
        cached = rows(path)
        if len(cached) != 210 or {str(r['id']) for r in cached} != {str(i) for i in range(210)}:
            raise ValueError(f'Rewrite sample coverage differs: {path}')
        for row in cached:
            if not isinstance(row.get('rewritten'), str):
                raise ValueError(f'Missing rewrite text: {path}/{row["id"]}')
            rewrite_errors += not row['rewritten'].strip() or row['rewritten'].lstrip().startswith('Error in generation:')
        if digest(path) != manifest['rewrite_checksums'][str(path.relative_to(dataset))]:
            raise ValueError(f'Rewrite cache changed: {path}')
    if set(manifest['rewrite_checksums']) != {str(p.relative_to(dataset)) for p in cache_files}:
        raise ValueError('Rewrite manifest coverage differs')
    report = {'task_id': 'S.3', 'experiment_version': VERSION,
              'status': ('technical_integrity_passed_with_historical_rewrite_errors' if rewrite_errors else 'technical_integrity_passed'), 'sample_count': 210, 'image_count': 210,
              'rewrite_cache_files': len(cache_files), 'rewrite_rows': 210 * len(cache_files),
              'historical_empty_or_error_rewrites': rewrite_errors,
              'protocol': manifest['protocol'],
              'shared_rewrites_frozen': manifest['shared_rewrites_frozen'],
              'clinical_review': 'no_new_clinical_review_claimed', 'errors': []}
    write_json(dataset / 'validation_report.json', report)
    return report


def build(source, cache_root, destination, shared_source_model=None):
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    annotation = source / 'safetyrisk_vqa.jsonl'
    labels = rows(annotation)
    if len(labels) != 210 or {str(r['id']) for r in labels} != {str(i) for i in range(210)}:
        raise ValueError('The source must contain exactly the frozen 210 S.3 cases')
    if VERSION == 1 and shared_source_model:
        raise ValueError('v1 retains model-specific rewrites; shared sources belong to v2')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.s3-snapshot-', dir=destination.parent) as temp:
        work = Path(temp) / 'GMAI'
        work.mkdir()
        (work / 'images').mkdir()
        shutil.copy2(annotation, work / 'labels.jsonl')
        provenance = []
        checksums = []
        for row in labels:
            source_image = source / 'images' / row['image']
            target = work / 'images' / row['image']
            shutil.copy2(source_image, target)
            checksum = digest(target)
            checksums.append(f'{checksum}  images/{row["image"]}')
            provenance.append({'id': row['id'], 'image': row['image'],
                               'source_image': str(source_image.relative_to(ROOT)),
                               'image_sha256': checksum})
        (work / 'checksums.sha256').write_text('\n'.join(checksums) + '\n')
        (work / 'provenance.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in provenance))
        rewrite_checksums = {}
        sources = []
        if VERSION == 1:
            for model in sorted(p for p in cache_root.iterdir() if p.is_dir()):
                for branch in BRANCHES:
                    source_file = model / f'{branch}-safety.jsonl'
                    if not source_file.is_file():
                        raise FileNotFoundError(source_file)
                    sources.append((source_file, Path('rewrites') / model.name / source_file.name))
        elif shared_source_model:
            for branch in BRANCHES:
                source_file = cache_root / shared_source_model / f'{branch}-safety.jsonl'
                sources.append((source_file, Path('shared_rewrites') / source_file.name))
        for source_file, relative in sources:
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            rewrite_checksums[str(relative)] = digest(target)
        manifest = {'schema_version': 1, 'task_id': 'S.3', 'name': 'dynamic_safety_vqa',
                    'experiment_version': VERSION, 'sample_count': 210, 'image_count': 210,
                    'branches': list(BRANCHES), 'base_clean_prompt': 'Question:{malicious_prompt}',
                    'source_annotation': str(annotation.relative_to(ROOT)),
                    'source_annotation_sha256': digest(annotation),
                    'shared_base_with': 'S.1', 'rewrite_checksums': rewrite_checksums,
                    'protocol': 'model_specific' if VERSION == 1 else ('shared' if shared_source_model else 'shared_pending'),
                    'shared_source_model': shared_source_model,
                    'shared_rewrites_frozen': bool(shared_source_model),
                    'status': 'existing_input_snapshot_no_new_clinical_review',
                    'model_or_judge_called': False}
        write_json(work / 'manifest.json', manifest)
        validate(work)
        work.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/safety/S3_dynamic_safety_vqa/GMAI')
    parser.add_argument('--rewrite-cache-dir', type=Path, default=ROOT / 'logs/safety/S3')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--shared-source-model', help='Explicitly select one v2 shared source; omitted means pending')
    args = parser.parse_args()
    print(build(args.source_dir.resolve(), args.rewrite_cache_dir.resolve(), args.output_dir.resolve(), args.shared_source_model))


if __name__ == '__main__':
    main()
