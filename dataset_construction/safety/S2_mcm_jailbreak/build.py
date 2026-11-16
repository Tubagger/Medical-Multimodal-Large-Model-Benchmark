"""Freeze the original 270-case cohort with the correct MCM attack artifacts."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEST = ROOT / 'data/safety/S2_mcm_jailbreak/GMAI'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_root):
    legacy = ROOT / 'data/safety/mcn_jailbreak/labels.json'
    selected = json.loads(legacy.read_text(encoding='utf-8'))
    assert len(selected) == len({r['id'] for r in selected}) == 270
    assert len(Counter(r['class'] for r in selected)) == 18
    assert set(Counter(r['class'] for r in selected).values()) == {15}
    artifacts = source_root / 'results/mcm/both'
    # Check everything before writing a release. Never fall back to clean images.
    for row in selected:
        image = artifacts / f"{row['id']}.jpg"
        text = artifacts / f"question_{row['id']}.txt"
        prompt = text.read_text(encoding='utf-8').strip()
        assert prompt == row['answer'].strip(), row['id']
        assert prompt.startswith(row['question']), row['id']
        assert prompt[len(row['question']):].strip(), row['id']
        with Image.open(image) as im:
            im.load()
            assert im.mode == 'RGB' and im.size == (336, 336)
        clean = ROOT / 'data/safety/mcn_jailbreak/images' / row['image']
        assert sha256(image) != sha256(clean), row['id']

    (DEST / 'images').mkdir(parents=True, exist_ok=True)
    (HERE / 'source/attack_prompts').mkdir(parents=True, exist_ok=True)
    (HERE / 'legacy').mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, HERE / 'source/legacy_labels.json')
    shutil.copy2(source_root / 'select_balanced_mcn.py', HERE / 'legacy/select_balanced_mcn.py')
    csv_dir = HERE / 'source/results'
    csv_dir.mkdir(parents=True, exist_ok=True)
    for csv in sorted((source_root / 'results').glob('*_mcm.csv')):
        shutil.copy2(csv, csv_dir / csv.name)

    labels, provenance = [], []
    for row in selected:
        sample_id = row['id']
        image = artifacts / f'{sample_id}.jpg'
        text = artifacts / f'question_{sample_id}.txt'
        prompt = text.read_text(encoding='utf-8').strip()
        name = f'{sample_id}.jpg'
        shutil.copy2(image, DEST / 'images' / name)
        shutil.copy2(text, HERE / 'source/attack_prompts' / text.name)
        labels.append({
            'id': sample_id, 'image': name, 'class': row['class'],
            'question': row['question'], 'attack_prompt': prompt,
            'attack_suffix': prompt[len(row['question']):],
        })
        provenance.append({
            'id': sample_id, 'source_image': str(image),
            'image_sha256': sha256(image), 'source_text': str(text),
            'source_text_sha256': sha256(text),
            'attack_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        })
    (DEST / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (DEST / 'provenance.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in provenance), encoding='utf-8')
    manifest = {
        'task_id': 'S.2', 'name': 'MCM jailbreak', 'sample_count': 270,
        'class_count': 18, 'samples_per_class': 15, 'sampling_seed': 42,
        'input_protocol': 'optimized image + exact saved question-and-suffix text; no appended instruction',
        'historical_runtime_compatible': False, 'requires_model_rerun': True,
        'source_root': str(source_root),
    }
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    paths = [DEST / 'labels.json', DEST / 'provenance.jsonl'] + sorted((DEST / 'images').glob('*.jpg'))
    (DEST / 'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(DEST)}\n' for p in paths), encoding='utf-8')
    print(f'Built {len(labels)} MCM samples -> {DEST}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path,
                        default=Path(os.environ.get('O2M_SOURCE_ROOT', 'third_party/O2M_attack-main')))
    build(parser.parse_args().source_root.resolve())
