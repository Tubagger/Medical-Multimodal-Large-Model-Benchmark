"""Validate the frozen release without calling any model."""

import json
from collections import Counter

from PIL import Image

from build import DEST, HERE, sha256


def validate():
    labels = json.loads((DEST / 'labels.json').read_text(encoding='utf-8'))
    legacy = json.loads((HERE / 'source/legacy_labels.json').read_text(encoding='utf-8'))
    old = {r['id']: r for r in legacy}
    provenance = {r['id']: r for r in map(json.loads, (DEST / 'provenance.jsonl').read_text().splitlines())}
    assert len(labels) == len({r['id'] for r in labels}) == 270
    assert {r['id'] for r in labels} == set(old) == set(provenance)
    counts = Counter(r['class'] for r in labels)
    assert len(counts) == 18 and set(counts.values()) == {15}
    assert len(list((DEST / 'images').glob('*.jpg'))) == 270
    for row in labels:
        i = row['id']
        assert row['question'] == old[i]['question']
        assert row['class'] == old[i]['class']
        text = (HERE / f'source/attack_prompts/question_{i}.txt').read_text(encoding='utf-8').strip()
        assert row['attack_prompt'] == text == old[i]['answer'].strip()
        assert row['attack_prompt'] == row['question'] + row['attack_suffix']
        assert row['attack_suffix'].strip()
        assert sha256(DEST / 'images' / row['image']) == provenance[i]['image_sha256']
        with Image.open(DEST / 'images' / row['image']) as im:
            im.load()
            assert im.mode == 'RGB' and im.size == (336, 336)
    for line in (DEST / 'checksums.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        assert sha256(DEST / relative) == digest
    report = {'task_id': 'S.2', 'status': 'passed', 'samples': 270,
              'image_protocol': 'MCM optimized 336x336 RGB images',
              'text_protocol': 'exact source attack text including suffix',
              'class_counts': dict(sorted(counts.items())), 'requires_model_rerun': True}
    (DEST / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    validate()
