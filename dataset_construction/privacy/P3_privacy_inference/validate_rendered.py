#!/usr/bin/env python3
"""Check rendered P.3 candidates and create a local review contact sheet."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs/pilot')
    args = parser.parse_args()
    root = args.output_dir
    rows = [json.loads(line) for line in (root / 'image_generation_manifest.jsonl').read_text().splitlines()]
    instances = [json.loads(line) for line in (root / 'evaluation_candidates.jsonl').read_text().splitlines()]
    missing, invalid, inventory = [], [], []
    seen = {}
    sheet = Image.new('RGB', (1500, ((len(rows) + 2) // 3) * 430), 'white')
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for index, row in enumerate(rows):
        path = root / 'images' / row['image_filename']
        x, y = index % 3 * 500, index // 3 * 430
        if not path.exists():
            missing.append(row['base_id'])
            draw.text((x + 8, y + 50), 'Awaiting image', fill='red', font=label_font)
            continue
        try:
            with Image.open(path) as im:
                im.load()
                assert im.mode == 'RGB'
                assert min(im.size) >= 512
                thumb = im.copy()
                thumb.thumbnail((480, 360))
                sheet.paste(thumb, (x + (500 - thumb.width) // 2, y + 4))
                size = list(im.size)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest not in seen, f'Duplicate image: {seen.get(digest)}'
            seen[digest] = row['base_id']
            inventory.append({'base_id': row['base_id'], 'image': row['image_filename'], 'sha256': digest, 'size': size})
        except Exception as exc:
            invalid.append({'base_id': row['base_id'], 'error': str(exc)})
        draw.text((x + 8, y + 370), row['base_id'], fill='black', font=label_font)
        draw.text((x + 8, y + 398), row['attribute_id'], fill='black', font=label_font)
    assert len({row['base_id'] for row in rows}) == len(rows)
    assert len({row['instance_id'] for row in instances}) == len(instances)
    for row in rows:
        pair = [instance for instance in instances if instance['base_id'] == row['base_id']]
        assert len(pair) == 2
        assert {r['request_context'] for r in pair} == {'unauthorized_privacy', 'authorized_utility'}
        assert all(r['image_filename'] == row['image_filename'] for r in pair)
        assert all(row['visual_specification'].casefold() not in r['question'].casefold() for r in pair)
    report = {
        'status': 'candidate_assets_complete_review_pending' if not missing and not invalid else 'candidate_assets_incomplete',
        'base_image_count': len(rows), 'instance_count': len(instances),
        'rendered_image_count': len(inventory), 'missing': missing, 'invalid': invalid,
        'paired_questions': 'passed', 'verbatim_evidence_not_in_questions': 'passed',
        'clinical_review': 'pending', 'privacy_review': 'pending',
        'inventory': inventory,
    }
    (root / 'render_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    sheet.save(root / 'contact_sheet.jpg')
    print(json.dumps({k: v for k, v in report.items() if k != 'inventory'}, indent=2))


if __name__ == '__main__':
    main()
