#!/usr/bin/env python3
"""Render six auditable synthetic records for the P.3 pilot (no AI image edits)."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def font(size):
    return ImageFont.truetype(str(FONT), size)


def render(document):
    image = Image.new('RGB', (1664, 1200), '#f1f3f5')
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((48, 40, 1616, 1160), radius=14, fill='white')
    draw.text((96, 76), 'SYNTHETIC CLINICAL RECORD', font=font(24), fill='#555555')
    draw.text((96, 126), document['title'], font=font(46), fill='#17212c')
    draw.text((96, 202), 'Patient code: ' + document['patient_code'], font=font(30), fill='#17212c')
    draw.line((96, 264, 1568, 264), fill='#b8c0c7', width=2)
    y = 296
    for heading, body in document['sections']:
        draw.text((96, y), heading, font=font(32), fill='#17212c')
        y += 54
        for line in body.splitlines():
            # Fail rather than silently cropping clinical content.
            assert draw.textlength(line, font=font(27)) < 1472, line
            assert y < 1060, 'Document is too long for the template'
            draw.text((96, y), line, font=font(27), fill='#252c33')
            y += 42
        y += 36
    draw.text((96, 1104), 'Fictional data for research; no real patient or institution.', font=font(22), fill='#555555')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs/pilot')
    args = parser.parse_args()
    fixture = HERE / 'pilot_documents.json'
    documents = json.loads(fixture.read_text(encoding='utf-8'))
    images = args.output_dir / 'images'
    images.mkdir(parents=True, exist_ok=True)
    provenance = []
    for base_id, document in documents.items():
        destination = images / f'{base_id}.png'
        render(document).save(destination)
        provenance.append({
            'base_id': base_id, 'image': destination.name,
            'source': 'programmatic_synthetic_document',
            'image_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'fixture': str(fixture.relative_to(HERE)),
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'font_sha256': hashlib.sha256(FONT.read_bytes()).hexdigest(),
            'status': 'rendered_candidate_pending_medical_review',
        })
    (args.output_dir / 'document_provenance.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in provenance), encoding='utf-8')
    print(f'Rendered {len(provenance)} fictional documents to {images}')


if __name__ == '__main__':
    main()
