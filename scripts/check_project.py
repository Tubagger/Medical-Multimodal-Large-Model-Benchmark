#!/usr/bin/env python3
"""Read-only validation of framework, launch scripts and frozen shared inputs."""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', action='store_true', help='Also verify external local model files')
    args = parser.parse_args()
    errors = []
    paths = list((ROOT/'ours').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    paths += [ROOT/name for name in ('run_model.py','run_task.py')]
    for path in paths:
        try: ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc: errors.append(str(exc))
    for path in (ROOT/'scripts/run').glob('*.sh'):
        result = subprocess.run(['bash','-n',str(path)],capture_output=True,text=True)
        if result.returncode: errors.append(result.stderr)
    warnings = []
    manifest_path = ROOT/'data/shared_inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {'jobs': []}
    if not manifest_path.is_file():
        warnings.append('Data are not included in this code-only release; skipped shared-input checks.')
    seen = set()
    pending_inputs = []
    from ours.utils.shared_inputs import validate_inputs
    for job in manifest['jobs']:
        shared = ROOT/'logs'/job['output_folder']/f'{job["dataset_id"]}.inputs.json'
        if not shared.is_file():
            pending_inputs.append(str(shared.relative_to(ROOT)))
            continue
        try:
            rows = validate_inputs(json.loads(shared.read_text()), shared)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if len(rows) != job['count']:
            errors.append(f'Invalid sample count: {shared}')
        for row in rows:
            image = ROOT / row['content']['image_path']
            if image in seen: continue
            seen.add(image)
            if not image.is_file(): errors.append(f'Image missing: {image}')
    plan = json.loads((ROOT/'scripts/run/three_machine_model_plan.json').read_text())
    models = [model for machine in plan['machines'] for worker in machine['workers'] for model in worker['models']]
    catalog = json.loads((ROOT/'run_state/shared/model_catalog.json').read_text())
    for model in models:
        entry = catalog[model['id']]
        if not (ROOT/entry['config_file']).is_file(): errors.append(f'Model config missing: {model["id"]}')
        if args.weights:
            folder=Path(entry['active_path'])
            if not folder.is_absolute(): folder=ROOT/folder
            for relative,size in entry.get('required_files',{}).items():
                path=folder/relative
                if not path.is_file() or path.stat().st_size!=size:errors.append(f'Weight missing/incomplete: {path}')
    print(json.dumps({'python_files_checked':len(paths),'local_models':len(models),
                      'shared_branches':len(manifest['jobs']),'images_checked':len(seen),
                      'pending_rewrite_files':pending_inputs, 'warnings':warnings,
                      'errors':errors},ensure_ascii=False,indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
