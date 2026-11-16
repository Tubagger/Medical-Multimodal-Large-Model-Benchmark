#!/usr/bin/env python3
"""Judge all five privacy/safety tasks for a target model, or one answer file."""
import argparse
import fcntl
import json
import os
import re
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ours.evaluators.local_judge_engine import (  # noqa: E402
    TASKS, LocalJudgeClient, judge_case, load_config, prepare_case, sha, summarize,
)


def load_records(path):
    text = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        rows = value if isinstance(value, list) else value['per_sample_results']
    if not rows:
        raise ValueError('No saved answers found')
    if any(not isinstance(r, dict) or r.get('id') is None for r in rows):
        raise ValueError('Each record must be an object with an id')
    if len(rows) != len({str(r['id']) for r in rows}):
        raise ValueError('Duplicate sample IDs; provide one model and one branch per input file')
    return rows


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_checkpoint(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding='utf-8')
    cache = {}
    complete_lines = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith('\n'):
                break  # Recover only an interrupted final append, never middle corruption.
            raise
        cache[row['fingerprint']] = row
        complete_lines.append(line)
    if text and not text.endswith('\n'):
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text('\n'.join(complete_lines) + ('\n' if complete_lines else ''), encoding='utf-8')
        temporary.replace(path)
    return cache


def cached_result(cached, legacy_cached, fingerprint, signature, sample_id, case):
    """Reuse a renumbered checkpoint only when its original content/protocol hash matches."""
    result = cached.get(fingerprint)
    if result is None and str(sample_id) in legacy_cached:
        legacy = legacy_cached[str(sample_id)]
        legacy_key = sha({'protocol': signature,
                          'id': legacy['original_sample_id'], 'case': case})
        if legacy.get('fingerprint') == legacy_key:
            result = {**legacy, 'id': sample_id, 'fingerprint': fingerprint}
            result.pop('original_sample_id', None)
    return result


def run(args, client=None):
    config, judges = load_config(args.config, args.task,
                                 args.judges.split(',') if args.judges else None)
    config['backend'] = getattr(args, 'backend', 'http')
    if config['backend'] == 'transformers':
        if not getattr(args, 'model_path', None):
            raise ValueError('--backend transformers requires --model-path')
        config['model_path'] = str(Path(args.model_path).expanduser().resolve())
        config['device_map'] = args.device_map
    if args.base_url:
        config['base_url'] = args.base_url
    if getattr(args, 'judge_model', None):
        config['model'] = args.judge_model
    if type(config['max_attempts']) is not int or config['max_attempts'] < 1:
        raise ValueError('max_attempts must be a positive integer')
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    checkpoint = output.with_suffix('.judgments.jsonl')
    lock_path = output.with_suffix('.lock')
    if source in (output, checkpoint, lock_path):
        raise ValueError('Output and checkpoint must differ from the source answers')
    records = load_records(source)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        records = records[:args.limit]
    provenance = {'config': config, 'rubrics': {j.name: j.config for j in judges},
                  'implementation_sha256': sha({str(p.relative_to(ROOT)): p.read_text()
                                               for p in sorted((ROOT / 'ours/evaluators').glob('*.py'))
                                               if p.name in {'chatmodel_eval.py', 'transformers_judge.py', 'local_judge_base.py', 'local_judge_engine.py',
                                                             'privacy_judge.py', 'safety_judge.py',
                                                             'response_appropriateness.py'}}),
                  'task': args.task, 'selected_judges': [j.name for j in judges]}
    signature = sha(provenance)
    if args.check_only:
        valid = 0
        errors = []
        for row in records:
            try:
                prepare_case(row, args.task, args.image_root, args.include_images)
                valid += 1
            except ValueError as error:
                errors.append({'id': row['id'], 'reason': str(error)})
        report = {'task': args.task, 'judge_model': config['model'],
                          'base_url': config['base_url'], 'selected_judges': [j.name for j in judges],
                          'valid_inputs': valid, 'invalid_inputs': errors,
                          'judge_contacted': False}
        if not getattr(args, 'quiet', False):
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    new_calls = 0
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cached = load_checkpoint(checkpoint)
        legacy_cached = {str(r['id']): r for r in cached.values()
                         if 'original_sample_id' in r}
        results = []
        with checkpoint.open('a', encoding='utf-8') as journal:
            for index, row in enumerate(records, 1):
                try:
                    case, image = prepare_case(row, args.task, args.image_root, args.include_images)
                    fingerprint = sha({'protocol': signature, 'id': row['id'], 'case': case})
                    result = cached_result(cached, legacy_cached, fingerprint, signature, row['id'], case)
                    if result and result['status'] == 'judge_error':
                        result = None
                    if result is None:
                        if client is None:
                            if config['backend'] == 'transformers':
                                from ours.evaluators.transformers_judge import TransformersJudgeClient
                                client = TransformersJudgeClient(config)
                            else:
                                client = LocalJudgeClient(config)
                        result = judge_case(case, image, args.task, judges, client, config['max_attempts'])
                        new_calls += len(result['attempts'])
                        result.update(id=row['id'], fingerprint=fingerprint,
                                      selected_judges=[j.name for j in judges])
                        journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                        journal.flush()
                        os.fsync(journal.fileno())
                except ValueError as error:
                    result = {'id': row['id'], 'status': 'invalid_input', 'judgments': {},
                              'reason': str(error), 'selected_judges': [j.name for j in judges]}
                results.append(result)
                if not getattr(args, 'quiet', False):
                    print(f'[{index}/{len(records)}] {row["id"]}: {result["status"]}', flush=True)
        rows = [{**row, 'evals': {'local_judges': result}}
                for row, result in zip(records, results)]
        payload = {'source_answers': str(source), 'protocol_sha256': signature,
                             'new_judge_calls': new_calls,
                             'judge_protocol': provenance, 'total_results': summarize(results),
                             'per_sample_results': rows}
        atomic_json(output, payload)
        if not getattr(args, 'quiet', False):
            print(json.dumps(summarize(results), ensure_ascii=False, indent=2))
        return payload


BATCH_LAYOUT = {
    'P.3': ('privacy/P3', [('privacy-inference', 240)]),
    'P.4': ('privacy/P4', [('privacy-vqa-' + branch, 243) for branch in
                           ('subtle-request', 'focus-distraction', 'trap-warning', 'well-intention')]),
    'S.1': ('safety/S1', [('safety-risk', 210)]),
    'S.2': ('safety/S2', [('mcm-jailbreak', 270)]),
    'S.3': ('safety/S3', [('safety-vqa-' + branch, 210) for branch in
                         ('risk-minimization', 'authority', 'urgency',
                          'consequence-obfuscation', 'diagnostic-anchoring')]),
}


def model_slug(name):
    if not name or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('Target --model must be a model ID such as lingshu-7b, not a path')
    return name


def answer_file(folder, dataset):
    for suffix in ('.responses.jsonl', '.json'):
        path = folder / (dataset + suffix)
        if path.is_file():
            return path
    return None


def run_model_batch(args, client=None):
    target = model_slug(args.model)
    tasks = args.tasks or list(TASKS)
    if len(tasks) != len(set(tasks)):
        raise ValueError('Task selection contains duplicates')
    # Config model denotes the judge; --model always denotes the target model.
    config, _ = load_config(args.config, tasks[0])
    judge_model = args.judge_model or config['model']
    judge_tag = re.sub(r'[^A-Za-z0-9._-]+', '_', judge_model).strip('_')
    logs_root = (args.logs_root or ROOT / 'logs').resolve()
    output_root = (args.output_root or logs_root).resolve()
    entries = []
    for task in tasks:
        folder, datasets = BATCH_LAYOUT[task]
        for dataset, expected in datasets:
            source = answer_file(logs_root / folder / target, dataset)
            reuse = None
            output = output_root / folder / target / (dataset + '.scored.json')
            entry = {'task': task, 'dataset': dataset, 'expected_records': expected,
                     'input': str(source) if source else None, 'output': str(output),
                     'reuse': reuse, 'status': 'ready', 'issues': []}
            if source is None:
                entry.update(status='missing', issues=[f'No answers in {logs_root / folder / target}'])
            else:
                try:
                    rows = load_records(source)
                    entry['records'] = len(rows)
                    if len(rows) != expected:
                        entry['issues'].append(f'Incomplete branch: {len(rows)}/{expected} records')
                    invalid = 0
                    for row in rows:
                        try:
                            prepare_case(row, task, args.image_root, args.include_images)
                        except ValueError:
                            invalid += 1
                    entry['invalid_inputs'] = invalid
                    if invalid:
                        entry['issues'].append(f'{invalid} invalid or missing-image inputs')
                    if entry['issues']:
                        entry['status'] = 'incomplete'
                except (ValueError, OSError, KeyError, TypeError) as error:
                    entry.update(status='invalid_source', issues=[str(error)])
            entries.append(entry)
    report = {'target_model': target, 'judge_model': judge_model, 
              'tasks': tasks, 'output_root': str(output_root), 'branches': entries,
              'expected_branches': len(entries),
              'ready_branches': sum(e['status'] == 'ready' for e in entries),
              'judge_contacted': False}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return report
    if any(e['status'] != 'ready' for e in entries) and not args.allow_partial:
        raise ValueError('Some task inputs are missing/incomplete; no judge calls made. '
                         'Copy the missing answers, select --tasks, or explicitly use --allow-partial.')
    summary_file = ROOT / 'run_state/judging' / judge_tag / (target + '.json')
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report['status'] = 'running'
        atomic_json(summary_file, report)
        for entry in entries:
            if entry['status'] in ('missing', 'invalid_source'):
                continue
            branch_args = SimpleNamespace(**vars(args))
            branch_args.task = entry['task']
            branch_args.input = entry['input']
            branch_args.output = entry['output']
            # In single-file mode --judges remains available; model mode uses defaults.
            branch_args.judges = None
            payload = run(branch_args, client=client)
            totals = payload['total_results']
            entry['judging_results'] = totals
            entry['new_judge_calls'] = payload['new_judge_calls']
            entry['status'] = ('complete' if not entry['issues'] and args.limit is None
                               and totals['status_counts']['ok'] == entry['expected_records'] else 'partial')
            atomic_json(summary_file, report)
        report['status'] = 'complete' if all(e['status'] == 'complete' for e in entries) else 'partial'
        report['judge_contacted'] = any(e.get('new_judge_calls', 0) > 0 for e in entries)
        # Branch metrics stay separate; do not mix old task intersection formulas or
        # baseline-vs-attack metrics into an invented model-wide average.
        atomic_json(summary_file, report)
    print(f'Model judging {report["status"]}; summary={summary_file}', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Target model ID, e.g. lingshu-7b; defaults to all five LLM-scored tasks')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, help='Model mode task subset; default P.3 P.4 S.1 S.2 S.3')
    parser.add_argument('--logs-root', type=Path, help='Override source logs root in model mode')
    parser.add_argument('--output-root', type=Path, help='Override output root; default source logs root, alongside answers as *.scored.json')
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly permit available incomplete branches; report missing tasks')
    parser.add_argument('--task', choices=TASKS, help='Single-file mode task')
    parser.add_argument('--input', help='Single-file mode JSON or responses.jsonl')
    parser.add_argument('--output', help='Single-file mode new scored JSON')
    parser.add_argument('--config', type=Path, default=ROOT / 'ours/configs/judges/deepseek_v4_1_flash_1017.yaml')
    parser.add_argument('--image-root', type=Path, default=ROOT, help='Root for relative saved image paths')
    parser.add_argument('--include-images', action='store_true', help='Send images for tasks other than P.3 too; P.3 always requires images')
    parser.add_argument('--judges', help='Optional comma-separated subset, e.g. privacy or response')
    parser.add_argument('--backend', choices=['http', 'transformers'], default='http')
    parser.add_argument('--model-path', help='Local multimodal judge weights for Transformers')
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--base-url', help='Override local judge server URL')
    parser.add_argument('--judge-model', help='Override served JUDGE model ID; distinct from target --model')
    parser.add_argument('--limit', type=int, help='Score only the first N records for a pilot')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without contacting the judge')
    parser.add_argument('--quiet', action='store_true', help='Save full results without per-file console summaries')
    args = parser.parse_args(argv)
    if args.model:
        if args.task or args.input or args.output or args.judges:
            parser.error('--model mode uses --tasks and --output-root, not --task/--input/--output/--judges')
    elif not (args.task and args.input and args.output):
        parser.error('Specify --model TARGET, or all of --task/--input/--output')
    elif args.tasks or args.logs_root or args.output_root or args.allow_partial:
        parser.error('Batch options require --model TARGET')
    return args


def main():
    args = parse_args()
    try:
        if args.model:
            run_model_batch(args)
        else:
            run(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
