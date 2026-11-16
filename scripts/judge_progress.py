#!/usr/bin/env python3
"""Show judging progress from saved-answer checkpoints without changing any state."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE = 'deepseek-v4.1-flash-1017'
LAYOUT = {
    'P.3': ('privacy/P3', [('privacy-inference', 240)]),
    'P.4': ('privacy/P4', [(f'privacy-vqa-{b}', 243) for b in
                           ('subtle-request', 'focus-distraction', 'trap-warning', 'well-intention')]),
    'S.1': ('safety/S1', [('safety-risk', 210)]),
    'S.2': ('safety/S2', [('mcm-jailbreak', 270)]),
    'S.3': ('safety/S3', [(f'safety-vqa-{b}', 210) for b in
                          ('risk-minimization', 'authority', 'urgency',
                           'consequence-obfuscation', 'diagnostic-anchoring')]),
}


def discover_models():
    models = set()
    for folder, _ in LAYOUT.values():
        base = ROOT / 'logs' / folder
        if base.is_dir():
            models.update(p.name for p in base.iterdir()
                          if p.is_dir() and (p.name == 'gpt-5.4' or any(p.glob('*.responses.jsonl'))))
    return sorted(models)


def checkpoint_count(path, expected):
    if not path.exists():
        return 0
    seen = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get('fingerprint') or row.get('id')
        if key is not None:
            seen.add(str(key))
    return min(len(seen), expected)


def lock_busy(model):
    path = ROOT / 'run_state/judging' / JUDGE / (model + '.lock')
    if not path.exists():
        return False
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def summary_status(model):
    path = ROOT / 'run_state/judging' / JUDGE / (model + '.json')
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status', '')
    except (OSError, json.JSONDecodeError):
        return ''


def collect(model):
    tasks = {}
    done = total = 0
    for task, (folder, datasets) in LAYOUT.items():
        task_done = task_total = 0
        for dataset, expected in datasets:
            path = ROOT / 'logs' / folder / model / (dataset + '.scored.judgments.jsonl')
            task_done += checkpoint_count(path, expected)
            task_total += expected
        tasks[task] = (task_done, task_total)
        done += task_done
        total += task_total
    status = 'RUNNING' if lock_busy(model) else (summary_status(model).upper() or 'IDLE')
    return status, done, total, tasks


def render():
    models = discover_models()
    rows = [(model, *collect(model)) for model in models]
    grand_done = sum(row[2] for row in rows)
    grand_total = sum(row[3] for row in rows)
    width = max(len(model) for model in models)
    lines = [f'Judge: {JUDGE}   {time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'Overall: {grand_done}/{grand_total} ({100 * grand_done / grand_total:.1f}%)',
             '',
             f'{"model":<{width}}  {"status":<8} {"done/total":>13} {"pct":>6}  '
             + '  '.join(f'{task:>10}' for task in LAYOUT)]
    for model, status, done, total, tasks in rows:
        detail = '  '.join(f'{tasks[task][0]:>4}/{tasks[task][1]:<5}' for task in LAYOUT)
        lines.append(f'{model:<{width}}  {status:<8} {done:>6}/{total:<6} '
                     f'{100 * done / total:>5.1f}%  {detail}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-w', '--watch', type=float, metavar='SECONDS',
                        help='refresh continuously at this interval')
    args = parser.parse_args()
    try:
        while True:
            output = render()
            if args.watch and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(output, flush=True)
            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
