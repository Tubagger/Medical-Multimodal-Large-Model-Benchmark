"""Rewrite input, cache, inference and scoring helpers for R.4/P.4/S.3.

The task runner remains ``run_task.py``.  Rewrites are owned by the task
itself: an existing ``logs/<task>/<dataset>.inputs.json`` is reused, and only
a missing file enters the task's rewrite method.
"""
import fcntl
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ours.utils.response_records import (
    cached_answer_complete,
    default_response_file,
    default_score_file,
    equivalent_content,
    has_final_answer,
    load_response_records,
)


def valid_answer(value):
    return has_final_answer(value)


def reusable_answer(record):
    return cached_answer_complete(record)


def say(message):
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {message}', flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def result_record(row):
    """Keep answers and cache validation data without question provenance."""
    result = dict(row)
    extra = dict(result.get('extra', {}))
    for key in ('input_fingerprint', 'shared_source_model', 'generated_on'):
        extra.pop(key, None)
    if extra:
        result['extra'] = extra
    else:
        result.pop('extra', None)
    return result


def load_jobs(prepare=True):
    """Reuse task rewrites from each task's log directory.

    Missing files are generated only for R.4/P.4/S.3.
    """
    from ours.utils.shared_inputs import validate_inputs
    manifest = json.loads((ROOT / 'data/shared_inputs/manifest.json').read_text())
    jobs = []
    for job in manifest['jobs']:
        path = ROOT / 'logs' / job['output_folder'] / f'{job["dataset_id"]}.inputs.json'
        def generate(job=job, path=path):
            import yaml
            from ours.tasks.base import BaseTask
            dimension, prefix = {'R.4': ('robustness', 'r4'),
                                 'P.4': ('privacy', 'p3'),
                                 'S.3': ('safety', 's3')}[job['task_id']]
            config_path = ROOT / 'ours/configs/task' / dimension / f'{prefix}-{job["dataset_id"]}.yaml'
            config = yaml.safe_load(config_path.read_text())
            task_log_dir = ROOT / 'logs' / job['output_folder']
            task = BaseTask(dataset_id=job['dataset_id'], model_id='rewriter',
                            orchestrator_id=None, method_cfg=config.get('method_cfg', {}),
                            dataset_cfg={**config.get('dataset_cfg', {}),
                                         '_dataset_config': f'ours/configs/datasets/shared-{dimension}.yaml'},
                            log_file=str(task_log_dir / f'{job["dataset_id"]}.scored.json'),
                            score_mode='none')
            # BaseTask.get_rewritten_dataset owns the per-task rewrite lock.
            from ours.utils.shared_inputs import to_record
            return [to_record(sample) for sample in task.get_rewritten_dataset().dataset]
        if path.exists():
            samples = validate_inputs(json.loads(path.read_text()), path)
        elif prepare:
            # BaseTask.get_rewritten_dataset performs the atomic locking itself.
            samples = generate()
        else:
            # Preflight never invokes a rewrite model or creates outputs.
            archive = ROOT / job['samples']
            if not archive.is_file():
                raise FileNotFoundError(f'Missing shared inputs: {path}; run without --check-only to generate')
            samples = json.loads(archive.read_text())
        if len(samples) != job['count']:
            raise ValueError(f'Shared sample count differs: {path}')
        for sample in samples:
            content = sample['content']
            if not (ROOT / content['image_path']).is_file():
                raise FileNotFoundError(content['image_path'])
        jobs.append((job, samples))
    return manifest, jobs


def load_cached(path, samples):
    expected = {str(r['id']): r for r in samples}
    cached = {}
    sources = [Path(path)]
    # Older runs kept answers only in the evaluated JSON file.
    for candidate in (Path(default_score_file(str(path))), path.with_suffix('.json')):
        if candidate not in sources:
            sources.append(candidate)
    for candidate in sources:
        if not candidate.exists():
            continue
        if candidate.suffix == '.jsonl':
            rows = load_response_records(str(candidate))
        else:
            rows = json.loads(candidate.read_text()).get('per_sample_results', [])
        for row in rows:
            if not isinstance(row, dict) or row.get('id') is None:
                continue
            key = str(row['id'])
            if key in cached and reusable_answer(cached[key]) and not reusable_answer(row):
                continue
            if key not in expected:
                continue
            if (not equivalent_content(row.get('content'), expected[key].get('content'))
                    or row.get('target') != expected[key].get('target')):
                raise ValueError(f'Cached input differs from frozen release: {candidate}/{key}')
            cached[key] = {**row, 'content': expected[key]['content']}
    return cached


def save_scored(job, records, log, score_mode):
    if score_mode == 'none' or (score_mode == 'rules' and job['task_id'] != 'R.4'):
        return
    # Reuse the existing framework evaluator chains and scoring switches.
    import ours.evaluators  # noqa: F401
    from ours.tasks.base import BaseTask
    task = BaseTask(dataset_id=job['dataset_id'], model_id='', orchestrator_id=None,
                    evaluator_seq_cfgs=job['evaluator_seq_cfgs'], log_file=str(log), score_mode=score_mode)
    task.evaluators = task.get_evaluators()
    if not task.evaluators:
        return
    task.save_results(task.eval(records, use_eval_cache=False), merge_existing=False)


def score_saved_model(model_id, score_mode='all'):
    """Score saved answers only. Do not read images, construct prompts or load targets."""
    if score_mode == 'none':
        raise ValueError('--eval-only requires --score all or --score rules')
    manifest = json.loads((ROOT / 'data/shared_inputs/manifest.json').read_text())
    ready = []
    errors = []
    for job in manifest['jobs']:
        if score_mode == 'rules' and job['task_id'] != 'R.4':
            continue
        folder = ROOT / 'logs' / job['output_folder'] / model_id
        source = folder / f'{job["dataset_id"]}.responses.jsonl'
        if 'samples' in job:
            samples = json.loads((ROOT / job['samples']).read_text())
        else:
            samples = [{'id': row.get('id'), 'content': row.get('content'), 'target': row.get('target')}
                       for row in load_response_records(str(source))] if source.is_file() else []
        cache_path = source if source.is_file() else folder / f'{job["dataset_id"]}.scored.json'
        if not cache_path.is_file():
            errors.append(f'Missing answers: {cache_path}')
            continue
        cached = load_cached(source, samples)
        rows = [cached[str(sample['id'])] for sample in samples if str(sample['id']) in cached]
        ids = [str(row.get('id')) for row in rows]
        if len(rows) != job['count'] or len(set(ids)) != len(rows):
            errors.append(f'Incomplete/duplicate answer IDs: {cache_path}; expected {job["count"]}, got {len(rows)}')
            continue
        if any(not reusable_answer(row) for row in rows):
            errors.append(f'Invalid/missing answers: {cache_path}')
            continue
        output = folder / f'{job["dataset_id"]}.scored.json'
        ready.append((job, rows, cache_path, output))
    if errors:
        raise ValueError('Cannot score saved answers:\n' + '\n'.join(errors))
    for job, rows, source, output in ready:
        save_scored(job, rows, output, score_mode)
        say(f'Scored saved answers: {source} -> {output}')


def infer_model(model_id, score_mode, machine_id):
    manifest, jobs = load_jobs()
    state_dir = ROOT / 'run_state/shared' / machine_id
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / f'{model_id}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = None
        totals = {'model': model_id, 'score_mode': score_mode, 'jobs': []}
        sys.path.insert(0, str(ROOT))
        for job, samples in jobs:
            log = ROOT / 'logs' / job['output_folder'] / model_id / f'{job["dataset_id"]}.scored.json'
            responses = Path(default_response_file(str(log)))
            cached = load_cached(responses, samples)
            pending = [r for r in samples if not reusable_answer(cached.get(str(r['id']), {}))]
            say(f'{model_id} {job["task_id"]}/{job["branch"]}: reuse={len(samples)-len(pending)}, generate={len(pending)}')
            if pending and model is None:
                import ours.models  # noqa: F401
                from ours.utils.registry import registry
                cls = registry.get_chatmodel_class(model_id)
                if cls is None:
                    raise ValueError(f'Unknown model: {model_id}')
                model = cls(model_id=model_id)
            responses.parent.mkdir(parents=True, exist_ok=True)
            with responses.open('a') as stream:
                for number, sample in enumerate(pending, 1):
                    content = dict(sample['content'])
                    content['image_path'] = str(ROOT / content['image_path'])
                    kwargs = dict(job['generation_kwargs'])
                    system = kwargs.pop('system_prompt', None)
                    messages = [{'role': 'user', 'content': content}]
                    if system is not None:
                        messages.insert(0, {'role': 'system', 'content': system})
                    result = model.chat(messages=messages, **kwargs)
                    answer = result if isinstance(result, str) else getattr(result, 'content', str(result))
                    row = {'id': sample['id'], 'content': sample['content'], 'target': sample.get('target'),
                           'response': answer}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    cached[str(sample['id'])] = row
                    import torch
                    torch.cuda.empty_cache()
                    if number % 10 == 0 or number == len(pending):
                        say(f'{model_id} {job["branch"]}: {number}/{len(pending)} new responses')
            records = [cached[str(r['id'])] for r in samples]
            # Canonicalize the append checkpoint after successful branch completion.
            temporary = responses.with_name(responses.name + '.tmp')
            temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
            temporary.replace(responses)
            save_scored(job, records, log, score_mode)
            invalid = sum(not reusable_answer(r) for r in records)
            totals['jobs'].append({'task': job['task_id'], 'branch': job['branch'],
                                   'responses': len(records), 'invalid_answers': invalid,
                                   'new_responses': len(pending)})
            write_json(state_dir / f'{model_id}.json', totals)
        say(f'{model_id}: all {len(jobs)} branches processed; invalid answers are listed in the state file')
