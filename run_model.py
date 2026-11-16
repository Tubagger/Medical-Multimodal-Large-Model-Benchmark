#!/usr/bin/env python3
"""One-model entry point: select its environment and run frozen v2 tasks."""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALIASES = {
    'medgemma-4b': 'med-gemma-4b-it', 'medgemma-27b': 'med-gemma-27b-it',
    'qwen32b': 'qwen3-vl-32b-instruct',
}


def model_specs(root=ROOT):
    plan = json.loads((root / 'scripts/run/three_machine_model_plan.json').read_text())
    return {model['id']: {**model, 'machine': machine['id']}
            for machine in plan['machines'] for worker in machine['workers'] for model in worker['models']}


def environment_prefix(name, explicit=None):
    if explicit is not None:
        roots = [Path(explicit)]
    elif os.getenv('MEDBENCH_ENV_ROOT'):
        roots = [Path(os.environ['MEDBENCH_ENV_ROOT'])]
    else:
        roots = [ROOT.parent / 'envs']
        current = os.getenv('CONDA_PREFIX')
        if current:
            roots.extend([Path(current).parent, Path(current) / 'envs'])
        roots.extend([Path.home() / 'anaconda3/envs', Path.home() / 'miniconda3/envs', Path.home() / '.conda/envs'])
    for base in roots:
        prefix = (base / name).resolve()
        if (prefix / 'bin/python').is_file():
            return prefix
    raise FileNotFoundError(f'环境 {name} 尚未解包。请先在 {roots[0]} 解压 {name}.tar.gz 并运行 conda-unpack。')


def checkpoint_path(entry, explicit=None):
    if explicit:
        choices = [Path(explicit).resolve()]
    else:
        choices = [ROOT.parent / entry['directory_name']]
        for key in ('active_path', 'source_path'):
            path = Path(entry[key])
            choices.append(path if path.is_absolute() else ROOT / path)
    for path in choices:
        if (path / 'config.json').is_file():
            for relative, size in entry.get('required_files', {}).items():
                file = path / relative
                if not file.is_file() or file.stat().st_size != size:
                    raise FileNotFoundError(f'模型尚未传齐：{file}')
            return path.resolve()
    raise FileNotFoundError(f'未找到模型权重：{choices[0]}；各模型目录应与 code/ 同级。')


def configure_selected(model, checkpoint):
    import yaml
    catalog_path = ROOT / 'run_state/shared/model_catalog.json'
    lock_path = catalog_path.with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog = json.loads(catalog_path.read_text())
        entry = catalog[model]
        config_path = ROOT / entry['config_file']
        config = yaml.safe_load(config_path.read_text())
        config['model_path'] = str(checkpoint)
        temporary = config_path.with_name(config_path.name + '.tmp')
        temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        temporary.replace(config_path)
        entry['active_path'] = str(checkpoint)
        temporary = catalog_path.with_name(catalog_path.name + '.tmp')
        temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(catalog_path)


def main():
    parser = argparse.ArgumentParser(description='模型回答生成与判分统一入口；默认只生成共享任务回答。')
    parser.add_argument('--model', help='模型名；使用--list-models查看全部16个模型')
    parser.add_argument('--gpus', default='0,1', help='例如0,1或2,3，默认0,1')
    parser.add_argument('--score', choices=['none', 'rules', 'all'], default=None)
    parser.add_argument('--eval-only', action='store_true', help='只对已有JSONL评分，不加载目标模型或生成改写')
    parser.add_argument('--tasks', nargs='+', choices=['P.3','P.4','S.1','S.2','S.3'], help='只评分时的LLM任务，默认五项')
    parser.add_argument('--judge-model', help='裁判服务的模型名')
    parser.add_argument('--base-url', help='裁判服务URL')
    parser.add_argument('--judge-config', type=Path, help='裁判配置文件')
    parser.add_argument('--env-root', type=Path, help='环境目录，默认先查找与code同级的envs/')
    parser.add_argument('--model-path', type=Path, help='覆盖默认的同级模型目录')
    parser.add_argument('--check-only', action='store_true', help='检查当前模型的环境、权重和数据，不运行推理')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--_in-env', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.score is None:
        args.score = 'all' if args.eval_only else 'none'
    if args.eval_only and args.score == 'none':
        parser.error('--eval-only 不能与 --score none 同时使用')
    if (args.tasks or args.judge_model or args.base_url or args.judge_config) and not (args.eval_only and args.score=='all'):
        parser.error('裁判参数只用于 --eval-only --score all')
    specs = model_specs()
    if args.list_models:
        for model, spec in specs.items():
            print(f'{model:32} {spec["env"]}')
        return
    model = ALIASES.get(args.model, args.model)
    api_target = model in {'gpt-5.4','gemini-3.5-flash','claude-opus-4.6'}
    if model not in specs and not (args.eval_only and api_target):
        parser.error('请指定有效的--model，或用--list-models查看模型名；MedGemma现有4B和27B。')
    if not args.gpus or any(not part.isdigit() for part in args.gpus.split(',')):
        parser.error('--gpus必须是例如0,1这样的GPU编号')
    spec = specs.get(model, {'env':'dmxapi','machine':'api'})
    if args.eval_only:
        os.chdir(ROOT)
        if args._in_env:
            if args.score == 'all':
                from scripts.judge_saved_answers import parse_args as judge_args, run_model_batch
                forwarded = ['--model', model]
                for flag, value in [('--base-url', args.base_url), ('--judge-model', args.judge_model),
                                    ('--config', args.judge_config)]:
                    if value is not None: forwarded.extend([flag, str(value)])
                if args.tasks: forwarded.extend(['--tasks', *args.tasks])
                if args.check_only: forwarded.append('--check-only')
                run_model_batch(judge_args(forwarded))
            else:
                from ours.utils.shared_jobs import score_saved_model
                score_saved_model(model, args.score)
            return
        prefix = environment_prefix('dmxapi', args.env_root)
        env = os.environ.copy()
        env['CONDA_PREFIX'] = str(prefix)
        env['PATH'] = str(prefix / 'bin') + os.pathsep + env.get('PATH', '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env.setdefault('APIKEY_FILE', str(ROOT / 'env/apikey.yaml'))
        command = [str(prefix / 'bin/python'), str(Path(__file__).resolve()),
                   '--model', model, '--eval-only', '--score', args.score, '--_in-env']
        for flag, value in [('--base-url', args.base_url), ('--judge-model', args.judge_model),
                            ('--judge-config', args.judge_config)]:
            if value is not None: command.extend([flag, str(value)])
        if args.tasks: command.extend(['--tasks', *args.tasks])
        if args.check_only: command.append('--check-only')
        os.execve(command[0], command, env)
        return
    catalog = json.loads((ROOT / 'run_state/shared/model_catalog.json').read_text())
    if model not in catalog:
        raise ValueError(f'该代码包未包含{model}的配置清单；请使用统一code包。')
    checkpoint = checkpoint_path(catalog[model], args.model_path)
    from ours.utils.shared_jobs import load_jobs
    manifest, jobs = load_jobs(prepare=not args.check_only)
    if args._in_env:
        configure_selected(model, checkpoint)
        from ours.utils.shared_jobs import infer_model
        infer_model(model, args.score, spec['machine'])
        return
    prefix = environment_prefix(spec['env'], args.env_root)
    print(f'模型：{model}\n环境：{prefix}\n权重：{checkpoint}\nGPU：{args.gpus}\n任务：R.4/P.4/S.3（{len(jobs)}个分支）\n判分：{args.score}', flush=True)
    if args.check_only:
        print('检查通过，未加载模型。', flush=True)
        return
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': args.gpus, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    env.setdefault('APIKEY_FILE', str(ROOT / 'env/apikey.yaml'))
    command = ['/bin/bash', '-c', 'source "$1/bin/activate"; shift; exec python "$@"',
               'run-model', str(prefix), str(Path(__file__).resolve()), '--model', model,
               '--gpus', args.gpus, '--score', args.score, '--model-path', str(checkpoint), '--_in-env']
    os.chdir(ROOT)
    os.execve('/bin/bash', command, env)


if __name__ == '__main__':
    main()
