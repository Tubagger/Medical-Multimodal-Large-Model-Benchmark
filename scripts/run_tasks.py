#!/usr/bin/env python3
"""Run the complete 16-task benchmark."""
import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def task_catalog():
    groups={}
    def add(task, dimension, configs, folder=None):
        groups[task]=[{'config':f'ours/configs/task/{dimension}/{name}.yaml',
                       'folder':f'logs/{dimension}/{folder or task}'} for name in configs]
    add('C1','capability',[f't1-anomaly-detection-{name}' for name in ['brain','chexpert','isic','mura','oct']])
    add('C2','capability',['t1-lesion-localization'])
    add('C3','capability',['t2-logical-reasoning'])
    add('R1','robustness',['r1-ood'])
    add('R2','robustness',['r2-adv-untarget'])
    add('R3','robustness',['r3-adv-target'])
    for task,dataset in [('R2','adv-untarget'),('R3','adv-target')]:
        groups[task]=[{**groups[task][0],'dataset_id':dataset+'-clean'},groups[task][0]]
    add('R4','robustness',['r4-robustness-vqa-'+name for name in ['answer-flip','question-negation','option-expansion','narrative-distraction']])
    add('F1','fairness',['f1-visual-preference'])
    add('F2','fairness',['f2-bias-vqa-'+name for name in ['race','language','emotion','cognitive']])
    add('P1','privacy',['p1-privacy-recognition'])
    add('P2','privacy',['p2-privacy-detection'])
    add('P3','privacy',['p3-privacy-inference'])
    add('P4','privacy',['p3-privacy-vqa-'+name for name in ['subtle-request','focus-distraction','trap-warning','well-intention']])
    add('S1','safety',['s1-safety-risk'])
    add('S2','safety',['s2-mcm-jailbreak'],'S2')
    add('S3','safety',['s3-safety-vqa-'+name for name in ['clean','risk-minimization','authority','urgency','consequence-obfuscation','diagnostic-anchoring']])
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--tasks',nargs='+',default=['all'])
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--score',choices=['none','rules','all'],default=None)
    parser.add_argument('--eval-only',action='store_true',help='只对已有回答评分，缺少回答时报错')
    parser.add_argument('--env-root',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.score is None:args.score='all' if args.eval_only else 'none'
    if args.eval_only and args.score=='none':parser.error('--eval-only requires --score all or rules')
    spec=importlib.util.spec_from_file_location('model_entry',ROOT/'run_model.py')
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    specs=entry.model_specs()
    env_name=specs.get(args.model,{}).get('env')
    if not env_name:
        if args.model not in ['gpt-5.4','gemini-3.5-flash','claude-opus-4.6']:
            parser.error('Unknown model')
        env_name='dmxapi'
    prefix=entry.environment_prefix(env_name,args.env_root)
    python=str(prefix/'bin/python')
    os.chdir(ROOT)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    env['CONDA_PREFIX']=str(prefix)
    env['PATH']=str(prefix/'bin')+os.pathsep+env.get('PATH','')
    def run(command):
        print(shlex.join(command),flush=True)
        if not args.dry_run:subprocess.run(command,env=env,check=True)
    import yaml
    catalog=task_catalog()
    tasks=list(catalog) if args.tasks==['all'] else args.tasks
    for task in tasks:
        if task not in catalog:parser.error(f'Unknown task {task}; choose {list(catalog)}')
    for task in tasks:
        if task == 'F2' and not args.eval_only:
            baseline = f'logs/fairness/F2/{args.model}/bias-ref.scored.json'
            run([python, 'run_task.py', '--config', 'ours/configs/task/fairness/f2-bias-ref.yaml',
                 '--score', args.score, '--cfg-options', f'model_id={args.model}',
                 f'log_file={baseline}', f'dataset_cfg.log_file={baseline}'])
        for job in catalog[task]:
            config=yaml.safe_load((ROOT/job['config']).read_text())
            dataset=job.get('dataset_id',config['dataset_id'])
            stem=f'{job["folder"]}/{args.model}/{dataset}'
            log=stem+'.scored.json'
            scoring_options = (['--eval-only', '--responses-file', stem+'.responses.jsonl',
                                '--output-log', log] if args.eval_only else [])
            run([python,'run_task.py','--config',job['config'],'--score',args.score,*scoring_options,'--cfg-options',
                 f'model_id={args.model}',f'dataset_id={dataset}',f'log_file={log}',f'dataset_cfg.log_file={log}'])

        merge = {'R4': ('robustness', 'robustness-vqa'), 'P4': ('privacy', 'privacy-vqa'),
                 'S3': ('safety', 'safety-vqa'), 'F2': ('bias', 'bias-vqa')}.get(task)
        if merge and (args.score=='all' or (args.score=='rules' and task in ('R4','F2'))):
            utility, dataset = merge
            sources=[]
            for job in catalog[task]:
                config=yaml.safe_load((ROOT/job['config']).read_text())
                name=config['dataset_id']
                if name.endswith('-clean'): continue
                extension='scored.json'
                sources.append(f'{job["folder"]}/{args.model}/{name}.{extension}')
            extension='scored.json'
            output=f'{catalog[task][0]["folder"]}/{args.model}/{dataset}.{extension}'
            run([python,f'ours/utils/{utility}_utils.py','--json_files',*sources,'--output_file',output])

if __name__=='__main__':main()
