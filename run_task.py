import yaml
import argparse
import warnings
warnings.filterwarnings("ignore")
from pprint import pprint

from ours.tasks.base import BaseTask
from ours.utils.registry import registry
# from ours.evaluators.metrics import _supported_metrics
from ours.utils.utils import DictAction, merge_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', choices=['all', 'rules', 'none'], default=None,
                        help='all: configured evaluators; rules: no LLM judges; none: answers only. '
                             'Overrides config score_mode, then SCORE_MODE environment (default: all).')
    parser.add_argument('--config', default='ours/configs/task/robustness/r2-adv-untarget.yaml', help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='re-evaluate saved response records without calling the target model',
    )
    parser.add_argument(
        '--responses-file',
        help='response JSONL to use; defaults to <log_file>.responses.jsonl',
    )
    parser.add_argument(
        '--output-log',
        help='new evaluated JSON path for --eval-only; defaults to *.scored.json',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    # List all available modules:
    pprint("models: {}".format(registry.list_chatmodels()))
    pprint("datasets: {}".format(registry.list_datasets()))
    pprint("methods: {}".format(registry.list_methods()))
    pprint("evaluators: {}".format(registry.list_evaluators()))
    pprint("metrics: {}".format(list(_supported_metrics.keys())))
    '''

    args = parse_args()
    config = args.config

    with open(config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if hasattr(args, "cfg_options") and args.cfg_options is not None:
            cfg = merge_config(cfg, args.cfg_options)
        log_file = cfg.get('log_file')
        response_file = args.responses_file or cfg.get('response_file')
        model_id = cfg.get('model_id')
        dataset_id = cfg.get('dataset_id')
        orchestrator_id = cfg.get('orchestrator_id', None)
        method_cfg = cfg.get('method_cfg', {})
        dataset_cfg = cfg.get('dataset_cfg', {})
        generation_kwargs = cfg.get('generation_kwargs', {})
        log_file = cfg.get('log_file')
        debug_mode = cfg.get('debug_mode')
        evaluator_seq_cfgs = cfg.get('evaluator_seq_cfgs', [])
        score_mode = args.score if args.score is not None else cfg.get('score_mode')

        if 'max_new_tokens' not in generation_kwargs.keys():
            generation_kwargs['max_new_tokens'] = 50
        if 'do_sample' not in generation_kwargs.keys(): 
            generation_kwargs['do_sample'] = False

        cfg['generation_kwargs'] = generation_kwargs
        cfg['config_path'] = config

        # if debug_mode is not None:
        #     if debug_mode == "True":
        #         debug_mode = True
        #     elif debug_mode == "False":
        #         debug_mode = False

        if debug_mode:
            pprint(cfg, width=150)

        runner = BaseTask(dataset_id=dataset_id, model_id=model_id, orchestrator_id=orchestrator_id, method_cfg=method_cfg, dataset_cfg=dataset_cfg, generation_kwargs=generation_kwargs, log_file=log_file, response_file=response_file, debug_mode=debug_mode, evaluator_seq_cfgs=evaluator_seq_cfgs, score_mode=score_mode)
        if args.eval_only:
            runner.rejudge(response_file=args.responses_file, output_log=args.output_log)
        else:
            runner.pipeline()
