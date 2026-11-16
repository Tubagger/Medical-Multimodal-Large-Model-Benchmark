"""Shared rubrics, output validation, and framework adapters for local judges."""
import json
import re
from pathlib import Path

import yaml

from ours.evaluators.chatmodel_eval import LLMJudgeEvaluator

ROOT = Path(__file__).resolve().parents[2]


class Judge:
    name = None
    score_key = None
    allowed_scores = ()

    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding='utf-8'))
        if self.config['judge'] != self.name:
            raise ValueError(f'Wrong rubric for {self.name}: {config_path}')

    def instructions(self, task):
        tasks = self.config['tasks']
        if task not in tasks:
            raise ValueError(f'{self.name} does not support task {task}')
        return self.config['prompt'] + '\nTask-specific rules:\n' + tasks[task]

    def schema(self):
        return {self.score_key: list(self.allowed_scores), 'reason': 'short explanation'}

    def validate(self, value, response):
        raise NotImplementedError('Each judge defines its own result extraction')


def parse_final_json(text, allow_unclosed_wrapper=False):
    """Never accept JSON inside an unfinished reasoning block or extra prose."""
    if not isinstance(text, str):
        raise ValueError('Judge returned no final text')
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    if '<think>' in text:
        raise ValueError('Judge reasoning was truncated before a final answer')
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=unique_keys)
    except json.JSONDecodeError:
        if not allow_unclosed_wrapper:
            raise
        # Accept only one complete named criterion object with no trailing data.
        # No score, string, escape, or incomplete inner object is repaired.
        match = re.match(r'^\{\s*"(privacy|safety|response)"\s*:\s*', text)
        if not match:
            raise
        decoder = json.JSONDecoder(object_pairs_hook=unique_keys)
        inner, end = decoder.raw_decode(text[match.end():])
        if not isinstance(inner, dict) or text[match.end() + end:].strip():
            raise
        value = {match.group(1): inner}
    if not isinstance(value, dict):
        raise ValueError('Final answer must be a JSON object')
    return value


class LocalJudgeEvaluator(LLMJudgeEvaluator):
    """BaseTask adapter: receives original records, never only response strings.

    The OpenAI-compatible client is created only for the first valid uncached
    case. Rules-only scoring therefore neither loads a model nor contacts a server.
    """

    requires_llm_judge = True
    requires_case_context = True
    requires_task_id = True
    judge_names = None

    def __init__(self, evaluator_id, task_id, metrics_cfg=None,
                 judge_config='ours/configs/judges/deepseek_v4_1_flash_1017.yaml',
                 image_root=None, include_images=False, model_id=None, base_url=None):
        if metrics_cfg:
            raise ValueError('Local judges use built-in PPR/HCR/RAS metrics; omit metrics_cfg or use {}')
        super().__init__(evaluator_id, {})
        from .local_judge_engine import TASKS, load_config
        if task_id not in TASKS:
            if str(task_id).startswith('privacy-vqa-'):
                task_id = 'P.4'
            elif str(task_id).startswith('safety-vqa-'):
                task_id = 'S.3'
            elif task_id == 'privacy-inference':
                task_id = 'P.3'
            elif task_id == 'safety-risk':
                task_id = 'S.1'
            elif task_id == 'mcm-jailbreak':
                task_id = 'S.2'
        if task_id not in TASKS:
            raise ValueError(f'Cannot map dataset to judge task: {task_id}')
        config_path = Path(judge_config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        self.config, self.judges = load_config(config_path, task_id, self.judge_names)
        if model_id:
            self.config['model'] = model_id
        if base_url:
            self.config['base_url'] = base_url
        self.task_id = task_id
        self.image_root = Path(image_root) if image_root else ROOT
        self.include_images = include_images
        self.client = None
        self._cache = {}

    def process(self, preds, labels=None, extras=None, records=None, **kwargs):
        from .local_judge_engine import LocalJudgeClient, judge_case, prepare_case, sha
        if records is None or len(records) != len(preds):
            raise ValueError('Local judges require original records, including content.text; use BaseTask.eval or judge_saved_answers.py')
        results = []
        for response, record in zip(preds, records):
            if response != record.get('response'):
                raise ValueError('Local judges cannot follow a stage that rewrites model responses')
            try:
                case, image = prepare_case(record, self.task_id, self.image_root, self.include_images)
                fingerprint = sha(case)
                result = self._cache.get(fingerprint)
                if result is None:
                    if self.client is None:
                        self.client = LocalJudgeClient(self.config)
                    result = judge_case(case, image, self.task_id, self.judges,
                                        self.client, self.config['max_attempts'])
                    if result['status'] != 'judge_error':
                        self._cache[fingerprint] = result
                result = {**result, 'id': record.get('id'), 'fingerprint': fingerprint,
                          'selected_judges': [j.name for j in self.judges]}
            except ValueError as error:
                result = {'id': record.get('id'), 'status': 'invalid_input', 'judgments': {},
                          'reason': str(error), 'selected_judges': [j.name for j in self.judges]}
            results.append(result)
        return results, labels if labels is not None else [], extras, results

    def eval(self, preds, labels=None, extras=None, log_file=None, **kwargs):
        from .local_judge_engine import summarize, sha
        processed, _, _, evals = self.process(preds, labels, extras, **kwargs)
        summary = summarize(processed)
        protocol = {'runtime': self.config, 'rubrics': {j.name: j.config for j in self.judges},
                    'task': self.task_id}
        result = {**summary['metrics_0_to_100'],
                  'total_records': summary['total_records'],
                  'status_counts': summary['status_counts'],
                  'judge_metadata': {'formula': summary['formula'],
                                     'denominator_policy': summary['denominator_policy'],
                                     'protocol': protocol, 'protocol_sha256': sha(protocol)}}
        return result, evals
