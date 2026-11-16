from abc import ABC
from collections import Counter
from typing import Optional, List, Union, Sequence, Any, Dict, Tuple, Type
import torch
from torch.utils.data import DataLoader
from ours.datasets.base import BaseDataset, collate_fn
from ours.utils.registry import registry
from ours.methods.base import BaseMethod
from ours.evaluators.base import SequentialEvaluator
from ours.utils.response_records import (
    default_response_file,
    default_score_file,
    load_response_records,
    has_final_answer,
    cached_answer_complete,
    equivalent_content,
    upsert_response_records,
    write_response_records,
)
import warnings
import json
import os
from pathlib import Path
from datetime import datetime
import copy
import hashlib
from ours.utils.shared_inputs import input_path, ensure_inputs, to_record, as_dataset
warnings.filterwarnings("ignore")

class BaseTask(ABC):    
    def __init__(self, dataset_id: str, model_id: str, orchestrator_id: str, method_cfg: Optional[Dict] = {}, dataset_cfg: Optional[Dict] = {}, generation_kwargs: Optional[Dict] = {}, log_file: Optional[str] = None, response_file: Optional[str] = None, debug_mode: Optional[bool] = False, evaluator_seq_cfgs: List = [], score_mode: Optional[str] = None, ) -> None:
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.method_cfg = method_cfg
        self.dataset_cfg = dict(dataset_cfg or {})
        self.orchestrator_id =  orchestrator_id
        self.evaluator_seq_cfgs = evaluator_seq_cfgs
        self.score_mode = score_mode if score_mode is not None else os.getenv('SCORE_MODE', 'all')
        if self.score_mode not in {'all', 'rules', 'none'}:
            raise ValueError(f'Unknown scoring mode: {self.score_mode}; use all, rules, or none')
        self.generation_kwargs = generation_kwargs
        self.legacy_log_file = log_file
        self.log_file = default_score_file(log_file)
        self.response_file = response_file or default_response_file(log_file)
        if self.dataset_cfg.get('log_file'):
            self.dataset_cfg['log_file'] = default_score_file(self.dataset_cfg['log_file'])
        elif self.log_file:
            self.dataset_cfg['log_file'] = self.log_file
        self.debug_mode = debug_mode
    
    def get_handlers(self) -> None:
        self.model = None  # Load the target only if there are missing answers.
        if self.method_cfg:
            # Rewriting task: check this task's own log folder first.  The
            # rewritten inputs are task-level, not model-level, so one
            # ``<dataset>.inputs.json`` serves every model of that task.
            self.dataset = self.get_rewritten_dataset()
            self.method = None
        else:
            self.method = self.get_method()
            self.dataset = self.get_dataset()
        self.evaluators = self.get_evaluators()
        self.orchestrator = self.get_orchestrator()

    def get_model(self) -> Any:
        # Importing the model registry is expensive and unnecessary for
        # evaluation-only runs, so keep it on the inference path.
        import ours.models  # noqa: F401
        model_cls = registry.get_chatmodel_class(self.model_id)
        model = model_cls(self.model_id)
        return model
    
    def get_dataset(self) -> BaseDataset:
        import ours.datasets  # noqa: F401  Register dataset classes lazily.
        dataset_cls: Type[BaseDataset] = registry.get_dataset_class(self.dataset_id)
        kwargs = dict(self.dataset_cfg)
        config_override = kwargs.pop('_dataset_config', None)
        original_config = dataset_cls.dataset_config
        try:
            if config_override:
                dataset_cls.dataset_config = config_override
            return dataset_cls(dataset_id=self.dataset_id, model_id=self.model_id,
                               method_hook=self.method, debug_mode=self.debug_mode, **kwargs)
        finally:
            dataset_cls.dataset_config = original_config

    FROZEN_REWRITE_PREFIXES = ('robustness-vqa-', 'privacy-vqa-', 'safety-vqa-')

    def _rewrite_input_path(self) -> Path:
        """Where this task's rewritten inputs live.

        R.4/P.4/S.3 rewrites do not depend on the target model, so their
        rewritten file sits next to the task logs and is reused by every
        model.  F.2 rewrites depend on each model's own reference answers, so
        its file stays inside that model's folder.
        """
        if self.dataset_id.startswith(self.FROZEN_REWRITE_PREFIXES):
            return Path(input_path(self.log_file, self.dataset_id,
                                   self.model_id, self.debug_mode))
        folder = Path(self.log_file).parent
        suffix = '.debug.inputs.json' if self.debug_mode else '.inputs.json'
        return folder / (self.dataset_id + suffix)

    def get_rewritten_dataset(self):
        """Load rewritten inputs, generating them only when the file is absent.

        The rewritten file is checked first.  Only when it is missing does the
        task invoke its rewrite method; the result is then published atomically
        for later runs and other models.
        """
        path = self._rewrite_input_path()
        print(f'Rewrite inputs: {path}', flush=True)
        def build():
            original_cfg = self.method_cfg
            original_dataset_cfg = self.dataset_cfg
            original_model_id = self.model_id
            original_method = getattr(self, "method", None)
            try:
                import ours.models  # Register the configured rewrite API adapters.
                self.method = self.get_method()
                self.dataset_cfg = {**original_dataset_cfg, 'use_cache': False,
                                    'log_file': str(path.parent / '__inputs_only__.json')}
                dataset = self.get_dataset()
                return [to_record(dataset[index]) for index in range(len(dataset))]
            finally:
                self.model_id = original_model_id
                self.method_cfg = original_cfg
                self.dataset_cfg = original_dataset_cfg
                self.method = original_method
        return as_dataset(ensure_inputs(path, build))

    def get_method(self) -> BaseMethod:
        if not self.method_cfg:
            return None
        import ours.methods  # noqa: F401  Register rewrite methods lazily.
        assert len(self.method_cfg.keys()) == 1
        method_id = list(self.method_cfg.keys())[0]
        method_kwargs = self.method_cfg[method_id]

        method_cls = registry.get_method_class(method_id)
        method = method_cls(model_id=self.model_id , method_id=method_id, debug_mode=self.debug_mode, **method_kwargs)
        return method
    
    def get_orchestrator(self) -> Any:
        if self.orchestrator_id is not None:
            orch_cls = registry.get_chatmodel_class(self.orchestrator_model_id)
            orchestrator = orch_cls(self.orchestrator_model_id)
            return orchestrator
        else:
            return None
    
    def get_evaluators(self) -> List[SequentialEvaluator]:
        if self.score_mode == 'none':
            return None
        if not self.evaluator_seq_cfgs:
            return None
        import ours.evaluators  # noqa: F401  Register evaluator classes lazily.
        evaluators = []

        for evaluator_seq_cfg in self.evaluator_seq_cfgs:
            if self.score_mode == 'rules':
                # Skip the entire chain if any stage uses a judge. Removing
                # just that stage could change the input of downstream rules.
                uses_judge = False
                for evaluator_id in evaluator_seq_cfg:
                    evaluator_cls = registry.get_evaluator_class(evaluator_id)
                    if evaluator_cls is None:
                        raise ValueError(f'Unknown evaluator: {evaluator_id}')
                    uses_judge |= evaluator_cls.requires_llm_judge
                if uses_judge:
                    print(f'[Scoring] Deferred LLM judge chain: {list(evaluator_seq_cfg)}', flush=True)
                    continue
            evaluators.append(SequentialEvaluator(evaluator_seq_cfg=evaluator_seq_cfg,
                                                   task_id=self.dataset_id))

        return evaluators


    def get_dataloader(self) -> DataLoader:
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, collate_fn=collate_fn)
        return dataloader

    @staticmethod
    def _sample_id(sample: Any) -> Any:
        if isinstance(sample, dict):
            return sample.get("id")
        return getattr(sample, "id", None)

    def eval(self, responses: List[Dict[str, Any]], use_eval_cache: bool = False) -> Dict[str, Union[float, Sequence]]:
        ids: Sequence[str] = [response.get('id', None) for response in responses]
        contents: Sequence[str] = [response.get('content', '') for response in responses]
        preds: Sequence[str] = [
            response.get('response', response.get('major_answer', ''))
            for response in responses
        ]
        labels: Sequence[str] = [response.get('target', '') for response in responses]
        extras: Sequence[str] = [response.get('extra', None) for response in responses]
        results = {}
        evals = {}
        subset_evals = {}
        
        for evaluator in self.evaluators or []:
            evaluator_log = self.log_file if use_eval_cache else None
            context = {'records': responses} if getattr(evaluator, 'requires_case_context', False) else {}
            result,evals = evaluator(preds, labels, extras, evaluator_log, **context)
            for key in result.keys():
                if key in results.keys():
                    warnings.warn(f"{key} already exists in results.")
            results.update(result)   
        #sub aspects
        subset_eval = bool(extras) and isinstance(extras[0], dict) and "subset" in extras[0]
        if subset_eval:
            # Evaluate with subset of dataset, `extra` field in dataclass must have `subset` key to enable subset evaluation.
            subset_list = [item['subset'] for item in extras]
            
            assert any(subset_list)

            subsets = set(subset_list)
            for subset in subsets:
                preds_subset = [preds[i] for i in range(len(preds)) if subset_list[i] == subset]
                labels_subset = [labels[i] for i in range(len(labels)) if subset_list[i] == subset]
                extras_subset = [extras[i] for i in range(len(extras)) if subset_list[i] == subset]

                subset_results = {}
                for evaluator in self.evaluators or []:
                    context = {'records': [responses[i] for i in range(len(responses)) if subset_list[i] == subset]} if getattr(evaluator, 'requires_case_context', False) else {}
                    subset_result, subset_evals = evaluator(preds_subset, labels_subset, extras_subset, **context)
                    for key in subset_result.keys():
                        subset_key = f"{key}_{subset}"
                        if subset_key in results.keys():
                            warnings.warn(f"{subset_key} already exists in results.")
                        if not isinstance(subset_result[key], Sequence):
                            # only need summary_keys
                            subset_results[subset_key] = subset_result[key]
                results.update(subset_results)

        results.update(
            {
                'id': ids if any(ids) else None,
                'content': contents if any(contents) else None,
                'response': preds if any(preds) else None,
                'target': labels if any(labels) else None,
                'extra': extras if any(extras) else None,
                'evals': evals if any(evals) else None,\
                'subset_evals': subset_evals if any(subset_evals) else None
            }
        )
        return results

    def save_results(self, results: Dict[str, Any], suffix: str = "", merge_existing: bool = True) -> None:
        if self.score_mode == 'none':
            return

        scatter_keys = [key for key, value in results.items() if isinstance(value, Sequence)]
        summary_keys = [key for key, value in results.items() if not isinstance(value, Sequence) and value is not None]

        if scatter_keys:
            seq_len = len(results[scatter_keys[0]])
            for key in scatter_keys:
                assert len(results[key]) == seq_len

        per_sample_results = []
        # =========================
        # merge old results
        # =========================
        if merge_existing and self.log_file is not None and os.path.exists(self.log_file):

            with open(self.log_file, "r", encoding="utf-8") as f:

                old_results = json.load(f)

                old_samples = old_results.get("per_sample_results", [])

            per_sample_results.extend(old_samples)


        for idx in range(seq_len):
            per_sample_result = {}
            for key in scatter_keys:
                per_sample_result[key] = results[key][idx]
            per_sample_results.append(per_sample_result)

        formatted_results = {
            "total_results": {},
            "per_sample_results": per_sample_results
        }

        for key in summary_keys:
            formatted_results["total_results"][key] = results[key]

        if self.log_file is not None:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

            with open(self.log_file,"w", encoding="utf-8") as f:
                json.dump(formatted_results, f, indent=4, ensure_ascii=False)

    def generate(self, dataloader, **generate_kwargs):

        responses = []

        repeat_times = generate_kwargs.pop("repeat_times", 1)
        system_prompt = generate_kwargs.pop("system_prompt", None)
        total = len(dataloader.dataset)
        count = 0
        for batch_data in dataloader:

            for data in batch_data:
                count += 1
                id = data['id']
                message = data['message']
                target = data['target']
                extra = data.get('extra') or {}

                if system_prompt is not None:
                    message = [{"role": "system", "content": system_prompt}] + message

                preds = []
                repeat_extras = []
                for repeat_idx in range(repeat_times):
                    response = self.model.chat(
                        messages=message,
                        **generate_kwargs
                    )
                    if isinstance(response, str):
                        preds.append(response)
                    else:
                        preds.append(getattr(response, "content", str(response)))
                    repeat_extras.append({**extra, "repeat_idx": repeat_idx})
                torch.cuda.empty_cache()
                content = message[1]['content'] if system_prompt else message[0]['content']

                if repeat_times == 1:
                    output = {
                        "id": id,
                        "content": content,
                        "response": preds[0],
                        "target": target,
                        "extra": extra,
                    }
                else:
                    clean = [value for value in preds if isinstance(value, str) and value.strip()]
                    major_answer = None
                    major_ratio = None
                    if clean:
                        counts = Counter(clean)
                        best_count = max(counts.values())
                        major_answer = sorted(value for value, n in counts.items() if n == best_count)[0]
                        major_ratio = best_count / len(clean)
                    output = {
                        "id": id,
                        "content": content,
                        "pred": preds,
                        "major_answer": major_answer,
                        "major_ratio": major_ratio,
                        "target": target,
                        "extra": repeat_extras,
                    }

                if not extra and repeat_times == 1:
                    output.pop('extra', None)
                if self.debug_mode:
                    print("=========================================================result=========================================================")
                    print('id =',output['id'])
                    print('content =',output['content'])
                    print('response =',output['response'])
                    print('target =',output['target'])
                    print('extra =',output['extra'])
                    print("========================================================================================================================")

                responses.append(output)
                if count % 10 == 0 or count == total:
                    self._response_records = upsert_response_records(
                        getattr(self, "_response_records", []),
                        responses,
                    )
                    write_response_records(self.response_file, self._response_records)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Inference] {count}/{total} finished", flush=True)

        return responses

    def _load_cached_responses(self) -> List[Dict[str, Any]]:
        """Load every reusable answer source, preferring canonical JSONL.

        Older runs stored answers only inside ``*.scored.json``.  Both sources
        are merged by sample ID; a valid JSONL row always wins.  Rows carrying
        errors or truncated thinking are returned too, but the caller checks
        the answer and regenerates those samples.
        """
        sources = []
        # Lowest to highest priority: historical scored JSON, canonical
        # scored JSON, then append-only response JSONL.
        for path in (self.legacy_log_file, self.log_file, self.response_file):
            if path and Path(path).is_file() and str(Path(path).resolve()) not in sources:
                sources.append(str(Path(path).resolve()))
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for path in sources:
            if not path or not Path(path).is_file():
                continue
            if str(path).endswith('.jsonl'):
                loaded = load_response_records(path)
            else:
                loaded = json.loads(Path(path).read_text(encoding='utf-8')).get('per_sample_results', [])
            for row in loaded:
                if not isinstance(row, dict) or row.get('id') is None:
                    continue
                key = str(row['id'])
                complete = cached_answer_complete(row)
                if key in merged and cached_answer_complete(merged[key]) and not complete:
                    continue
                if key not in merged:
                    order.append(key)
                merged[key] = row
        return [merged[key] for key in order]

    def pipeline(self) -> None:
        self.get_handlers()
        cached_rows = self._load_cached_responses()
        by_id = {str(row['id']): row for row in cached_rows if row.get('id') is not None}
        pending = []
        selected_ids = []
        reused = 0
        for sample in self.dataset.dataset:
            sample_id = self._sample_id(sample)
            selected_ids.append(str(sample_id))
            cached = by_id.get(str(sample_id))
            if cached:
                expected = to_record(sample)
                if (not equivalent_content(cached.get('content'), expected['content'])
                        or cached.get('target') != expected.get('target')):
                    raise ValueError(f'Cached input differs for sample {sample_id}')
                if cached_answer_complete(cached):
                    reused += 1
                    continue
            pending.append(sample)
        total = len(self.dataset.dataset)
        print(f'Cache: reused={reused}, pending={len(pending)}, total={total}', flush=True)
        self.dataset.dataset = pending
        self._response_records = cached_rows
        if pending:
            if getattr(self, 'model', None) is None:
                self.model = self.get_model()
            generated = self.generate(self.get_dataloader(), **self.generation_kwargs)
            self._response_records = upsert_response_records(self._response_records, generated)
        if not self._response_records:
            print('No responses to save or score.')
            return
        write_response_records(self.response_file, self._response_records)
        if self.score_mode == 'none':
            print(f'Answers only -> {self.response_file}')
            return
        # Score exactly the current task selection, in dataset order; cached
        # rows outside a debug/sample_limit subset must not enter the metrics.
        by_cached_id = {str(row['id']): row for row in self._response_records if row.get('id') is not None}
        missing = [sample_id for sample_id in selected_ids if sample_id not in by_cached_id]
        if missing:
            raise ValueError(f'Missing answers after generation for ids: {missing[:10]}')
        records = [by_cached_id[sample_id] for sample_id in selected_ids]
        # Repeated sampling stores majority votes even without evaluator chains.
        if not self.evaluators and type(self).eval is BaseTask.eval:
            print(f'No applicable evaluators; answers saved -> {self.response_file}')
            return
        results = self.eval(records)
        self.save_results(results, merge_existing=False)

    def rejudge(self, response_file: Optional[str] = None, output_log: Optional[str] = None) -> None:
        """Evaluate saved responses without loading or calling the target model."""
        if self.score_mode == 'none':
            raise ValueError('Evaluation-only requires --score rules or --score all')
        source = response_file or self.response_file
        responses = load_response_records(source)
        if not responses:
            raise ValueError(f"No response records found: {source}")
        destination = default_score_file(output_log or self.log_file)
        if not destination:
            raise ValueError('An output log path is required for evaluation-only mode')
        if Path(destination).resolve() == Path(source).resolve():
            raise ValueError('Scoring output must differ from raw answers')
        self.log_file = destination
        self.evaluators = self.get_evaluators()
        if not self.evaluators:
            print('No applicable evaluators; saved answers are unchanged.')
            return
        results = self.eval(responses, use_eval_cache=False)
        self.save_results(results, merge_existing=False)
        print(f"Rejudged {len(responses)} responses -> {destination}")
