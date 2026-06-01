from abc import ABC
import time
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from ours.datasets.base import BaseDataset, collate_fn
from ours.utils.registry import registry
from ours.methods.base import BaseMethod
from ours.orchestrator.base import BaseOrchestrator
from ours.models.base import BaseChat
from ours.evaluators.base import SequentialEvaluator
import warnings
import json
import os
from datetime import datetime
from tqdm import tqdm


class BaseTask(ABC):    
    def __init__(self, dataset_id: str, model_id: str, orchestrator_id: str, method_cfg: Optional[Dict] = {}, dataset_cfg: Optional[Dict] = {}, generation_kwargs: Optional[Dict] = {}, evaluator_seq_cfgs: List = [], log_file: Optional[str] = None) -> None:
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.method_cfg = method_cfg
        self.dataset_cfg = dataset_cfg
        self.orchestrator_id =  orchestrator_id
        self.evaluator_seq_cfgs = evaluator_seq_cfgs
        self.generation_kwargs = generation_kwargs
        self.log_file = log_file
        
    
    def get_handlers(self) -> None:
        self.model = self.get_model()
        self.method = self.get_method() # get method before dataset
        self.dataset = self.get_dataset()
        self.evaluators = self.get_evaluators()
        self.orchestrator = self.get_orchestrator()
    
    def get_model(self) -> BaseChat:
        model_cls = registry.get_chatmodel_class(self.model_id)
        model = model_cls(self.model_id)
        return model
    
    def get_dataset(self) -> BaseDataset:
        dataset_cls: Type[BaseDataset] = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(dataset_id=self.dataset_id, model_id=self.model_id, method_hook=self.method, **self.dataset_cfg)
        return dataset

    def get_method(self) -> BaseMethod:
        if not self.method_cfg:
            return None
        
        assert len(self.method_cfg.keys()) == 1
        method_id = list(self.method_cfg.keys())[0]

        method_kwargs = self.method_cfg[method_id]
  
        method_cls = registry.get_method_class(method_id)
        method = method_cls(method_id, **method_kwargs)
        return method
    
    def get_orchestrator(self) -> BaseOrchestrator:
        if self.orchestrator_id is not None:
            orch_cls = registry.get_chatmodel_class(self.orchestrator_model_id)
            orchestrator = orch_cls(self.orchestrator_model_id)
            return orchestrator
        else:
            return None
    
    def get_evaluators(self) -> List[SequentialEvaluator]:
        if not self.evaluator_seq_cfgs:
            return None
        evaluators = []

        for evaluator_seq_cfg in self.evaluator_seq_cfgs:
            evaluators.append(SequentialEvaluator(evaluator_seq_cfg=evaluator_seq_cfg))

        return evaluators


    def get_dataloader(self) -> DataLoader:
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, collate_fn=collate_fn)
        return dataloader

    def eval(self, responses: List[Dict[str, Any]]) -> Dict[str, Union[float, Sequence]]:
        ids: Sequence[str] = [response.get('id', None) for response in responses]
        contents: Sequence[str] = [response.get('content', '') for response in responses]
        preds: Sequence[str] = [response.get('response', '') for response in responses]
        labels: Sequence[str] = [response.get('target', '') for response in responses]
        extras: Sequence[str] = [response.get('extra', None) for response in responses]
        results = {}
        evals = {}
        subset_evals = {}
        
        for evaluator in self.evaluators:
            result,evals = evaluator(preds, labels, extras, self.log_file)
            for key in result.keys():
                if key in results.keys():
                    warnings.warn(f"{key} already exists in results.")
            results.update(result)   
        #sub aspects
        subset_eval = extras[0] is not None and "subset" in extras[0]
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
                for evaluator in self.evaluators:
                    subset_result, subset_evals = evaluator(preds_subset, labels_subset, extras_subset)
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

    def save_results(self, results: Dict[str, Any], suffix: str = "") -> None:

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
        if self.log_file is not None and os.path.exists(self.log_file):

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
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

            with open(self.log_file,"w", encoding="utf-8") as f:
                json.dump(formatted_results, f, indent=4, ensure_ascii=False)

    def generate(self, dataloader, **generate_kwargs):

        responses = []

        system_prompt = generate_kwargs.pop("system_prompt", None)

        pbar = tqdm(total=len(dataloader.dataset), desc="Inference")

        for batch_data in dataloader:

            for data in batch_data:
                id = data['id']
                message = data['message']
                target = data['target']
                extra = data.get('extra') or {}

                if system_prompt is not None:
                    message = [{"role": "system", "content": system_prompt}] + message

                response = self.model.chat(
                    messages=message,
                    **generate_kwargs
                )

                content = message[1]['content'] if system_prompt else message[0]['content']

                # parse response
                if isinstance(response, str):
                    resp_text = response
                else:
                    resp_text = getattr(response, "content", str(response))

                output = {
                    "id": id,
                    "content": content,
                    "response": resp_text,
                    "target": target,
                    "extra": extra,
                }

                print("=========================================================result=========================================================")
                print('id =',output['id'])
                print('content =',output['content'])
                print('response =',output['response'])
                print('target =',output['target'])
                print('extra =',output['extra'])
                print("========================================================================================================================")
                
                responses.append(output)

                pbar.update(1)

        pbar.close()
        return responses

    def pipeline(self) -> None:
        self.get_handlers()
        dataloader = self.get_dataloader()

        # for batch in dataloader:
        #     print("First batch:", batch)
        #     break

        if len(dataloader.dataset) == 0:
            print("all data eval already done.")
            return

        responses = self.generate(dataloader, **self.generation_kwargs)

        results = self.eval(responses)

        self.save_results(results)
