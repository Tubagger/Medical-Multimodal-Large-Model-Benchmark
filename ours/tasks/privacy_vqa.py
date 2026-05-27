from ours import TxtSample
from ours.tasks.base import BaseTask
from typing import Optional, List, Union, Sequence, Any, Dict, Type
import warnings
from torch.utils.data import DataLoader
import torch
import os,json
from ours.utils.privacy_utils import PrivacyRewriter
from ours.datasets.base import collate_fn


class PrivacyVQATask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate(self, dataloader: DataLoader, **generate_kwargs) -> List[Dict[str, Any]]:
        print('len(self.dataset): ', len(dataloader.dataset))
        responses = []
        # 读取 system prompt（如果外部传了）
        system_prompt = generate_kwargs.pop("system_prompt", None)

        for batch_data in dataloader:
            for data in batch_data:  
                print(data)              
                message = data['message']
                target = data['target']
                extra: Dict[str, Any] = data['extra']

                if system_prompt is not None:
                    message = [{"role": "system", "content": system_prompt}] + message

                response = self.model.chat(messages=message, **generate_kwargs)
                output = {
                    "content": message[1]['content'],
                    "response": response.content,
                    "target": target,
                    "extra": extra,
                }
                print("output:",output)
            
                responses.append(output)

        return responses

    def eval(self, responses: List[Dict[str, Any]]) -> Dict[str, Union[float, Sequence]]:
        contents: Sequence[str] = [response['content'] for response in responses]
        preds: Sequence[str] = [response['response'] for response in responses]
        labels: Sequence[str] = [response['target'] for response in responses]
        extras: Sequence[str] = [response['extra'] for response in responses]
        results = {}

        for evaluator in self.evaluators:
            result = evaluator(preds, labels, extras=contents)

            for metric_name, metric_result in result.items():
                if "remain_details" in metric_result:
                    remain_list = metric_result.pop("remain_details")

                    for detail in remain_list:
                        idx = detail["id"]

                        if extras[idx] is None:
                            extras[idx] = {}

                        if not isinstance(extras[idx], dict):
                            extras[idx] = {"value": extras[idx]}

                        extras[idx]["score"] = detail["score"]
                        extras[idx]["reason"] = detail["reason"]
            
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
                    subset_result = evaluator(preds_subset, labels_subset, extras_subset)

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
                'content': contents if any(contents) else None,
                'pred': preds if any(preds) else None,
                'label': labels if any(labels) else None,
                'extra': extras if any(extras) else None,
            }
        )
        return results
    
    def run_second_round(self, first_round_file: str):

        dirpath, filename = os.path.split(first_round_file)
        name, ext = os.path.splitext(filename)
        first_round_file = os.path.join(dirpath, f"{name}_first_round{ext}")

        assert os.path.exists(first_round_file), f"[ERROR] File not found: {first_round_file}"

        # 读取 JSON
        with open(first_round_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 读取 remain_ids
        remain_ids = data["total_results"]["chatmodel_privacy_score:privacy_rate"]["remain_ids"]
        # print("Remain IDs:", remain_ids)

        if len(remain_ids) == 0:
            print("====================== No remaining samples, skip second round.")
            return None
        print(f" Running second round on {len(remain_ids)} remaining samples...")

        rewritten_dataset = []  # 用于后续 dataloader

        for idx in remain_ids:
            sample = self.dataset[idx]
            original_text = sample["text"]

            # four types rewrite
            for rewrite_type, rewrite_fn in {
                "subtle_request": PrivacyRewriter.subtle_request,
                "focus_misdirection": PrivacyRewriter.focus_misdirection,
                "trap_warning": PrivacyRewriter.trap_warning,
                "well_meaning_intention": PrivacyRewriter.well_meaning_intention,
            }.items():

                rew_text, _ = rewrite_fn(
                    original_prompt=original_text,
                    model="deepseek-r1",
                    temperature=0
                )

                #  关键修改：生成 TxtSample
                rewritten_dataset.append(
                    TxtSample(
                        text=rew_text,
                        target=None,
                        extra={
                            "id": idx,
                            "rewrite_type": rewrite_type,
                            "original": original_text
                        }
                    )
                )

        
        second_loader = DataLoader(
            dataset=rewritten_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        # for i, batch in enumerate(second_loader):
        #     print(f"Batch {i}:")
        #     print(batch)
        #     if i == 0:   # 只看前 4 个，避免打印太多
        #         break
        second_responses = self.generate(
            second_loader,
            **self.generation_kwargs
        )
        second_results = self.eval(second_responses)
        self.save_results(second_results, "second_round")

    def run_third_round(self, first_round_file: str):

        dirpath, filename = os.path.split(first_round_file)
        name, ext = os.path.splitext(filename)
        first_round_file = os.path.join(dirpath, f"{name}_second_round{ext}")

        assert os.path.exists(first_round_file), f"[ERROR] File not found: {first_round_file}"

        # 读取 JSON
        with open(first_round_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 读取 remain_ids
        remain_ids = data["total_results"]["chatmodel_privacy_score:privacy_rate"]["remain_ids"]
        # print("Remain IDs:", remain_ids)

        if len(remain_ids) == 0:
            print("====================== No remaining samples, skip second round.")
            return None
        print(f" Running second round on {len(remain_ids)} remaining samples...")

        rewritten_dataset = []  # 用于后续 dataloader

        for idx in remain_ids:
            sample = self.dataset[idx]
            original_text = sample["text"]

            rew_text, _ = PrivacyRewriter.combined(
                original_prompt=original_text,
                model="deepseek-r1",
                temperature=0
            )

            #  关键修改：生成 TxtSample
            rewritten_dataset.append(
                TxtSample(
                    text=rew_text,
                    target=None,
                    extra={
                        "id": idx,
                        "rewrite_type": "combined",
                        "original": original_text
                    }
                )
            )
    
        
        third_loader = DataLoader(
            dataset=rewritten_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        # for i, batch in enumerate(second_loader):
        #     print(f"Batch {i}:")
        #     print(batch)
        #     if i == 0:   # 只看前 4 个，避免打印太多
        #         break
        third_responses = self.generate(
            third_loader,
            **self.generation_kwargs
        )
        third_results = self.eval(third_responses)
        self.save_results(third_results, "third_round")

    def pipeline(self) -> None:
        self.get_handlers()
        dataloader = self.get_dataloader()
         # first_round
        first_responses = self.generate(dataloader, **self.generation_kwargs)
        first_results = self.eval(first_responses)
        self.save_results(first_results,'first_round')
         # second_round
        self.run_second_round(self.log_file)
        self.run_third_round(self.log_file)
        # print(self.dataset[1])