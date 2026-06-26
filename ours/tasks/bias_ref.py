from abc import ABC
from collections import Counter
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from ours.tasks.base import BaseTask
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


class BiasRefTask(BaseTask):
    """
    BiasRefTask:
    - 与 BaseTask 基本一致
    - 不同点：对每个样本重复询问模型 repeat_times 次
      例如:3 个样本、repeat_times=2 -> 总调用 6 次 chat
    - 输出 responses 会包含 repeat_idx 字段（放在 extra 里）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def eval(self, responses):

        def _major_vote_and_ratio(votes):
            clean = [v for v in votes if v is not None and str(v).strip() != ""]
            if not clean:
                return None, None

            c = Counter(clean)
            top = c.most_common()

            best_cnt = top[0][1]

            tied = sorted(
                [k for k, v in top if v == best_cnt],
                key=str
            )

            major = tied[0]
            ratio = best_cnt / len(clean)

            return major, ratio

        ids = []
        contents = []
        preds_all = []
        major_answers = []
        major_ratios = []
        extras_all = []

        for r in responses:

            preds = r.get("pred", [])

            major_answer, major_ratio = _major_vote_and_ratio(preds)

            ids.append(r.get("id"))
            contents.append(r.get("content"))
            preds_all.append(preds)
            major_answers.append(major_answer)
            major_ratios.append(major_ratio)
            extras_all.append(r.get("extra"))

        return {
            "id": ids,
            "content": contents,
            "pred": preds_all,
            "major_answer": major_answers,
            "major_ratio": major_ratios,
            "extra": extras_all,
        }



    def generate(self, dataloader: DataLoader, **generate_kwargs):

        self.repeat_times = generate_kwargs.pop("repeat_times", 1)
        system_prompt = generate_kwargs.pop("system_prompt", None)

        grouped_responses = []

        for batch_data in dataloader:
            for data in batch_data:

                id = data['id']
                message = data['message']
                target = data['target']
                extra = data.get('extra', {}) or {}

                content = message[1]['content'] if system_prompt else message[0]['content']

                preds = []
                extras = []

                for repeat_idx in range(self.repeat_times):

                    msg = message
                    if system_prompt is not None:
                        msg = [{"role": "system", "content": system_prompt}] + msg

                    response = self.model.chat(messages=msg, **generate_kwargs)

                    if isinstance(response, str):
                        resp_text = response
                    else:
                        resp_text = getattr(response, "content", str(response))

                    preds.append(resp_text)
                    extras.append({"repeat_idx": repeat_idx})

                grouped_responses.append({
                    "id": id,
                    "content": content,
                    "target": target,
                    "pred": preds,
                    "extra": extras
                })

        return grouped_responses
