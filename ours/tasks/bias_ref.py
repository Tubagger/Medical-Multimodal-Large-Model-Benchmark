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

    def eval(self, responses: List[Dict[str, Any]]) -> Dict[str, Union[float, Sequence]]:
        print(responses)

        def _major_vote_and_ratio(votes: List[Any]):
            clean = [v for v in votes if v is not None and str(v).strip() != ""]
            if not clean:
                return None, None
            c = Counter(clean)
            top = c.most_common()
            best_cnt = top[0][1]
            # tie：稳定选字典序最小
            tied = sorted([k for k, v in top if v == best_cnt], key=lambda x: str(x))
            major = tied[0]
            ratio = best_cnt / len(clean)
            return major, ratio

        grouped = bool(responses) and isinstance(responses[0], list)

        if grouped:
            # ===== grouped: List[List[Dict]] =====
            preds = [[r.get("response") for r in group] for group in responses]
            labels = [[r.get("target") for r in group] for group in responses]
            extras = [[r.get("extra") for r in group] for group in responses]

            # 每组只输出一个 content：取该组第一个非空 content
            contents = []
            for group in responses:
                base_content = None
                for r in group:
                    if r.get("content") is not None:
                        base_content = r.get("content")
                        break
                contents.append(base_content)

            major_answer, major_ratio = [], []
            for group_preds in preds:
                ma, mr = _major_vote_and_ratio(group_preds)
                major_answer.append(ma)
                major_ratio.append(mr)

        else:
            # ===== non-grouped: List[Dict] =====
            preds = [r.get("response") for r in responses]
            labels = [r.get("target") for r in responses]
            extras = [r.get("extra") for r in responses]
            contents = [r.get("content") for r in responses]

            major_answer = preds
            major_ratio = [1.0 if p is not None and str(p).strip() != "" else None for p in preds]

        results: Dict[str, Union[float, Sequence]] = {}

        results.update(
            {
                "content": contents if any(contents) else None,
                "pred": preds if any(preds) else None,
                "major_answer": major_answer if any(major_answer) else None,
                "major_ratio": major_ratio if any(major_ratio) else None,
                "extra": extras if any(extras) else None,
            }
        )

        return results
    
    def generate(self, dataloader: DataLoader, **generate_kwargs):

        self.repeat_times = generate_kwargs.pop("repeat_times", 1)
        grouped_responses = []
        system_prompt = generate_kwargs.pop("system_prompt", None)

        for batch_data in dataloader:
            for data in batch_data:
                message = data['message']
                target = data['target']
                extra: Dict[str, Any] = data.get('extra', {}) or {}

                per_sample_responses = []

                for repeat_idx in range(self.repeat_times):
                    msg = message
                    if system_prompt is not None:
                        msg = [{"role": "system", "content": system_prompt}] + msg

                    response = self.model.chat(messages=msg, **generate_kwargs)

                    extra_out = dict(extra)
                    extra_out["repeat_idx"] = repeat_idx

                    # content 只取一次
                    content = msg[1]['content'] if system_prompt else msg[0]['content']
                    content_once = content if repeat_idx == 0 else None
                    
                    # ✅ 统一解析 response
                    if isinstance(response, str):
                        resp_text = response
                    else:
                        resp_text = getattr(response, "content", str(response))

                    per_sample_responses.append({
                        "content": content_once,     # ✅ 后续 repeat 不重复存 content
                        "response": resp_text,
                        "target": target,
                        "extra": extra_out,
                    })

                grouped_responses.append(per_sample_responses)

        return grouped_responses

