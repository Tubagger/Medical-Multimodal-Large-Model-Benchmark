from typing import Any, Sequence, List, Tuple, Dict, Optional
from ours.evaluators.base import BaseEvaluator
from ours.utils.registry import registry
import string
import re, json, ast
from ours.evaluators.metrics import _supported_metrics
from ours.models import load_chatmodel


def parse_bbox(pred_str):
    """解析模型输出成 [xmin, ymin, xmax, ymax] 的 float 列表"""
    if isinstance(pred_str, (list, tuple)):
        return [float(x) for x in pred_str]

    s = str(pred_str).strip()
    # 去掉 markdown、三引号、空格
    s = re.sub(r"^```(?:json)?|```$", "", s)
    s = s.replace("'''", "").replace('"', '').replace("'", "").strip()

    # 尝试解析 JSON
    try:
        box = json.loads(s)
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return [float(x) for x in box]
    except Exception:
        pass

    # 尝试解析 Python 表达式
    try:
        box = ast.literal_eval(s)
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return [float(x) for x in box]
    except Exception:
        pass

    # 最后用正则匹配数字
    nums = re.findall(r"-?\d+\.?\d*", s)
    if len(nums) >= 4:
        return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]

    # 无法解析就返回空框
    return [0.0, 0.0, 0.0, 0.0]

@registry.register_evaluator()
class ChatModelBoxEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_bbox_eval']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from ours.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
        formatter = string.Formatter()
        self.prompt_template_fields = [fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]
    
    def get_prompt(self, pred, label, extra):
        prompt_params = {}
        for key in self.prompt_template_fields:
            if key == "pred":
                prompt_params[key] = pred
            elif key == "label":
                prompt_params[key] = label
            elif extra is not None and key in extra:
                prompt_params[key] = extra[key]
            else:
                raise KeyError("Fail to format the prompt. Can not find the specific field in pred/label/extra.")
        prompt = self.prompt_template.format(**prompt_params)
        return prompt

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        processed_labels = []
        for (pred, label, extra) in zip(preds, labels, extras):
            assert isinstance(pred, str)

            eval_prompt = self.get_prompt(pred, label, extra)
            
            eval_message = [
                                {
                                    "role": "user",
                                    "content": eval_prompt
                                }
                            ]
            eval_response = self.chatmodel.chat(messages=eval_message, **self.generation_kwargs)
            # print(eval_response.content)
            # 修改这一行
            bbox = parse_bbox(eval_response.content)
            processed_preds.append(bbox)
            processed_labels.append(label)


        return processed_preds, processed_labels, extras


@registry.register_evaluator()
class ChatModelDetection(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_privacy_score']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from ours.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
        formatter = string.Formatter()
        self.prompt_template_fields = [fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]
    

    def run_detector(self,user_prompt: str, rabbit_response: str) -> Tuple[float, str]:
        """
        输入:
            user_prompt: 用户输入
            rabbit_response: 模型输出（需要检测的内容）
        返回:
            (violation_score, reason)
        """

        detector_input = [
            {
                "role": "system",
                "content": self.prompt_template
            },
            {
                "role": "user",
                "content": (
                    "USER_PROMPT:\n"
                    f"{user_prompt}\n\n"
                    "RABBIT_RESPONSE:\n"
                    f"{rabbit_response}"
                )
            }
        ]

        # 调用你的模型代理
        raw = self.chatmodel.chat(messages=detector_input, **self.generation_kwargs)
        # print(f"raw{raw.content}")
        try:
            data = json.loads(raw.content)
            score = data.get("violation", 0.5)
            reason = data.get("reason", "")
        except Exception as e:
            score = 0.5
            reason = f"ParseError: {e}"
        return score,reason

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        processed_labels = []
        for (pred, label, extra) in zip(preds, labels, extras):
            score , reason = self.run_detector(extra,pred)

            processed_preds.append({"score": score, "reason": reason})

        return processed_preds, processed_labels, extras

@registry.register_evaluator()
class ChatModelEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_jailbreak_score']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from ours.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
        formatter = string.Formatter()
        self.prompt_template_fields = [fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]
    
    def parse_content_json(self, raw_content: str) -> dict:
        if raw_content is None:
            raise ValueError("raw.content is None")
        s = raw_content.strip()
        if not s:
            raise ValueError("raw.content is empty")

        # 去掉 ```json / ``` 围栏
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)

        return json.loads(s)

    def run_detector(self,user_prompt: str, rabbit_response: str) -> Tuple[float, str]:
        """
        输入:
            user_prompt: 用户输入
            rabbit_response: 模型输出（需要检测的内容）
        返回:
            (jailbreak_success, reason)
        """

        detector_input = [
            {
                "role": "system",
                "content": self.prompt_template
            },
            {
                "role": "user",
                "content": (
                    "USER_PROMPT:\n"
                    f"{user_prompt}\n\n"
                    "RABBIT_RESPONSE:\n"
                    f"{rabbit_response}"
                )
            }
        ]

        # 调用你的模型代理
        raw = self.chatmodel.chat(messages=detector_input, **self.generation_kwargs)
        # print(f"raw{raw.content}")
        try:
            data = self.parse_content_json(raw.content)
            jailbreak_success = data.get("jailbreak_success", 0)
            reason = data.get("reason", "")
        except Exception as e:
            jailbreak_success = 0
            reason = f"ParseError: {e}"
        return jailbreak_success,reason

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        processed_labels = []
        processed_extras = []
        for (pred, label, extra) in zip(preds, labels, extras):
            jailbreak_success , reason = self.run_detector(pred,label)
         
            processed_preds.append(jailbreak_success)
            processed_labels.append(extra['riskcategory'])
            processed_extras.append({
                "riskcategory": extra["riskcategory"],
                "reason": reason
            })

        return processed_preds, processed_labels,processed_extras
    

@registry.register_evaluator()
class AbstentionEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_abstention']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from ours.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
        formatter = string.Formatter()
        self.prompt_template_fields = [fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]

    def run_detector(self, rabbit_response: str) -> Tuple[float, str]:
        """
        输入:
            user_prompt: 用户输入
            rabbit_response: 模型输出（需要检测的内容）
        返回:
            (abstenion, reason)
        """

        detector_input = [
            {
                "role": "system",
                "content": self.prompt_template
            },
            {
                "role": "user",
                "content": (
                    "RABBIT_RESPONSE:"
                    f"{rabbit_response}"
                )
            }
        ]
        print("shuru=====:",detector_input)
        # 调用你的模型代理
        raw = self.chatmodel.chat(messages=detector_input, **self.generation_kwargs)
        try:
            data = json.loads(raw.content)
            abstention = data.get("abstention", 0)
            reason = data.get("reason", "")

        except Exception as e:

            abstention = 0
            reason = f"ParseError: {e}"

        return abstention,reason

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        processed_labels = []
        processed_extras = []
        for (pred, label, extra) in zip(preds, labels, extras):
            absteniton,reason = self.run_detector(preds)
            processed_preds.append(absteniton)
            processed_extras.append({"abstention":absteniton,"reason": reason})

        return processed_preds, processed_labels,processed_extras
    


@registry.register_evaluator()
class ToxictyScoreEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_toxicty_score']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from ours.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
        formatter = string.Formatter()
        self.prompt_template_fields = [fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]
    
    def parse_content_json(self, raw_content: str) -> dict:
        if raw_content is None:
            raise ValueError("raw.content is None")
        s = raw_content.strip()
        if not s:
            raise ValueError("raw.content is empty")

        # 去掉 ```json / ``` 围栏
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)

        return json.loads(s)

    def run_detector(self,user_prompt: str, rabbit_response: str) -> Tuple[float, str]:
        """
        输入:
            user_prompt: 用户输入
            rabbit_response: 模型输出（需要检测的内容）
        返回:
            (abstenion, reason)
        """

        detector_input = [
            {
                "role": "system",
                "content": self.prompt_template
            },
            {
                "role": "user",
                "content": (
                    "USER_PROMPT:\n"
                    f"{user_prompt}\n\n"
                    "RABBIT_RESPONSE:\n"
                    f"{rabbit_response}"
                )
            }
        ]

        # 调用你的模型代理
        raw = self.chatmodel.chat(messages=detector_input, **self.generation_kwargs)
        # print(f"raw{raw.content}")
        try:
            data = self.parse_content_json(raw.content)
            toxicity_score = data.get("toxicity_score", 0)
            reason = data.get("reason", "")
        except Exception as e:
            toxicity_score = 0
            reason = f"ParseError: {e}"

        return toxicity_score,reason

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        processed_labels = []
        processed_extras = []
        for (pred, label, extra) in zip(preds, labels, extras):
            toxicity_score,reason = self.run_detector(pred,label)
            print("toxicity_score",toxicity_score,reason)
            processed_preds.append(toxicity_score)
            processed_extras.append({"toxicity_score":toxicity_score,"reason": reason})

        return processed_preds, processed_labels, processed_extras
    
    def eval(self, preds, labels=None, extras=None, **kwargs):
        print("This is my custom eval")

        processed_preds, processed_labels, processed_extras = self.process(preds, labels, extras)

        results = {}

        # 自己写新的逻辑
        for metrics_id, cfg in self.metrics_cfg.items():
            metrics_fn = _supported_metrics[metrics_id]
            results[metrics_id] = metrics_fn(
                processed_preds, processed_labels, **cfg
            )

        return results,processed_extras
    
@registry.register_evaluator()
class UntargetAccuracyEvaluator(BaseEvaluator):

    evaluator_ids: List[str] = ['chat_untarget_accuracy']

    def __init__(
        self,
        evaluator_id: str,
        model_id: str,
        prompt_template: str = None,
        generation_kwargs: Dict[str, Any] = None,
        metrics_cfg: Dict[str, Any] = None,
        device: str = "cuda"
    ):
        super().__init__(evaluator_id, metrics_cfg)
        self.evaluator_id = evaluator_id

        self.chatmodel = load_chatmodel(model_id=model_id)

        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs

    # -------------------------
    # LLM judge
    # -------------------------
    def run_detector(self, ground_truth: str, rabbit_response: str):

        detector_input = [
            {
                "role": "system",
                "content": self.prompt_template
            },
            {
                "role": "user",
                "content": (
                    "GROUND_TRUTH:\n"
                    f"{ground_truth}\n\n"

                    "MODEL_PREDICTION:\n"
                    f"{rabbit_response}"
                )
            }
        ]

        raw = self.chatmodel.chat(
            messages=detector_input,
            **self.generation_kwargs
        )

        try:
            data = self.parse_content_json(raw.content)
            score = data.get("correct", 0)
            reason = data.get("reason", "")
        except Exception as e:
            score = 0
            reason = f"ParseError: {e}"

        return score, reason

    # -------------------------
    # process
    # -------------------------
    def process(
        self,
        preds: Sequence[Any],
        labels: Optional[Sequence[Any]] = None,
        extras: Optional[Sequence[Any]] = None,
        **kwargs
    ):
        processed_preds = []
        processed_labels = []
        processed_extras = []

        for pred, label, extra in zip(preds, labels, extras):

            # 用 LLM 判断语义是否一致
            success, reason = self.run_detector(
                ground_truth=label,        # GT
                rabbit_response=pred      # model output
            )

            processed_preds.append(success)
            processed_labels.append(1)  # 这里标签全设为1，表示都应该正确无误地回答

            processed_extras.append({
                "success": success,
                "pred": pred,
                "label": label,
                "reason": reason
            })

        return processed_preds, processed_labels, processed_extras
    
@registry.register_evaluator()
class TargetAccuracyEvaluator(BaseEvaluator):

    evaluator_ids: List[str] = ['chat_target_accuracy']

    def __init__(
        self,
        evaluator_id: str,
        model_id: str,
        prompt_template: str = None,
        generation_kwargs: Dict[str, Any] = None,
        metrics_cfg: Dict[str, Any] = None,
        device: str = "cuda"
    ):
        super().__init__(evaluator_id, metrics_cfg)
        self.evaluator_id = evaluator_id

        self.chatmodel = load_chatmodel(model_id=model_id)

        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs

    # -------------------------
    # LLM judge
    # -------------------------
    def run_detector(self, ground_truth: str, rabbit_response: str):

        detector_input = [
            {
                "role": "system",
                "content": self.prompt_template
            },
            {
                "role": "user",
                "content": (
                    "GROUND_TRUTH:\n"
                    f"{ground_truth}\n\n"

                    "MODEL_PREDICTION:\n"
                    f"{rabbit_response}"
                )
            }
        ]

        raw = self.chatmodel.chat(
            messages=detector_input,
            **self.generation_kwargs
        )

        try:
            data = self.parse_content_json(raw.content)
            score = data.get("correct", 0)
            reason = data.get("reason", "")
        except Exception as e:
            score = 0
            reason = f"ParseError: {e}"

        return score, reason

    # -------------------------
    # process
    # -------------------------
    def process(
        self,
        preds: Sequence[Any],
        labels: Optional[Sequence[Any]] = None,
        extras: Optional[Sequence[Any]] = None,
        **kwargs
    ):
        processed_preds = []
        processed_labels = []
        processed_extras = []

        for pred, label, extra in zip(preds, labels, extras):

            # 用 LLM 判断语义是否一致
            success, reason = self.run_detector(
                ground_truth=label,        # GT
                rabbit_response=pred      # model output
            )

            processed_preds.append(success)
            processed_labels.append(0)  # 这里标签全设为0，表示都应该回答错误

            processed_extras.append({
                "success": success,
                "pred": pred,
                "label": label,
                "reason": reason
            })

        return processed_preds, processed_labels, processed_extras