from typing import Any, Counter, Optional, Sequence, List, Tuple, Dict
from ours.evaluators.base import BaseEvaluator
from ours.utils.registry import registry
import re
import numpy as np
import json

@registry.register_evaluator()
class CheXpertKeywordEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['rule_chexpert_eval']

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any], keyword_map: Dict[str, List[str]] = None) -> None:
        super().__init__(evaluator_id, metrics_cfg)
        # 定义每个CheXpert异常的关键词（可以自己扩展）
        self.keyword_map = keyword_map or {
            "No Finding": ["No Finding"],
            "Enlarged Cardiomediastinum": ["Enlarged Cardiomediastinum"],
            "Cardiomegaly": ["Cardiomegaly"],
            "Lung Opacity": ["Lung Opacity"],
            "Lung Lesion": ["Lung Lesion"],
            "Edema": ["Edema"],
            "Consolidation": ["Consolidation"],
            "Pneumonia": ["Pneumonia"],
            "Atelectasis": ["Atelectasis"],
            "Pneumothorax": ["Pneumothorax"],
            "Pleural Effusion": ["Pleural Effusion"],
            "Pleural Other": ["Pleural Other"],
            "Fracture": ["Fracture"]
        }

    def process(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any],
        extras: Sequence[Any]
        ):

        label_names = list(self.keyword_map.keys())

        y_pred = []
        y_true = []
        evals = []

        for pred, label_dict in zip(preds, labels):

            # ---------- parse prediction ----------
            pred_dict = {}

            if isinstance(pred, str):

                pred_clean = re.sub(
                    r"```json|```",
                    "",
                    pred
                ).strip()

                match = re.search(r"\{.*\}", pred_clean, re.S)

                if match:
                    pred_clean = match.group()

                try:
                    pred_dict = json.loads(pred_clean)

                except Exception as e:
                    print(f"⚠️ JSON parse error: {e}")

            # ---------- vectorize ----------
            pred_vec = np.array([
                max(int(pred_dict.get(disease, 0)), 0)
                for disease in label_names
            ])

            true_vec = np.array([
                max(int(label_dict.get(disease, 0)), 0)
                for disease in label_names
            ])

            y_pred.append(pred_vec)
            y_true.append(true_vec)
            evals.append({
                "pred": str(pred_vec.tolist()),
                "gt": str(true_vec.tolist()),
                "match": bool(np.array_equal(pred_vec, true_vec))
            })

        return np.array(y_pred), np.array(y_true), extras, evals



@registry.register_evaluator()
class BBoxEvaluator(BaseEvaluator):

    evaluator_ids: List[str] = ['rule_bbox_eval']

    def __init__(
        self,
        evaluator_id: str,
        metrics_cfg: Dict[str, Any],
    ) -> None:

        super().__init__(evaluator_id, metrics_cfg)

    def compute_iou(self, box1, box2):
        """
        box: [xmin, ymin, xmax, ymax]
        """

        if box1 is None or box2 is None:
            return 0.0

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)

        inter_area = inter_w * inter_h

        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def process(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any],
        extras: Sequence[Any]
    ) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:

        processed_preds = []
        evals = []

        for pred, label, extra in zip(preds, labels, extras):

            pred_bbox = None

            try:

                # =========================
                # 情况1: pred已经是dict
                # =========================
                if isinstance(pred, dict):

                    pred_bbox = pred.get("bbox", None)

                # =========================
                # 情况2: pred是json/string
                # =========================
                elif isinstance(pred, str):

                    # 去掉 ```json ```
                    pred_clean = re.sub(
                        r"```json|```",
                        "",
                        pred
                    ).strip()

                    # 尝试json解析
                    try:
                        pred_dict = json.loads(pred_clean)

                        if isinstance(pred_dict, dict):
                            pred_bbox = pred_dict.get("bbox", None)

                    except Exception:

                        # 如果不是合法json
                        # 用正则提取 bbox
                        match = re.search(
                            r'"bbox"\s*:\s*\[([^\]]+)\]',
                            pred_clean
                        )

                        if match:

                            bbox_str = match.group(1)

                            pred_bbox = [
                                float(x.strip())
                                for x in bbox_str.split(",")
                            ]

            except Exception as e:
                print(f"⚠️ bbox parse error: {e}")

            iou = self.compute_iou(pred_bbox, label)
            processed_preds.append(
                self.compute_iou(pred_bbox, label)
            )
            evals.append({
                'iou': str(iou),
                'match':bool(iou>0.5)
            })

        return processed_preds, labels, extras, evals

@registry.register_evaluator()
class GenericMCQEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ["rule_mcq_eval"]

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]) -> None:
        super().__init__(evaluator_id, metrics_cfg)
        # 最大支持到 Z（一般够用了）
        self.all_options = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def _infer_valid_options(self, extra: Any, default_max: int = 8) -> List[str]:
        """
        从 extras 推断本题有哪些选项。
        支持：
          - extra 是 dict,包含 {"options": ["A","B","C",...]}
          - extra 是 dict,包含 {"A": "...", "B": "..."} 这种字段
        否则：默认给 A..(A+default_max-1)
        """
        if isinstance(extra, dict):
            # 1) 显式给了 options
            opts = extra.get("options", None)
            if isinstance(opts, (list, tuple)) and all(isinstance(x, str) for x in opts):
                opts = [x.strip().upper() for x in opts if x and x.strip()]
                return [x for x in opts if x in self.all_options]

            # 2) extra 里有 A/B/C... 字段
            present = []
            for ch in self.all_options:
                if ch in extra and isinstance(extra[ch], str) and extra[ch].strip() != "":
                    present.append(ch)
            # 如果 E 为空也算一个选项？你可以按需要改：这里默认“字段存在就算选项”，即便为空
            if not present:
                for ch in self.all_options:
                    if ch in extra:
                        present.append(ch)
            if present:
                return present

        # 3) fallback：默认 A..H（可调）
        return self.all_options[:default_max]

    def _normalize_label(self, label: Any, valid_options: List[str]) -> Optional[str]:
        """
        label 可能是 "D" / "D." / "(D)" / "Answer: D"
        """
        if not isinstance(label, str):
            return None
        text = label.strip().upper()
        # 找第一个合法选项字母
        m = re.search(r"\b([A-Z])\b", text)
        if m and m.group(1) in valid_options:
            return m.group(1)
        # 兜底：取首字符
        if text and text[0] in valid_options:
            return text[0]
        return None

    def extract_option(self, text: Any, valid_options: List[str]) -> Optional[str]:
        """
        从模型输出中抽取选项字母。
        策略：
        1) 优先匹配 “ANSWER: X / FINAL: X / SELECT: X” 等强信号
        2) 其次匹配独立字母边界（避免把 'BARRETT' 里的 A 误判）
        3) 若输出包含多个字母（如 "A or C"），优先取强信号，否则取最后一次出现的独立选项
        """
        if not isinstance(text, str):
            return None
        t = text.strip().upper()
        if not t:
            return None

        # 1) 强信号：Answer/Final/Choice/Option 后面跟字母
        strong_patterns = [
            r"(?:FINAL|ANSWER|ANS|CHOICE|OPTION|SELECT(?:ED)?)\s*[:\-]?\s*[\(\[]?\s*([A-Z])\s*[\)\]]?",
        ]
        for p in strong_patterns:
            m = re.search(p, t)
            if m:
                ch = m.group(1)
                return ch if ch in valid_options else None

        # 2) 常见格式：(A) / A. / - A / [A]
        m = re.findall(r"[\(\[\{]?\s*([A-Z])\s*[\)\]\}]?(?:\s*[\.\:])?\b", t)
        # 过滤到 valid_options
        m = [ch for ch in m if ch in valid_options]
        if m:
            # 取最后一个出现的（很多模型最后会总结 “So the answer is D”）
            return m[-1]

        return None

    def process(self, preds: Sequence[Any], labels: Sequence[Any], extras: Sequence[Any]):
        y_pred = []
        y_true = []
        evals = []

        for pred, label, extra in zip(preds, labels, extras):
            valid_options = self._infer_valid_options(extra)

            true_opt = self._normalize_label(label, valid_options)
            pred_opt = self.extract_option(pred, valid_options)

            # 你原来用 y_true 全 1 的设计（把 accuracy 当成“是否命中”）是可以的
            # 这里保持一致：y_true 固定 1，y_pred 为 1/0
            y_true.append(1)
            y_pred.append(1 if (true_opt is not None and pred_opt == true_opt) else 0)
            evals.append({
                'pred_opt': str(pred_opt),
                'true_opt': str(true_opt),
                'match': bool(pred_opt == true_opt)
            
            })

        return np.array(y_pred), np.array(y_true), extras, evals
    

    
@registry.register_evaluator()
class PHIDetectionEvaluator(BaseEvaluator):

    evaluator_ids = ["phi_detection_eval"]

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]):
        super().__init__(evaluator_id, metrics_cfg)

    def _parse_pred_list(self, pred):
        """
        将模型输出解析成 list。
        支持：
        - 真实 list
        - JSON 字符串
        - 换行分割的字符串
        """
        if isinstance(pred, list):
            return pred

        if isinstance(pred, str):
            # 尝试 JSON 解析
            try:
                return json.loads(pred)
            except:
                # fallback：按行切
                return [x.strip() for x in pred.split("\n") if x.strip()]

        return []

    def process(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any] = None,
        extras: Sequence[Any] = None,
        **kwargs
    ) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        
        all_pred_binary = []
        all_label_binary = []
        evals = []

        for pred, label in zip(preds, labels):

            pred_list = self._parse_pred_list(pred)
            label_list = label  # 本来就是真实 PHI list

            pred_set = set(pred_list)
            label_set = set(label_list)

            # === 处理 TP/FN: 每个真实 PHI 是否被检出 ===
            for item in label_list:
                all_label_binary.append(1)  # 每条 label 都是真 PHI
                all_pred_binary.append(1 if item in pred_set else 0)

            # === 处理 FP: 模型多报的 PHI ===
            for item in pred_list:
                if item not in label_set:
                    all_label_binary.append(0)  # 非真实 PHI
                    all_pred_binary.append(1)  # 模型误报
            
            evals.append({
                "pred": str(pred_list),
                "gt": str(label_list),
                "match": pred_set == label_set
            })

        return all_pred_binary, all_label_binary, extras, evals


@registry.register_evaluator()
class InfoFlowPrivacyEvaluator(BaseEvaluator):
    """
    Evaluator for detecting PHI in 'information-stream' style dataset.
    Compares model predicted PHI list with ground truth PHI list.
    Calculates per-PHI true positives, false negatives, and false positives.
    """

    evaluator_ids = ["infoflow_privacy_eval"]

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]):
        super().__init__(evaluator_id, metrics_cfg)

    def _parse_pred(self, pred: Any) -> str:
        """Normalize model output to one of allowed/conditional/prohibited"""
        if isinstance(pred, str):
            # Remove whitespace, quotes, markdown
            s = pred.strip().lower().replace('`','').replace('"','')
            # Map variations to standard categories
            mapping = {
                "allowed": "allowed",
                "allow": "allowed",
                "yes": "allowed",
                "conditional": "conditional",
                "conditionally allowed": "conditional",
                "prohibited": "prohibited",
                "forbidden": "prohibited",
                "no": "prohibited",
            }
            return mapping.get(s, None)
        return None

    def process(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any] = None,
        extras: Sequence[Any] = None,
        **kwargs
    ) -> Tuple[List[str], List[str], List[Any]]:
        """
        Returns:
            processed_preds: list of normalized predictions (strings)
            processed_labels: list of ground truth labels (strings)
            extras: unchanged
        """
        processed_preds = [self._parse_pred(p) for p in preds]
        processed_labels = [self._parse_pred(l) for l in labels] if labels is not None else None
        return processed_preds, processed_labels, extras
    

@registry.register_evaluator()
class PrivacyRecognitionEvaluator(BaseEvaluator):

    evaluator_ids = ["privacy-recognition-eval"]

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]):
        super().__init__(evaluator_id, metrics_cfg)

    # -------------------------------
    # 将模型输出规范化为 0 / 1 / 2
    # -------------------------------
    def _to_three_class(self, x: Any) -> int:

        # -------- 数值 --------
        if isinstance(x, (int, float)):
            val = int(x)
            if val in [0, 1, 2]:
                return val
            # 超范围兜底
            return 0

        # -------- 字符串 --------
        if isinstance(x, str):
            s = x.strip().lower()

            # ---------- clean ----------
            if s in [
                "0", "clean", "no text", "no watermark",
                "no phi", "no phi detected"
            ]:
                return 0

            # ---------- PHI ----------
            if s in [
                "1", "phi", "phi exists", "contains phi",
                "yes", "true"
            ]:
                return 1

            # ---------- noise ----------
            if s in [
                "2", "noise", "random text", "non-phi",
                "irrelevant text"
            ]:
                return 2

            # ---------- 尝试数字 ----------
            try:
                val = int(float(s))
                if val in [0, 1, 2]:
                    return val
            except:
                pass

        # 默认：clean（最保守）
        return 0

    # -------------------------------
    # 主评估流程
    # -------------------------------
    def process(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any] = None,
        extras: Sequence[Any] = None,
        **kwargs
    ) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:

        y_true = []
        y_pred = []
        evals = []

        for pred, gt in zip(preds, labels):
            y_true.append(int(gt))  # 0 / 1 / 2
            y_pred.append(self._to_three_class(pred))
            evals.append({
                "pred": str(int(gt)),
                "gt": str(self._to_three_class(pred)),
                "match": bool(int(gt)==self._to_three_class(pred))
            })

        return y_pred, y_true, extras, evals
    

@registry.register_evaluator()
class PrivacyInflowEvaluator(BaseEvaluator):

    evaluator_ids = ["privacy-inflow-eval"]

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]):
        super().__init__(evaluator_id, metrics_cfg)

    def _to_three_class(self, x: Any) -> int:

        if isinstance(x, (int, float)):
            val = int(x)
            return val if val in [0, 1, 2] else 2

        if isinstance(x, str):
            s = x.strip().lower()

            if len(s) > 0:
                first = s[0]
                if first in ["a", "b", "c"]:
                    return {"a": 0, "b": 1, "c": 2}[first]

            return 2

        return 2
    # -------------------------------
    # 主评估流程
    # -------------------------------
    def process(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any] = None,
        extras: Sequence[Any] = None,
        **kwargs
    ) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:

        y_true = []
        y_pred = []
        evals = []

        for pred, gt in zip(preds, labels):

            # GT 也是字符串 → 转换
            if isinstance(gt, str):
                gt = gt.strip().lower()
                if gt == "allow":
                    gt_val = 0
                elif gt == "not_allow":
                    gt_val = 1
                elif gt == "ambiguous":
                    gt_val = 2
                else:
                    gt_val = 2
            else:
                gt_val = int(gt)

            y_true.append(gt_val)
            y_pred.append(self._to_three_class(pred))
            evals.append({
                "pred": str(gt_val),
                "gt": str(self._to_three_class(pred)),
                "match": bool(gt_val==self._to_three_class(pred))
            })

        return y_pred, y_true, extras, evals
