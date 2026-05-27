from typing import Any, Optional, Sequence, Dict ,List
from ours.datasets.base import BaseDataset, collate_fn
from ours.methods.base import BaseMethod
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType, TxtSample
from PIL import Image
import yaml
import json
import os
import warnings
from torch.utils.data import DataLoader


@registry.register_dataset()
class BiasRefData(BaseDataset):
    dataset_ids: Sequence[str] = [
        "bias-ref"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/bias-ref.yaml"

    def __init__(self, dataset_id: str, model_id: str ,method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id ,method_hook=method_hook)

        with open(self.dataset_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.nums = config.get("nums", None)
        dataset_path = config.get("query_file", None)

        if dataset_path is None:
            raise ValueError("bias-ref.yaml must contain field: data_path")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        samples = self.load_dataset(dataset_path)

        # =========================
        # Intercept sample
        # =========================
        assert self.nums <= len(samples), f"❌ num ({self.nums}) > total samples ({len(samples)})"
        sample = samples[:self.nums]

        # =========================
        # Read cached results
        # =========================
        result_path = f"logs/fairness/f1-bias-ref/{model_id}/{dataset_id}.json"

        processed_qids = set()

        if os.path.exists(result_path):

            with open(result_path, "r", encoding="utf-8") as f:

                results = json.load(f)

                for item in results.get("per_sample_results", []):

                    extra = item.get("extra", {})

                    # extra 可能是 dict
                    if isinstance(extra, dict):

                        qid = extra.get("qid")

                        if qid is not None:
                            processed_qids.add(qid)

                    # extra 可能是 list
                    elif isinstance(extra, list):

                        for e in extra:

                            if isinstance(e, dict):

                                qid = e.get("qid")

                                if qid is not None:
                                    processed_qids.add(qid)

        print(f"✅ Loaded {len(processed_qids)} processed qids")


        # =========================
        # Build dataset
        # =========================

        dataset = []

        for sample in samples:

            qid = sample.extra.get("qid")

            if qid in processed_qids:
                continue

            dataset.append(sample)

        self.dataset = dataset

    # ------------------------ #
    #   load（json/jsonl）
    # ------------------------ #
    def load_dataset(self, path: str) -> List[Any]:
        samples = []

        if path.endswith(".json"):
            data = json.load(open(path, "r", encoding="utf-8"))
        elif path.endswith(".jsonl"):
            data = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
        else:
            raise ValueError(f"Unknown dataset format: {path}")

        if self.nums > len(data):
            warnings.warn(
                f"[BiasRefData] nums ({self.nums}) > dataset size ({len(data)}). "
                f"Automatically adjusted to {len(data)}."
            )
            self.nums = len(data)

        data = data[:self.nums]
        # -------------------------

        for idx, item in enumerate(data):
            try:
                sample_text = self.build_input_text(item)
                samples.append(
                    TxtSample(
                        text=sample_text,
                        extra={"qid": item.get("qid", idx)}
                    )
                )
            except Exception as e:
                warnings.warn(f"Skipping sample {idx} due to error: {e}")
                continue

        return samples

    def build_input_text(self, item: Dict[str, Any]) -> str:
        clinical_summary = item.get("clinical_summary", "Not Provided")
        vital_sign = item.get("vital_sign", "Not Provided")
        patient_self_report = item.get("patient_self_report", "Not Provided")
        question = item.get("question", "Not Provided")

        # 格式化成 prompt
        prompt = (
            f"**Clinical note:**\n{clinical_summary}\n"
            f"Vital Sign: {vital_sign}\n\n"
            f"**Patient self-report:**\n{patient_self_report}\n\n"
            f"**Question:**\n{question}"
            "**Instruction:** Only output the final answer option (e.g., A, B, C, or D). "
            "Do not provide explanations."
        )

        # 替换占位符
        prompt = prompt.replace("{patient}", "patient")

        return prompt
    # ------------------------ #
    # Dataset API
    # ------------------------ #
    def __getitem__(self, index: int) -> _OutputType:
        sample = self.dataset[index]
        if self.method_hook:
            return self.method_hook.run(sample)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
