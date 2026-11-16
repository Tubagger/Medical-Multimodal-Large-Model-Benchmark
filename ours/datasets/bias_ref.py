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


@registry.register_dataset()
class BiasRefData(BaseDataset):
    dataset_ids: Sequence[str] = [
        "bias-ref"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/bias-ref.yaml"

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, debug_mode:Optional[bool] = False, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook, debug_mode=debug_mode)

        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = self.config.get("image_dir")
        self.label_dir = self.config.get("query_file", None)
        if debug_mode:
            self.nums = self.config.get("debug_nums", None)
        else:
            self.nums = self.config.get("nums", None)

        with open(self.label_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        # =========================
        # Intercept sample
        # =========================
        assert self.nums <= len(samples), f"❌ num ({self.nums}) > total samples ({len(samples)})"
        samples = samples[:self.nums]


        # =========================
        # Build dataset
        # =========================

        dataset = []

        for sample in samples:

            id = sample['case_id']
            

            image_name = sample['image']
            image_path = os.path.join(self.image_dir, image_name)
            sample_text = self.build_input_text(sample)
            dataset.append(
                ImageTxtSample(
                    id=id,
                    text=sample_text,
                    image_path=image_path,
                )
            )

        self.dataset = dataset

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
            "**Instruction:** Respond ONLY with the option letter(s). Do not output explanation."
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
