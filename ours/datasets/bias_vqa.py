import random
from typing import Any, List, Optional, Sequence, Dict
from ours.datasets.base import BaseDataset
from ours.datasets.unrelatedimg import UnrelatedImageDataset
from ours.methods.base import BaseMethod
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType, TxtSample
from PIL import Image
import yaml
import json
import os
import warnings


@registry.register_dataset()
class BiasData(BaseDataset):
    dataset_ids: Sequence[str] = [
        "bias-vqa-text",
        "bias-vqa",
        "bias-vqa-color",
        "bias-vqa-nature",
        "bias-vqa-noise"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/bias-vqa.yaml"

    def __init__(self, dataset_id: str, model_id: str,method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id,method_hook=method_hook)

        with open(self.dataset_config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.image_dir = self.config.get('image_dir')
        self.query_dir = self.config.get('query_file')
        self.nums = self.config.get('nums')
        
        assert os.path.exists(self.image_dir), f"❌ Image directory not found: {self.image_dir}"
        assert os.path.exists(self.query_dir ), f"❌ query file not found: {self.query_dir}"
        
        with open(self.query_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)


        samples = samples[:self.nums]

        # =========================
        # Read cached results
        # =========================
        result_path = f"logs/fairness/f1-bias-ref/{model_id}/{dataset_id}.json"
        processed_ids = set()

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for item in results.get("per_sample_results", []):
                    processed_ids.add(item["qid"])   # extra unique
            print(f"✅ Loaded {len(processed_ids)} processed samples")
        else:
            print("⚠️ No cached results found")


        dataset = []
        for idx, item in enumerate(samples):

            sample_text = self.build_input_text(item)
            ids = item['case_id']
            if ids in processed_ids:
                    continue
            if self.dataset_id == "bias-vqa-text":
                dataset.append(
                    TxtSample(
                        text=sample_text,
                        target=item.get("answer", None),
                        extra={"qid": item.get("qid", idx)}
                    )
                )
            elif self.dataset_id == "bias-vqa":
                image_path = os.path.join(
                    self.image_dir,
                    item['image']
                )
                dataset.append(
                    ImageTxtSample(
                        image_path=image_path,
                        text=sample_text,
                        target=item.get("answer", None),
                        extra={"qid": item.get("qid", idx)}
                    )
                )

            else:
                unrelated_sample: ImageTxtSample = random.sample(UnrelatedImageDataset.dataset, k=1)[0]
                dataset.append(
                    ImageTxtSample(
                        image_path=unrelated_sample.image_path,
                        text=sample_text,
                        target=item.get("answer", None),
                        extra={"qid": item.get("qid", idx)}
                    )
                )


        self.dataset = dataset

    # ------------------------ #
    # build question prompt
    # ------------------------ #
    def build_input_text(self, item: Dict[str, Any]) -> str:
        clinical_summary = item.get("clinical_summary", "Not Provided")
        vital_sign = item.get("vital_sign", "Not Provided")
        patient_self_report = item.get("patient_self-report", "Not Provided")
        question = item.get("question", "Not Provided")

        # 格式化成 prompt
        prompt = (
            f"**Clinical note:**\n{clinical_summary}\n"
            f"Vital Sign: {vital_sign}\n"
            f"**Patient self-report:**\n{patient_self_report}\n"
            f"**Question:**\n{question}\n"
            "**Instruction:** Only output the final answer option (e.g., A, B, C, or D). "
            "Do not provide explanations."
        )

        # 替换占位符
        # prompt = prompt.replace("{patient}", "patient")

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




