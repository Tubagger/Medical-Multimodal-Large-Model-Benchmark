import random
from torch.utils.data import DataLoader
from typing import Dict, Optional, Sequence
from ours.methods.base import BaseMethod
from ours.datasets.base import BaseDataset, collate_fn

from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType
import yaml
import os
import json
import re

@registry.register_dataset()
class AdvTarget(BaseDataset):

    dataset_ids: Sequence[str] = [
        "adv-target-clean",
        "adv-target",
    ]
    
    dataset_config: Optional[Dict[str, str]] = { 
        "adv-target-clean": "ours/configs/datasets/adv-target-clean.yaml",
        "adv-target": "ours/configs/datasets/adv-target.yaml",
    }

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, debug_mode:Optional[bool] = False, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook, debug_mode=debug_mode)


        with open(self.dataset_config[dataset_id]) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.generate_image = kwargs.get("generate_image", False)

        # ========= dirs =========
        self.image_dir = self.config.get("image_dir")
        self.label_dir = self.config.get("annotation_file")
        if debug_mode:
            self.nums = self.config.get("debug_nums", None)
        else:
            self.nums = self.config.get("nums", None)

        assert os.path.exists(self.image_dir), f"Image dir not found: {self.image_dir}"
        assert os.path.exists(self.label_dir), f"Anno file not found: {self.label_dir}"

        with open(self.label_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)


        # =========================
        # Intercept sample
        # =========================
        assert self.nums <= len(samples), f"❌ num ({self.nums}) > total samples ({len(samples)})"
        samples = samples[:self.nums]

        # PROMPT = """
        #     Given the image, identify the anatomical structure (organ, tissue, or region) and briefly describe its appearance.
        # """

        dataset = []

        for sample in samples:
            id = sample['id']

            image_name = sample['image']
            image_path = os.path.join(self.image_dir, image_name)

            answer = sample["answer"]

            # ======================
            # prompt（模型输入）
            # ======================
            prompt = (
                f"Question:{sample['question']}\n\n"
                f"Options:{sample['options']}\n\n"
                f"Please choose the correct option.\n"
                f"Answer with the option letter (A, B, C, D, or E)."
            )


            dataset.append(
                ImageTxtSample(
                    id=id,
                    image_path=image_path,
                    text=prompt,
                    target=answer,
                    extra={
                        "r4_option_map": {
                            letter: value.strip()
                            for letter, value in re.findall(
                                r"(?m)^([A-E])\.\s*([^\n]+)$", sample["options"]
                            )
                        },
                        "r4_ground_truth_answer": sample["ground_truth_answer"],
                        "r4_attack_target_answer": sample["attack_target_answer"],
                    },
                )
            )

        self.dataset = dataset


    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
