from typing import Optional, Sequence, Dict
from ours.datasets.base import BaseDataset
from ours.methods.base import BaseMethod
from ours.datasets.unrelatedimg import UnrelatedImageDataset
from ours.utils.registry import registry
from ours import ImageTxtSample, TxtSample,_OutputType
from PIL import Image
import yaml
import json
import random
import os,re
import warnings


@registry.register_dataset()
class MedReasonData(BaseDataset):

    dataset_ids: Sequence[str] = [
        "logical-reasoning",
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/logical-reasoning.yaml"
    

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, debug_mode:Optional[bool] = False, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook, debug_mode=debug_mode)

        # =========================
        # load config
        # =========================
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = self.config.get("image_dir", "")
        self.label_dir = self.config.get("annotation_file")
        if debug_mode:
            self.nums = self.config.get("debug_nums", None)
        else:
            self.nums = self.config.get("nums", None)

        assert os.path.exists(self.image_dir), f"Image dir not found: {self.image_dir}"
        assert os.path.exists(self.label_dir), f"Anno file not found: {self.annotation_file}"


        with open(self.label_dir, encoding="utf-8") as f:
            samples = json.load(f)

        # =========================
        # Intercept sample
        # =========================

        assert self.nums <= len(samples), (
            f"❌ num ({self.nums}) > total samples ({len(samples)})"
        )

        samples = samples[:self.nums]

        # =========================
        # prompt template
        # =========================

        self.prompt_template = (
            "Question:\n{question}\n\n"
            "{options}\n\n"
            "(Please directly answer with the correct option letter without explanation.)"
        )

        # =========================
        # Build dataset
        # =========================

        dataset = []

        for sample in samples:

            id = sample['id']

            # =========================
            # image
            # =========================

            image_name = sample["image"]
            image_path = os.path.join(
                self.image_dir,
                image_name
            )

            # =========================
            # question
            # =========================

            question = sample["question"]

            options = sample.get("options", "")

            prompt = self.prompt_template.format(
                question=question,
                options=options
            )

            # =========================
            # extract answer
            # =========================

            answer = sample.get("answer", "")
            extra = {
                "options": options
            }

            dataset.append(
                ImageTxtSample(
                    id=id,
                    image_path=image_path,
                    text=prompt,
                    target=answer,
                    extra=extra
                )
            )

        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
