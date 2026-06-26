from typing import Optional, Sequence, Dict, Any
from ours.datasets.base import BaseDataset
from ours.utils.registry import registry
from ours import ImageTxtSample, TxtSample
import yaml
import json,csv
import os
from itertools import islice

@registry.register_dataset()
class OODDataset(BaseDataset):

    dataset_ids: Sequence[str] = [
        "ood"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/ood.yaml"

    def __init__(
        self,
        dataset_id: str,
        model_id: str,
        method_hook=None,
        **kwargs
    ):
        super().__init__(dataset_id, model_id ,method_hook, **kwargs)

        # =========================
        # load yaml config
        # =========================
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = self.config.get('image_dir')
        self.label_dir = self.config.get('annotation_file')
        self.nums = self.config.get('nums')

        assert os.path.exists(self.image_dir), f"❌ Image directory not found: {self.image_dir}"
        assert os.path.exists(self.label_dir), f"❌ Label file not found: {self.label_dir}"

        with open(self.label_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)


        assert self.nums <= len(samples), f"❌ num ({self.nums}) > total samples ({len(samples)})"
        samples = samples[:self.nums]

        dataset = []

        # =========================
        # read cached results
        # =========================
        result_path = f"logs/robustness/r1-ood/{model_id}/{dataset_id}.json"
        processed_ids = set()

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_samples = results.get("per_sample_results", [])

                for item in processed_samples:
                    id = item["id"]
                    
                    processed_ids.add(id)
            print(f"✅ Loaded {len(processed_ids)} cached samples")

        # =========================
        # prompt template
        # =========================
        self.prompt_template = (
            "Question:\n{question}\n\n"
            "{options}\n\n"
            "(Please directly answer with the correct "
            "option letter without explanation.)"
        )

        # =========================
        # Build dataset
        # =========================

        dataset = []

        for sample in samples:

            id = sample['id']
            if id in processed_ids:
                continue

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

            dataset.append(
                ImageTxtSample(
                    id=id,
                    image_path=image_path,
                    text=prompt,
                    target=answer,
                )
            )

        self.dataset = dataset


    def __getitem__(self, idx: int) -> ImageTxtSample:
        sample = self.dataset[idx]
        if self.method_hook:
            return self.method_hook.run(sample)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)
