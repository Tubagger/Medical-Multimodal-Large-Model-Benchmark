from typing import Optional, Sequence, Dict, Any
from ours.datasets.base import BaseDataset
from ours.utils.registry import registry
from ours import ImageTxtSample, TxtSample
import yaml
import json,csv
import os
from itertools import islice

@registry.register_dataset()
class PerturbedDataset(BaseDataset):

    dataset_ids: Sequence[str] = [
        "perturbed-data"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/perturbed-data.yaml"

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


        # =========================
        # read TSV
        # =========================
        with open(self.label_dir, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            samples = list(reader)

        assert self.nums <= len(samples), \
            f"❌ num ({self.nums}) > total samples ({len(samples)})"

        samples = samples[:self.nums]

        dataset = []

        # =========================
        # read cached results
        # =========================
        result_path = f"logs/truthfulness/r3-perturbed/{model_id}/{dataset_id}.json"
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
        PROMPT = """
        You are a clinical diagnostic assistant.
        Given the following multiple-choice question, select the single best answer.

        Rules:
        - Only output one option letter: A, B, C, D, or E.
        - Do not provide any explanation.
        - Do not output anything else.
        ### Question:
        {question}
        ### Options:
        A. {A}
        B. {B}
        C. {C}
        D. {D}
        E. {E}
        """

        # =========================
        # build dataset
        # =========================

        for sample in samples:

            # =========================
            # skip cached sample
            # =========================
            id = sample.get('index')
            if id in processed_ids:
                continue
            
            img_name = str(sample.get("index", "0")) + ".png"
            image_path = os.path.join(self.image_dir, img_name)

            # =========================
            # build prompt
            # =========================
            try:

                prompt = PROMPT.format(
                    question=sample.get("question", ""),
                    A=sample.get("A", ""),
                    B=sample.get("B", ""),
                    C=sample.get("C", ""),
                    D=sample.get("D", ""),
                    E=sample.get("E", ""),
                )

            except Exception as e:

                print(f"❌ [PROMPT ERROR] {e}")

                continue

            # =========================
            # append dataset
            # =========================
            dataset.append(
                ImageTxtSample(
                    id=id,
                    image_path=image_path,
                    text=prompt,
                    target=sample.get('answer', None),
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
