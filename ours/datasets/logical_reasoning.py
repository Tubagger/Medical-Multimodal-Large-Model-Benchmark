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
        "logical-reasoning-text",
        "logical-reasoning",
        "logical-reasoning-color",
        "logical-reasoning-nature",
        "logical-reasoning-noise"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/logical-reasoning.yaml"
    

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook)

        # =========================
        # load config
        # =========================
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = self.config.get("image_dir", "")
        self.label_dir = self.config.get("annotation_file")
        self.nums = self.config.get("nums")

        assert os.path.exists(
            self.label_dir
        ), f"Label file not found: {self.label_dir}"

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
        # Read cached results
        # =========================

        result_path = f"logs/truthfulness/t2-logic/{model_id}/{dataset_id}.json"
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

            answer_text = sample.get("answer", "")

            answer = None

            option_lines = options.split("\n")

            for line in option_lines:

                line = line.strip()

                match = re.match(
                    r"([A-Z])\.\s*(.*)",
                    line
                )

                if match:

                    option_letter = match.group(1)

                    option_content = match.group(2)

                    # answer中包含该选项内容
                    if option_content.lower() in answer_text.lower():

                        answer = option_letter

                        break

            # =========================
            # text only
            # =========================

            if dataset_id == "logical-reasoning-text":

                dataset.append(
                    TxtSample(
                        id=id,
                        text=prompt,
                        target=answer,
                    )
                )

            # =========================
            # image + text
            # =========================

            elif dataset_id == "logical-reasoning":

                dataset.append(
                    ImageTxtSample(
                        id=id,
                        image_path=image_path,
                        text=prompt,
                        target=answer,
                    )
                )

            # =========================
            # unrelated image attack
            # =========================

            else:

                unrelated_sample: ImageTxtSample = random.sample(
                    UnrelatedImageDataset.dataset,
                    k=1
                )[0]

                dataset.append(
                    ImageTxtSample(
                        id=id,
                        image_path=unrelated_sample.image_path,
                        text=prompt,
                        target=answer,
                    )
                )

        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
