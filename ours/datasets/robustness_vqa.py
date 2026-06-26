import random
from typing import Optional, Sequence, Dict, Any
from ours.datasets.base import BaseDataset
from ours.datasets.unrelatedimg import UnrelatedImageDataset
from ours.utils.registry import registry
from ours import ImageTxtSample, TxtSample
import yaml
import json
import os


@registry.register_dataset()
class RobustnessVQADataset(BaseDataset):

    dataset_ids: Sequence[str] = [
        "robustness-vqa-clean",

        "robustness-vqa-text-answer-flip",
        "robustness-vqa-text-question-negation",
        "robustness-vqa-text-option-expansion",
        "robustness-vqa-text-narrative-distraction",


        "robustness-vqa-answer-flip",
        "robustness-vqa-question-negation",
        "robustness-vqa-option-expansion",
        "robustness-vqa-narrative-distraction",

        "robustness-vqa-color",
        "robustness-vqa-nature",
        "robustness-vqa-noise"
    ]
    dataset_config: Optional[str] = "./ours/configs/datasets/robustness-vqa.yaml"

    def __init__(
        self,
        dataset_id: str,
        model_id: str,
        method_hook=None,
        **kwargs
    ):
        super().__init__(dataset_id, model_id, method_hook, **kwargs)

        # =========================
        # load yaml config
        # =========================
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = self.config.get('image_dir')
        self.label_dir = self.config.get('query_file')
        self.nums = self.config.get('nums')

        assert os.path.exists(self.image_dir), \
            f"❌ Image directory not found: {self.image_dir}"

        assert os.path.exists(self.label_dir), \
            f"❌ Label file not found: {self.label_dir}"

        # =========================
        # load annotation
        # =========================
        with open(self.label_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        # =========================
        # intercept samples
        # =========================
        assert self.nums <= len(samples), \
            f"❌ num ({self.nums}) > total samples ({len(samples)})"

        samples = samples[:self.nums]

        dataset = []

        # =========================
        # scan images
        # =========================
        self.images = []

        for img in os.listdir(self.image_dir):

            image_path = os.path.join(self.image_dir, img)

            self.images.append(image_path)

        # =========================
        # load cached results
        # =========================
        result_path = f"logs/robustness/r3-robustness-vqa/{model_id}/{dataset_id}.json"

        processed_ids = set()
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for item in results.get("per_sample_results", []):
                    processed_ids.add(item["id"])   # extra unique
            print(f"✅ Loaded {len(processed_ids)} processed samples")




        # =========================
        # prompt
        # =========================
        # PROMPT = """
        # Output only the text template so that I can directly copy-paste.
        # Do not provide additional commentary or explanation.
        # """

        # =========================
        # build dataset
        # =========================


        for sample in samples:
            
            id = sample['id']
            if id in processed_ids:
                    continue
            
            sample_text = self.build_input_text(sample)

            image_path = os.path.join(
                    self.image_dir,
                    sample['image']
                )
            label = sample['answer']
            # =========================
            # text-only sample
            # =========================
            if dataset_id == "robustness-vqa-text":

                dataset.append(
                    TxtSample(
                        id=id,
                        text=sample_text,
                        target=label,
                    )
                )
            elif dataset_id ==  "robustness-vqa-clean" or dataset_id == "robustness-vqa-answer-flip" or dataset_id == "robustness-vqa-question-negation" or dataset_id == "robustness-vqa-option-expansion" or dataset_id == "robustness-vqa-narrative-distraction":
                dataset.append(
                    ImageTxtSample(
                        id=id,
                        image_path=image_path,
                        text=sample_text,
                        target=label,
                    )
                )
            else:
                unrelated_dataset = UnrelatedImageDataset(model_id=self.model_id,dataset_id=self.dataset_id)
                unrelated_sample: ImageTxtSample = random.sample(unrelated_dataset.dataset, k=1)[0]
                dataset.append(
                    ImageTxtSample(
                        id=id,
                        image_path=unrelated_sample.image_path,
                        text=sample_text,
                        target=label,
                    )
                )


        self.dataset = dataset

    # ------------------------ #
    # build question prompt
    # ------------------------ #
    def build_input_text(self, item):

        question = item.get("question", "")
        options = item.get("options", {})

        # 保证 A/B/C/D 顺序稳定
        option_text = "\n".join([
            f"{k}. {str(v).strip()}"
            for k, v in sorted(options.items())
        ])

        prompt = (
            "Medical Visual Question Answering task.\n"
            "Question:\n"
            f"{question}\n"
            "Options:\n"
            f"{option_text}\n"
            "Instruction:\n"
            "Select the correct answer based on the image and question.\n"
            "Respond ONLY with the option letter(s). Do not output explanation."
        )

        return prompt
    def __getitem__(self, idx: int) -> ImageTxtSample:
        sample = self.dataset[idx]
        if self.method_hook:
            return self.method_hook.run(sample)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)
