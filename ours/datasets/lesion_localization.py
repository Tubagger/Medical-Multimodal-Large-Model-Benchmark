from typing import Optional, Sequence, Dict
from ours.datasets.base import BaseDataset
from ours.methods.base import BaseMethod
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType
from PIL import Image
import yaml
import json
import os
import warnings


@registry.register_dataset()
class LesionData(BaseDataset):
    
    dataset_ids: Sequence[str] = [
        "lesion-localization"
    ]
    dataset_config: Optional[Dict[str, str]] = {
        "lesion-localization":"./ours/configs/datasets/lesion-localization.yaml"
    }

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook)

        # =========================
        # load config
        # =========================
        with open(self.dataset_config[dataset_id]) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = self.config.get('image_dir')
        self.label_dir = self.config.get('annotation_file')
        self.nums = self.config.get('nums')

        assert os.path.exists(self.image_dir), f"❌ Image directory not found: {self.image_dir}"
        assert os.path.exists(self.label_dir), f"❌ Label file not found: {self.label_dir}"

        with open(self.label_dir, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        # =========================
        # Intercept sample
        # =========================
        assert self.nums <= len(samples), f"❌ num ({self.nums}) > total samples ({len(samples)})"
        samples = samples[:self.nums]

        # =========================
        # Read cached results
        # =========================
        result_path = f"logs/truthfulness/t1-basic/{model_id}/{dataset_id}.json"
        processed_ids = set()

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_samples = results.get("per_sample_results", [])
                for item in processed_samples:
                    id = item["id"]
                    
                    processed_ids.add(id)
            print(f"✅ Loaded {len(processed_ids)} cached samples")

        PROMPT ="""
            You are a medical imaging expert. The input image DEFINITELY contains at least one lesion.
            Your task is to locate it and describe it.
            Output requirements:
            1. You must return a JSON object with two fields:
            - "bbox": a JSON array of 4 normalized numbers [xmin, ymin, xmax, ymax], each in [0,1].
            - "finding": a short text description of the abnormality.
            2. Because the image is GUARANTEED to contain at least one lesion:
            - "bbox" MUST NOT be [0,0,0,0].
            3. Do NOT output pixel coordinates.
            4. Do NOT include explanations, markdown, comments, or any extra text.
            5. Output ONLY the JSON object. Nothing else.
            Required output structure (DO NOT copy the numbers):
            {"bbox": [xmin, ymin, xmax, ymax], "finding": "description"}
            """
        dataset = []
        for sample in samples:

            id = sample['id']
            if id in processed_ids:
                continue
            
            img_name = sample['image']
            image_path = os.path.join(self.image_dir, img_name)
            
            # image size
            with Image.open(image_path) as img:
                width, height = img.size
            bbox = sample['bounding_box']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            normalized_bbox = [xmin, ymin, xmax, ymax]

            extra = {
                "img_size": (width, height),
            }

            label = normalized_bbox

            dataset.append(
                ImageTxtSample(
                    id=id,
                    image_path=image_path,
                    text=PROMPT,
                    target=label,
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
    