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


    def template_format(self, anno: Dict) -> ImageTxtSample:
        """
        Convert a single annotation dict into ImageTxtSample
        """
        image_path = os.path.join(self.image_dir, anno['image_filename'])
        
        # image size
        with Image.open(image_path) as img:
            width, height = img.size

        prompt = (
            "You are a medical imaging expert. The input image DEFINITELY contains at least one lesion. "
            "Your task is to locate it and describe it."

            "Output requirements:"
            "1. You must return a JSON object with two fields:"
            "- \"bbox\": a JSON array of 4 normalized numbers [xmin, ymin, xmax, ymax], each in [0,1]."
            "- \"finding\": a short text description of the abnormality."
            "2. Because the image is GUARANTEED to contain at least one lesion:"
            "- \"bbox\" MUST NOT be [0,0,0,0]."
            "3. Do NOT output pixel coordinates."
            "4. Do NOT include explanations, markdown, comments, or any extra text."
            "5. Output ONLY the JSON object. Nothing else."

            "Required output structure (DO NOT copy the numbers):"
            "{\"bbox\": [xmin, ymin, xmax, ymax], \"finding\": \"description\"}"
        )


        bbox = anno.get("bounding_box", [0, 0, 0, 0])
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        normalized_bbox = [xmin, ymin, xmax, ymax]

        extra = {
            "img_size": (width, height),
        }

        label = normalized_bbox

        return ImageTxtSample(image_path=image_path, text=prompt, target=label, extra=extra)

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
            samples = json.load(f)['samples']

        # =========================
        # Intercept sample
        # =========================
        assert self.nums <= len(samples), f"❌ num ({self.nums}) > total samples ({len(samples)})"
        samples = samples[:self.nums]

        # =========================
        # Read cached results
        # =========================
        result_path = f"logs/truthfulness/t1-basic/{model_id}/{dataset_id}.json"
        processed_images = set()

        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    processed_samples = results.get("per_sample_results", [])

                    for item in processed_samples:
                        image_path = item["content"]["image_path"]
                        image_name = os.path.basename(image_path)
                        processed_images.add(image_name)

                print(f"✅ Loaded {len(processed_images)} cached samples")
            except:
                print("⚠️ Failed to load cache, ignoring")
        else:
            print("⚠️ No cached results found")

        dataset = []
        for sample in samples:
            datasample = self.template_format(sample)

            image_name = os.path.basename(datasample.image_path)

            if image_name in processed_images:

                continue

            if os.path.exists(datasample.image_path):
                dataset.append(datasample)
            else:
                warnings.warn(f"file: {datasample.image_path} not exists!")


        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    