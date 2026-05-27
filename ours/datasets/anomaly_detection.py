from typing import Optional, Sequence, List, Dict
from ours.datasets.base import BaseDataset
from ours.methods.base import BaseMethod
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType
import yaml
import json
import os


@registry.register_dataset()
class AnomalyData(BaseDataset):
    dataset_ids: Sequence[str] = [
        "anomaly-detection"
    ]
    
    dataset_config: Optional[Dict[str, str]] = { 
        "anomaly-detection": "./ours/configs/datasets/anomaly-detection.yaml"
    }

    # 可选：定义关键词 map 用于评估报告或文本匹配
    keyword_map = {
        "Atelectasis": ["atelectasis", "collapse"],
        "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
        "Consolidation": ["consolidation"],
        "Edema": ["edema", "fluid overload", "pulmonary edema"],
        "Pleural Effusion": ["effusion", "pleural effusion"],
        "Pneumonia": ["pneumonia", "infection"],
        "Pneumothorax": ["pneumothorax", "collapsed lung"],
        "Fracture": ["fracture", "broken rib"],
        "Support Devices": ["pacemaker", "tube", "catheter"],
    }

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook)

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
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_samples = results.get("per_sample_results", [])

                for item in processed_samples:
                    image_path = item["content"]["image_path"]
                    image_name = os.path.basename(image_path)
                    processed_images.add(image_name)

            print(f"✅ Loaded {len(processed_images)} cached samples")
        else:
            print("⚠️ No cached results found")


        # =========================
        # Define key name
        # =========================

        self.labels_columns = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
            "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]

        # =========================
        # question template
        # =========================

        PROMPT = """
            You are a medical classification model for CheXpert. 
            Given a chest X-ray image, classify the following 13 conditions.
            You MUST output a pure JSON object with exactly these keys, no explanations, no extra text:
            {
            "No Finding": 0 or 1,
            "Enlarged Cardiomediastinum": 0 or 1,
            "Cardiomegaly": 0 or 1,
            "Lung Opacity": 0 or 1,
            "Lung Lesion": 0 or 1,
            "Edema": 0 or 1,
            "Consolidation": 0 or 1,
            "Pneumonia": 0 or 1,
            "Atelectasis": 0 or 1,
            "Pneumothorax": 0 or 1,
            "Pleural Effusion": 0 or 1,
            "Pleural Other": 0 or 1,
            "Fracture": 0 or 1
            }
            Rules:
            1. Output strictly JSON only. No Markdown, no code block, no commentary.
            2. Each value MUST be either 0 or 1 (integers, not strings).
            3. If ANY disease key has value 1, then "No Finding" MUST be 0.
            4. If ALL disease keys are 0, then "No Finding" MUST be 1.
            Now produce the JSON result for this image:
        """

        # =========================
        # Build dataset
        # =========================
        dataset = []

        for sample in samples:
            img_name = sample['image_filename']

            if img_name in processed_images:
                continue

            image_path = os.path.join(self.image_dir, img_name)
            label = sample['labels']

            dataset.append(
                ImageTxtSample(
                    image_path=image_path,
                    text=PROMPT,
                    target=label
                )
            )


        self.dataset = dataset


    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
