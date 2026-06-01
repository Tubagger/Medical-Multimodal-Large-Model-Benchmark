from torch.utils.data import DataLoader
from typing import Optional, Sequence
from ours.methods.base import BaseMethod
from ours.datasets.base import BaseDataset, collate_fn
from ours.methods import AdvGeneratedImage
from ours.utils.registry import registry
from ours import ImageTxtSample, _OutputType
import yaml
import os
import json

@registry.register_dataset()
class AdvTarget(BaseDataset):

    dataset_ids: Sequence[str] = ["adv-target"]

    def __init__(self, dataset_id: str, model_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, model_id=model_id, method_hook=method_hook)

        self.dataset_config: Optional[str] = "ours/configs/datasets/adv-target.yaml"

        with open(self.dataset_config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.generate_image = kwargs.get("generate_image", False)

        # ========= dirs =========
        self.image_dir = self.config.get("target_image_dir")
        self.annotation_file = self.config.get("annotation_file")
        self.nums = self.config.get("nums", None)

        assert os.path.exists(self.image_dir), f"Image dir not found: {self.image_dir}"
        assert os.path.exists(self.annotation_file), f"Anno file not found: {self.annotation_file}"

        # ========= load jsonl =========
        samples = self.read_jsonl(self.annotation_file)

        if self.nums is not None:
            samples = samples[:self.nums]

        # =========================
        # Read cached results
        # =========================
        result_path = f"logs/robustness/r2-adv-attack/{model_id}/{dataset_id}.json"
        processed_ids = set()

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_samples = results.get("per_sample_results", [])

                for item in processed_samples:
                    id = item["id"]
                    
                    processed_ids.add(id)
            print(f"✅ Loaded {len(processed_ids)} cached samples")

        PROMPT = """
            Given the image, identify the anatomical structure (organ, tissue, or region) and briefly describe its appearance.
        """

        dataset = []

        for sample in samples:
            id = sample['image_id']
            if id in processed_ids:
                continue

            image_name = f"{id}.png"
            image_path = os.path.join(self.image_dir, image_name)

            target = sample["target"][0] if isinstance(sample["target"], list) else sample["target"]

            dataset.append(
                ImageTxtSample(
                    id=id,
                    image_path=image_path,
                    text=PROMPT,
                    target=target   
                )
            )

        self.dataset = dataset

    # ========= jsonl reader =========
    def read_jsonl(self, json_file):
        data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    

if __name__ == '__main__':
    dataset = AdvTarget(dataset_id="adv-target")
    dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_fn)
    for data in dataloader:
        print(data)
        