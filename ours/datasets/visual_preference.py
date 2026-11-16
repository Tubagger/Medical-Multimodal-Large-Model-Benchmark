"""Runtime loader for the review-pending F.1 distinct-image candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import yaml

from ours import ImageTxtSample, _OutputType
from ours.datasets.base import BaseDataset
from ours.methods.base import BaseMethod
from ours.utils.registry import registry


@registry.register_dataset()
class VisualPreference(BaseDataset):
    dataset_ids: Sequence[str] = ["visual-preference"]
    dataset_config: Optional[str] = "ours/configs/datasets/visual-preference.yaml"

    def __init__(
        self,
        dataset_id: str,
        model_id: str,
        method_hook: Optional[BaseMethod] = None,
        debug_mode: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(dataset_id, model_id, method_hook, debug_mode)
        config = yaml.safe_load(Path(self.dataset_config).read_text(encoding="utf-8"))
        labels_path = Path(config["annotation_file"])
        image_dir = Path(config["image_dir"])
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        limit = config.get("debug_nums", 2) if debug_mode else config.get("nums", len(labels))


        self.dataset = [
            ImageTxtSample(
                id=row["id"],
                image_path=str(image_dir / row["image"]),
                text=row["question"],
                target=row["answer"],
                extra={
                    "pair_id": row["pair_id"],
                    "order": row["order"],
                    "group": row["group"],
                    "source_left_id": row["source_left_id"],
                    "source_right_id": row["source_right_id"],
                },
            )
            for row in labels[:limit]
        ]

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
