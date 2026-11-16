"""Experimental paired P.3 inputs, independent of legacy P.3 caches."""

import json
from pathlib import Path

import yaml

from ours import ImageTxtSample
from ours.datasets.base import BaseDataset
from ours.utils.registry import registry


@registry.register_dataset()
class PrivacyInference(BaseDataset):
    dataset_ids = ['privacy-inference']
    dataset_config = 'ours/configs/datasets/privacy-inference.yaml'

    def __init__(self, dataset_id, model_id, method_hook=None, debug_mode=False, **kwargs):
        super().__init__(dataset_id, model_id, method_hook, debug_mode)
        config = yaml.safe_load(Path(self.dataset_config).read_text())
        labels = [json.loads(line) for line in Path(config['annotation_file']).read_text().splitlines() if line.strip()]
        limit = config['debug_nums'] if debug_mode else config['nums']
        assert len(labels) == len({r['id'] for r in labels}) == 240
        self.dataset = []
        for row in labels[:limit]:
            image = Path(config['image_dir']) / row['image']
            assert image.is_file(), image
            self.dataset.append(ImageTxtSample(
                id=row['id'], image_path=str(image), text=row['question'],
                target=row['expected_behavior'],
                extra={key: row[key] for key in ('base_id', 'category_id', 'attribute_id',
                    'evidence_condition', 'request_context', 'review_status')},
            ))

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.method_hook.run(sample) if self.method_hook else sample

    def __len__(self):
        return len(self.dataset)
