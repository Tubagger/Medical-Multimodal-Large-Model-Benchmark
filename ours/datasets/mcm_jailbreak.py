"""S.2 MCM: optimized image and the exact optimized attack prompt."""

import json
from pathlib import Path

import yaml

from ours import ImageTxtSample
from ours.datasets.base import BaseDataset
from ours.utils.registry import registry


@registry.register_dataset()
class McmJailbreak(BaseDataset):
    dataset_ids = ['mcm-jailbreak']
    dataset_config = 'ours/configs/datasets/mcm-jailbreak.yaml'

    def __init__(self, dataset_id, model_id, method_hook=None, debug_mode=False, **kwargs):
        super().__init__(dataset_id, model_id, method_hook, debug_mode)
        config = yaml.safe_load(Path(self.dataset_config).read_text(encoding='utf-8'))
        rows = json.loads(Path(config['annotation_file']).read_text(encoding='utf-8'))
        limit = config['debug_nums'] if debug_mode else config['nums']
        self.dataset = []
        for row in rows[:limit]:
            image = Path(config['image_dir']) / row['image']
            assert image.is_file(), image
            assert row['attack_suffix'].strip(), row['id']
            assert row['attack_prompt'] == row['question'] + row['attack_suffix']
            self.dataset.append(ImageTxtSample(
                id=row['id'], image_path=str(image), text=row['attack_prompt'],
                extra={'class': row['class'], 'protocol': 'mcm-both-v1'},
            ))

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.method_hook.run(sample) if self.method_hook else sample

    def __len__(self):
        return len(self.dataset)
