from ours.datasets.base import BaseDataset, collate_fn
from ours.methods.base import BaseMethod
from ours import ImageTxtSample, _OutputType
from typing import Optional, Sequence
from ours.utils.registry import registry
from torch.utils.data import DataLoader
from natsort import natsorted
from glob import glob
import yaml
import os

@registry.register_dataset()
class UnrelatedImageDataset(BaseDataset):
    dataset_ids: Sequence[str] = ["color", "nature", "noise"]
    dataset_config: Optional[str] = "ours/configs/datasets/unrelatedimg.yaml"

    def __init__(self, model_id: str, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(model_id=model_id,dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir', '')
        assert os.path.exists(self.image_dir)

        data_type = dataset_id.split('-')[-1]
        self.dataset = [ImageTxtSample(image_path=path, text=None) for path in natsorted(glob(os.path.join(self.image_dir, f'*{data_type}*')))]

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    

