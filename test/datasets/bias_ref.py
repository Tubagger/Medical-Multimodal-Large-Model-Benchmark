from ours.datasets.bias_ref import BiasRefData
from torch.utils.data import DataLoader
from ours.datasets.base import BaseDataset, collate_fn

if __name__ == '__main__':
    dataset = BiasRefData(dataset_id="bias-ref")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for data in dataloader:
        print(data)
        print("=================================================")