from typing import List

from torch.utils.data import Dataset


class FilteredMNIST(Dataset):
    def __init__(self, base_dataset: Dataset, allowed_labels: List[int]) -> None:
        self.base_dataset = base_dataset
        self.allowed_labels = allowed_labels
        self.label_map = {label: i for i, label in enumerate(allowed_labels)}
        self.indices = []

        for idx in range(len(base_dataset)):
            _, y = base_dataset[idx]
            if int(y) in self.label_map:
                self.indices.append(idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[self.indices[idx]]
        return x, self.label_map[int(y)]