import torch
from torch.utils.data import Dataset

__all__ = ["XYDataset"]


class XYDataset(Dataset):

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        xi = self.x[idx, ...]
        yi = self.y[idx, ...]

        return xi, yi
