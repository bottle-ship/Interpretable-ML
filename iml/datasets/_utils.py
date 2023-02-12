import typing as t

import numpy as np
from torchvision.datasets import VisionDataset
from torch.utils.data.sampler import SubsetRandomSampler

__all__ = ["get_train_test_sampler"]


def get_train_test_sampler(
        dataset: VisionDataset,
        test_size: float = 0.3
) -> t.Tuple[SubsetRandomSampler, SubsetRandomSampler]:
    n_samples = len(dataset)
    indices = list(range(0, n_samples))
    np.random.shuffle(indices)
    n_test = int(np.floor(test_size * n_samples))
    train_idx, valid_idx = indices[n_test:], indices[:n_test]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler
