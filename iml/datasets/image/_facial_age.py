import typing as t

import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

__all__ = ["FacialAgeDataset"]


class FacialAgeDataset(ImageFolder):

    def __init__(
            self,
            regression: bool = True,
            target_dtype: t.Union[np.floating] = np.float32
    ):
        self.regression = regression
        self.target_dtype = target_dtype

        super(FacialAgeDataset, self).__init__(
            root="D:\\torchvision\\face_age",
            transform=ToTensor(),
            target_transform=self.target_dtype if self.regression else None
        )
