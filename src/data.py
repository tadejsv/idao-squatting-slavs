from pathlib import Path
import re
from typing import Any, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch


class IDAOData(torch.utils.data.dataset.Dataset):
    """Loads images from IDAO train dataset in numpy format."""

    def __init__(self, folder: Union[Path, str], transform: Optional[Any] = None):

        image_files, classes = [], []
        for file in Path(folder).glob("**/*.png"):
            image_files.append(file)
            classes.append(self._get_classes(file))

        self.image_files = image_files
        self.classes = classes
        self.transform = transform

    @staticmethod
    def _get_classes(file_name: Path) -> Tuple[int, int]:
        regex = re.search(r"_(ER|NR)_(\d{1,3})_", str(file_name))
        r_type = regex.group(1)
        energy = int(regex.group(2))

        return r_type, energy

    def __getitem__(self, index: int) -> Tuple[np.array, int, int]:
        image = cv2.imread(str(self.image_files[index]), 0)
        if self.transform:
            image = self.transform(image=image)["image"]

        r_type = int(self.classes[index][0] == "NR")
        energy = self.classes[index][1]

        return image, r_type, energy

    def __len__(self) -> int:
        return len(self.image_files)


NORM_MEAN = 0.3938
NORM_STD = 0.015


def train_transforms() -> Any:
    transforms = A.Compose(
        [
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

    return transforms


def val_transforms() -> Any:
    transforms = A.Compose(
        [
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

    return transforms
