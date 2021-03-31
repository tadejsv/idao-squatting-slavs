from pathlib import Path
import re
from typing import Any, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch

IMG_SIZE = 576
IMG_FINAL_SIZE = 120

def load_image(fname):
    img = cv2.imread(str(fname), 0)
    slc =  slice(576 // 2 - IMG_FINAL_SIZE // 2, 576 // 2 + IMG_FINAL_SIZE // 2)
    return img[slc, slc]


class IDAOData(torch.utils.data.dataset.Dataset):
    """Loads images from IDAO train dataset in numpy format."""

    def __init__(self, folder: Union[Path, str], transform: Optional[Any] = None, *, is_train=True,):
        self.is_train = is_train
        image_files, classes = [], []
        for file in Path(folder).glob("**/*.png"):
            image_files.append(file)
            classes.append(self._get_classes(file) if is_train else ('NA', -1))

        self.image_files = image_files
        self.classes = classes
        self.transform = transform
        self.list_images = [load_image(self.image_files[i]) for i in range(len(self))]


    @staticmethod
    def _get_classes(file_name: Path) -> Tuple[int, int]:
        regex = re.search(r"_(ER|NR)_(\d{1,3})_", str(file_name))
        r_type = regex.group(1)
        energy = int(regex.group(2))

        return r_type, energy

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        image = self.list_images[index]
        if self.transform:
            image = self.transform(image=image)["image"]

        # Add extra channel dimension
        image = np.expand_dims(image, 0)

        r_type = int(self.classes[index][0] == "ER") if self.is_train else -1
        energy = self.classes[index][1]

        idx = str(self.image_files[index]).split('/')[-1][:-4]

        return image, r_type, energy, idx

    def __len__(self) -> int:
        return len(self.image_files)


class IDAODataTest(torch.utils.data.dataset.Dataset):
    """Loads images from IDAO public / private dataset in numpy format."""

    def __init__(self, folder: Union[Path, str], transform: Optional[Any] = None):

        image_files = []
        for file in Path(folder).glob("**/*.png"):
            image_files.append(file)

        self.image_files = image_files
        self.transform = transform
        self.list_images = [load_image(self.image_files[i]) for i in range(len(self))]

    def __getitem__(self, index: int) -> np.ndarray:
        image = self.list_images[index]
        # Remove path to image and fromat '.png'
        idx = str(self.image_files[index]).split('/')[-1][:-4]
        if self.transform:
            image = self.transform(image=image)["image"]

        # Add extra channel dimension
        image = np.expand_dims(image, 0)

        return idx, image

    def __len__(self) -> int:
        return len(self.image_files)


CENTER = 120
NORM_MEAN = 0.3938
NORM_STD = 0.15


def train_transforms(center=CENTER) -> Any:
    assert center <= IMG_FINAL_SIZE
    transforms = A.Compose(
        [
            A.CenterCrop(center, center),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

    return transforms


def val_transforms(center=CENTER) -> Any:
    assert center <= IMG_FINAL_SIZE
    transforms = A.Compose(
        [
            A.CenterCrop(center, center),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )

    return transforms
