import os
from typing import List, Tuple
from os.path import join
import albumentations as alb
from torchvision.transforms import Normalize

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
import cv2


class MyDataset(Dataset):
    def __init__(self, data_path: str, mode: str) -> None:
        self._mode = mode
        mode_data = join(data_path, mode)
        self._A = join(mode_data, "A")
        self._B = join(mode_data, "B")
        self._label = join(mode_data, "label")
        self._list_images = self._read_images_list(data_path)

        if mode == 'train':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()

        if os.path.basename(data_path) == 'SYSU-CD':
            self._normalize_A = Normalize(mean=[0.39659573, 0.52846194, 0.46540028], std=[0.20213536, 0.15811189, 0.15296702])
            self._normalize_B = Normalize(mean=[0.40202362, 0.48766126, 0.39895686], std=[0.18235275, 0.15682769, 0.1543715]) 
        elif os.path.basename(data_path) == 'LEVIR-CD-processed' or os.path.basename(data_path)== '_LEVIR-CD':
            self._normalize_A = Normalize(mean=[0.45028868, 0.44673658, 0.38134101], std=[0.17450373, 0.16485656, 0.15314709])
            self._normalize_B = Normalize(mean=[0.34578751, 0.33841163, 0.28902323], std=[0.1292925,  0.12592704, 0.11870409])
        elif os.path.basename(data_path) == 'WHU-CD':
            # self._normalize_A = Normalize(mean=[0.49051637, 0.44972419, 0.39724921], std=[0.1474837, 0.14372663, 0.14352224])
            # self._normalize_A = Normalize(mean=[0.49726412, 0.49622711, 0.47865301], std=[0.18394293, 0.17459092, 0.18085868])
            self._normalize_A = Normalize(mean= [0.48334482, 0.4829116,  0.46035167], std=[0.17954556, 0.16984882, 0.17596323])
            self._normalize_B = Normalize(mean= [0.48522087, 0.44476541, 0.38734944],std=[0.14298548, 0.13841379, 0.13836961])
        elif os.path.basename(data_path) == 'CLCD' or  os.path.basename(data_path)=='_CLCD':
            self._normalize_A = Normalize(mean=[0.45028868, 0.44673658, 0.38134101], std=[0.17450373, 0.16485656, 0.15314709])
            self._normalize_B = Normalize(mean=[0.34578751, 0.33841163, 0.28902323], std=[0.1292925,  0.12592704, 0.11870409])
    def __getitem__(self, indx):
        imgname = self._list_images[indx].strip('\n')
        x_ref = cv2.imread(join(self._A, imgname + '.png'))
        x_test = cv2.imread(join(self._B, imgname + '.png'))
        x_mask = cv2.imread(join(self._label, imgname + '.png'), cv2.IMREAD_GRAYSCALE)
        
        if self._mode == "train":
            x_ref, x_test, x_mask = self._augment(x_ref, x_test, x_mask)

        x_ref, x_test, x_mask = self._to_tensors(x_ref, x_test, x_mask)
        return x_ref, x_test, x_mask, imgname

    def __len__(self):
        return len(self._list_images)
    
    def _to_tensors(self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        # Normalize pixel values to [0, 1] before converting to tensors
        x_ref = x_ref.astype(float) / 255.0
        x_test = x_test.astype(float) / 255.0
        x_mask = x_mask.astype(float) / 255.0  # Normalizing mask, if applicable

        return (
            self._normalize_A(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize_B(torch.tensor(x_test).permute(2, 0, 1)),
            torch.tensor(x_mask),
        )

    def _read_images_list(self, data_path: str) -> List[str]:
        images_list_file = join(data_path,'list', self._mode + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()

    def _augment(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # First apply augmentations in equal manner to test/ref/x_mask:
        transformed = self._augmentation(image=x_ref, image0=x_test, x_mask0=x_mask)
        x_ref = transformed["image"]
        x_test = transformed["image0"]
        x_mask = transformed["x_mask0"]

        # Then apply augmentation to single test ref in different way:
        x_ref = self._aberration(image=x_ref)["image"]
        x_test = self._aberration(image=x_test)["image"]

        return x_ref, x_test, x_mask


def _create_shared_augmentation():
    return alb.Compose(
        [
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(limit=30, p=0.5),
            alb.ShiftScaleRotate(rotate_limit=0,p=0.5)
        ],
        additional_targets={"image0": "image", "x_mask0": "mask"},
    )


def _create_aberration_augmentation():
    return alb.Compose([
        alb.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        alb.GaussianBlur(blur_limit=[3, 5], p=0.5),
    ])
