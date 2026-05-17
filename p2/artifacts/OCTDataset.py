import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms


class OCTDataset(Dataset):

    def __init__(self, image_path, mask_path, rsize=(416, 624), transform=None):
        super().__init__()
        self.img_files = glob.glob(os.path.join(image_path, '*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(mask_path, os.path.basename(img_path)))
        self.rsize     = rsize
        self.transform = transform

    def __getitem__(self, index):
        img_path  = self.img_files[index]
        mask_path = self.mask_files[index]
        image = plt.imread(img_path)
        mask  = plt.imread(mask_path)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        if len(image.shape) > 2:
            image = image[:, :, 0]
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        else:
            t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.rsize, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor()])
            image = t(image)
            mask  = t(mask)
        return image, mask

    def __len__(self):
        return len(self.img_files)


class EnhancedOCTDataset(OCTDataset):
    """OCTDataset con CLAHE opcional y ColorJitter sobre la imagen."""

    def __init__(self, image_path, mask_path, rsize=(416, 624), transform=None,
                 clip_limit=None, brightness=0.0, contrast=0.0):
        super().__init__(image_path, mask_path, rsize, transform)
        self.clahe = (cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                      if clip_limit else None)
        cj_active = brightness > 0 or contrast > 0
        self.color_jitter = (transforms.ColorJitter(brightness=brightness,
                                                    contrast=contrast)
                             if cj_active else None)

    def __getitem__(self, index):
        img_path  = self.img_files[index]
        mask_path = self.mask_files[index]
        image = plt.imread(img_path)
        mask  = plt.imread(mask_path)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        if len(image.shape) > 2:
            image = image[:, :, 0]
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        if self.clahe is not None:
            image = self.clahe.apply(image)

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        else:
            t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.rsize, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor()])
            image = t(image)
            mask  = t(mask)

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        return image, mask
