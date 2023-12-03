# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:14:33 2023

@author: SABARI
"""

import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, image_paths, targets, resize=None, augmentations=None, mean=None, std=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        # Make sure image is in RGB format (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        targets = self.targets[item]

        if self.resize is not None:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))
        
        # Normalize using mean and std
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalized_img = (image / 255.0 - mean) / std

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        
        # Convert the NumPy array to a PyTorch tensor
        #image = transforms.ToTensor()(image)
        # Normalize the image using mean and std
        #if self.mean is not None and self.std is not None:
        #    image = transforms.functional.normalize(image, mean=self.mean, std=self.std)

        # You can use transforms.ToTensor() to convert the image to a PyTorch tensor
        image = transforms.ToTensor()(normalized_img)

        # Note that for classification tasks, targets should be of type long
        targets = torch.tensor(targets, dtype=torch.long)

        return {"image": image, "targets": targets}

