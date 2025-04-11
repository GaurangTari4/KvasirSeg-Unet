import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

def load_images_from_folder(folder, target_size=(256, 256), grayscale=False):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, target_size)
            if grayscale:
                img = img / 255.0
            images.append(img)
    return np.array(images)

def load_dataset(image_folder, mask_folder, target_size=(256, 256)):
    images = load_images_from_folder(image_folder, target_size)
    masks = load_images_from_folder(mask_folder, target_size, grayscale=True)
    return images, masks

class KvasirSegDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.tensor(image).permute(2, 0, 1)
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
