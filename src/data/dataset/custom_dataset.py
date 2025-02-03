import torch
from PIL import Image
import cv2

import os

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset, loading image from disk so it is not necessary to load all images in memory
    """	
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if not os.path.exists(self.image_paths[idx]):
            raise FileNotFoundError(f"File {self.image_paths[idx]} not found")
            
        image = Image.open(self.image_paths[idx]).convert("RGB")        
        image = self.transform(image)

        labels = self.labels[idx]
        return image, int(labels)