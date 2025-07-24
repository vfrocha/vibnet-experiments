from sklearn.model_selection import train_test_split
from src.data.dataset.custom_dataset import CustomDataset
from src.data.dataset.dataset_interface import DatasetInterface

from src.data.dataset.augmentation import get_train_augmentation,get_test_augmentation
import numpy as np
import os


class VibnetDataset(DatasetInterface):
    name = "vibnet"

    def __init__(self, path, train_size=0.8, k_folds=5, train_transform=None, test_transform=None) -> None:
        self.path = path
        self.train_size = train_size
        self.k_folds = 1

        self.train_transform = train_transform if train_transform is not None else get_train_augmentation()
        self.test_transform = test_transform if test_transform is not None else get_test_augmentation()

        #self.labels_names = {0: "", 1: "", 2: "", 3: ""}
        self.labels_names = {0:"Normal",1:"Inner",2:"Outer",3:"Ball",4:"Cage"}

    def _get_all_files(self):
        classes = [i for i in range(len(self.labels_names))]
        all_files = []
        all_labels = []
        for c in classes:
            images = list(self.path.glob(f"{c}/*.png"))
            if len(images) == 0:
                continue  # Skip this class if no images are found
            all_files.extend(images)
            all_labels.extend([c] * len(images))  # Add labels for each image

        if len(all_files) == 0:
            raise ValueError("No images found in the dataset.")

        return all_files, all_labels

    def get_k_fold_tuple(self, k=None) -> tuple[CustomDataset, CustomDataset, CustomDataset]:
        """
        Get train, validation, and test datasets for the Vibnet dataset.
        """
        all_files, all_labels = self._get_all_files()

        # First, split into train+val and test sets
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            all_files, 
            all_labels, 
            test_size=0.1, 
            random_state=42, 
            stratify=all_labels
        )

        # Then, split train+val into train and validation sets
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, 
            train_val_labels, 
            test_size=0.1/ (1 - 0.1), 
            random_state=42, 
            stratify=train_val_labels
        )

        # Create datasets
        train_dataset = CustomDataset(train_files, train_labels, transform=self.train_transform)
        val_dataset = CustomDataset(val_files, val_labels, transform=self.test_transform)
        test_dataset = CustomDataset(test_files, test_labels, transform=self.test_transform)

        return train_dataset, val_dataset, test_dataset

    def __len__(self) -> int:
        return 0