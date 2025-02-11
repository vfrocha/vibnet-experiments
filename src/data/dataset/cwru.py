from src.data.dataset.custom_dataset import CustomDataset
from src.data.dataset.dataset_interface import DatasetInterface

from src.data.dataset.augmentation import get_train_augmentation,get_test_augmentation
import numpy as np
import os

class CWRUDataset(DatasetInterface):
    name = "cwru"

    def __init__(self, path, train_size=0.8, k_folds=5, train_transform=None, test_transform=None) -> None:
        self.path = path
        self.k_folds = 4
        self.train_size = train_size

        self.train_transform = train_transform if train_transform is not None else get_train_augmentation()
        
        self.test_transform = test_transform if train_transform is not None else get_test_augmentation()

        self.labels_names = {0: "", 1: "",2:"",3:""}

    def _train_test_split(self) -> tuple[tuple[list[str], list[int], list[str], list[int]]]:
        pass

    def _get_fold_files(self,k):

        classes = [0,1,2,3]
        classesReturn = []
        for c in classes:
            images = list(self.path.glob(f"fold{k}/{c}/*.png"))
            if len(images) == 0:
                #print(f"Warning: No images found for class {c} in fold {k}. Skipping.")
                continue  # Skip this class and move to the next one
            classesReturn.append((c,images))

        if len(classesReturn) == 0:
            raise(ValueError("Invalid Folders Probably"))

        return classesReturn

    
    def get_k_fold_tuple(self, k) -> tuple[CustomDataset, CustomDataset, CustomDataset]:
        """
        Get K-fold train and validation dataset
        """
        train_imgs = []
        train_labels = []

        val_imgs = []
        val_labels = []

        test_imgs = []
        test_labels = []
        for i in range(4):
            if i != k:
                labelX_imgs = self._get_fold_files(i)
                
                for class_number,image_paths in labelX_imgs:
                    train_imgs += image_paths
                    train_labels += list(np.full(len(image_paths),class_number))

            if i == (k+1) % 4:
                labelX_imgs = self._get_fold_files(i)
                
                for class_number,image_paths in labelX_imgs:
                    val_imgs += image_paths
                    val_labels += list(np.full(len(image_paths),class_number))
            
            if i == k:

                labelX_imgs = self._get_fold_files(i)
                
                for class_number,image_paths in labelX_imgs:
                    test_imgs += image_paths
                    test_labels += list(np.full(len(image_paths),class_number))


        train_dataset = CustomDataset(train_imgs, train_labels, transform=self.train_transform)
        test_dataset = CustomDataset(test_imgs, test_labels, transform=self.train_transform)
        val_dataset =  CustomDataset(val_imgs, val_labels, transform=self.train_transform)

        return train_dataset, val_dataset, test_dataset

    def __len__(self) -> int:
        return 0