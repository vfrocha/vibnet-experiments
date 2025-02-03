import torch
from torchvision.transforms import v2

from src.data.dataset.custom_dataset import CustomDataset

class DatasetInterface:
    name = None
    """
    Dataset for patches images. Arranges the patches in train and test sets considering the parent image (prefix of the image name). It has a train and test dataset that can be accessed by the attributes train_dataset and test_dataset.
    These datasets can be used in the DataLoader class from PyTorch.
    """	
    def __init__(self, path, train_size=0.8, k_folds=5, train_transform=None, test_transform=None) -> None:
        self.path = path
        self.k_folds = k_folds
        self.train_size = train_size
        self.train_transform = train_transform if train_transform is not None else v2.Compose([v2.ToImage()])
        self.test_transform = test_transform if train_transform is not None else v2.Compose([v2.ToImage()])
        self.labels_names = dict()
        self.train_dataset = None
        self.test_dataset = None
        self.k_folds_dataset = None
        self.folds_df = None

    def _train_test_split(self) -> tuple[tuple[list[str], list[int], list[str], list[int]]]:
        pass
    
    def get_k_fold_train_val_tuple(self, k) -> tuple[CustomDataset, CustomDataset]:
        """
        Get K-fold train and validation dataset
        """
        train_dataset = None
        val_dataset = None

        return train_dataset, val_dataset

    def __len__(self) -> int:
        return 0