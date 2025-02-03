from src.data.dataset.custom_dataset import CustomDataset
from src.data.dataset.dataset_interface import DatasetInterface

from src.data.dataset.augmentation import get_train_augmentation,get_test_augmentation

class CWRUDataset(DatasetInterface):
    name = "cwru"

    def __init__(self, path, train_size=0.8, k_folds=5, train_transform=None, test_transform=None) -> None:
        self.path = path
        self.k_folds = k_folds
        self.train_size = train_size

        self.train_transform = train_transform if train_transform is not None else get_train_augmentation()
        
        self.test_transform = test_transform if train_transform is not None else get_test_augmentation()

        self.labels_names = {0: 'noncarcinoma', 1: 'carcinoma'}
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