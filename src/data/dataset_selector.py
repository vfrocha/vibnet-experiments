from src.data.dataset import CWRUDataset, VibnetDataset

class DatasetSelector:
    def __init__(self, dataset_name, path, **kwargs):
        self.dataset_name = dataset_name
        self.path = path
        self.dataset = self._select_dataset(**kwargs)

    def _select_dataset(self, **kwargs):
        dataset_name = self.dataset_name.lower()
        if dataset_name == 'cwru':
            return CWRUDataset(path=self.path, **kwargs)
        elif dataset_name == 'vibnet':
            return VibnetDataset(path=self.path, **kwargs)
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
        
    def get_dataset(self):
        return self.dataset

    def get_k_fold_train_val_tuple(self, k):
        return self.dataset.get_k_fold_train_val_tuple(k)

    def __len__(self):
        return len(self.dataset)