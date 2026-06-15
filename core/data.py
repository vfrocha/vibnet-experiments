import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .config import IMG_SIZE, DATASET_FINAL, BATCH_SIZE

class VibDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_list[idx]).convert("RGB")
            if self.transform: 
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e: 
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), self.labels[idx]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_dataset_data(ds_name, root_dir):
    """Varre um diretório de dataset e retorna caminhos e labels."""
    ds_path = os.path.join(root_dir, ds_name)
    paths, labels, label_map = [], [], {}
    
    if not os.path.exists(ds_path): 
        return [], [], 0
        
    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                cls_name = root.split(os.sep)[-1]
                if cls_name not in label_map: 
                    label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
                
    return paths, labels, len(label_map)

def load_source_data(target_to_exclude):
    """
    Carrega e unifica todas as bases de dados, excluindo a base alvo.
    """
    # Garante que target_to_exclude seja uma lista (útil para o CWRU unificado)
    if isinstance(target_to_exclude, str):
        target_to_exclude = [target_to_exclude]

    all_datasets = ["CWRU_12k", "CWRU_48k", "HUST", "PU", "UORED"]
    sources = [d for d in all_datasets if d not in target_to_exclude]

    all_paths, all_labels = [], []
    offset = 0

    for src in sources:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        if len(p) > 0:
            all_paths.extend(p)
            all_labels.extend([x + offset for x in l])
            offset += n_cls

    return all_paths, all_labels, offset


def get_target_splits(dataset_name, test_condition, predefined_class_map=None):
    """
    Isola os dados para LODO-CV.
    Aceita test_condition como string ou lista de strings (para grupos virtuais).
    Aceita um mapa de classes predefinido (essencial para UORED).
    """
    ds_root = os.path.join(DATASET_FINAL, dataset_name)
    conditions = sorted([d for d in os.listdir(ds_root) if os.path.isdir(os.path.join(ds_root, d))])
    
    # 1. MAPEAMENTO DE CLASSES (Usa o predefinido ou procura em TODAS as pastas)
    if predefined_class_map:
        cls_map = predefined_class_map
    else:
        all_classes = set()
        for c in conditions:
            c_path = os.path.join(ds_root, c)
            all_classes.update([d for d in os.listdir(c_path) if os.path.isdir(os.path.join(c_path, d))])
        cls_map = {c: i for i, c in enumerate(sorted(list(all_classes)))}

    # 2. FLEXIBILIDADE PARA GRUPOS
    if isinstance(test_condition, str):
        test_condition = [test_condition]

    train_x, train_y, test_x, test_y = [], [], [], []

    # 3. SEPARAÇÃO DE TREINO E TESTE
    for cond in conditions:
        c_path = os.path.join(ds_root, cond)
        
        # Verifica se a pasta atual começa com QUALQUER UM dos nomes no grupo de teste
        is_test = any(cond.startswith(tc) for tc in test_condition)
        
        for cls_name, cls_idx in cls_map.items():
            p_cls = os.path.join(c_path, cls_name)
            if not os.path.exists(p_cls): continue
            
            files = [os.path.join(p_cls, f) for f in os.listdir(p_cls) if f.endswith('.png')]
            if is_test:
                test_x.extend(files)
                test_y.extend([cls_idx] * len(files))
            else:
                train_x.extend(files)
                train_y.extend([cls_idx] * len(files))

    return train_x, train_y, test_x, test_y, len(cls_map)


def get_dataloaders(train_x, train_y, test_x, test_y):
    """
    Separa 20% do treino para validação (Early Stopping) e constrói os DataLoaders.
    """
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)

    loaders = {
        'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
        'test': DataLoader(VibDataset(test_x, test_y, data_transforms['val']), batch_size=BATCH_SIZE)
    }
    return loaders
