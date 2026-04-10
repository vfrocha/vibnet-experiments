import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

# --- CONFIGURAÇÕES ---
BASE_DRIVE = os.getcwd()
DATASET_FINAL = os.path.join(BASE_DRIVE, "dataset_final")
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = "uored_results.txt"

# --- LOGGER ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(OUTPUT_FILE)

# --- DATASET ---
class VibDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_list[idx]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
        except: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), self.labels[idx]

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

# --- MAPEAMENTO DE ROLAMENTOS PARA GRUPOS VIRTUAIS ---
# Agrupamos rolamentos diferentes para garantir que cada 'Fold' tenha todas as classes
VIRTUAL_CONDITIONS = {
    "Group_A": ["Bearing_1", "Bearing_6", "Bearing_11", "Bearing_16"],
    "Group_B": ["Bearing_2", "Bearing_7", "Bearing_12", "Bearing_17"],
    "Group_C": ["Bearing_3", "Bearing_8", "Bearing_13", "Bearing_18"],
    "Group_D": ["Bearing_4", "Bearing_9", "Bearing_14", "Bearing_19"],
    "Group_E": ["Bearing_5", "Bearing_10", "Bearing_15", "Bearing_20"]
}

# --- FUNÇÕES DE CARREGAMENTO ---
def load_dataset_data(ds_name, root_dir):
    ds_path = os.path.join(root_dir, ds_name)
    paths, labels = [], []
    # Mapa fixo para UORED para garantir consistência
    # Ajuste os IDs conforme o seu 'Class_XX'
    label_map = {
        "Class_Normal": 0, "Class_47": 1, "Class_48": 2, 
        "Class_49": 3, "Class_50": 4, "Class_51": 5
    }
    
    if not os.path.exists(ds_path): return [], [], 0

    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                # Estrutura: UORED/Bearing_X/Class_Y/img.png
                parts = root.split(os.sep)
                cls_name = parts[-1]
                
                if cls_name in label_map:
                    paths.append(os.path.join(root, file))
                    labels.append(label_map[cls_name])
    
    return paths, labels, len(label_map)

# Função auxiliar para pegar dados de um Grupo Virtual específico
def get_data_from_virtual_group(uored_root, group_name, class_map):
    bearings_in_group = VIRTUAL_CONDITIONS[group_name]
    paths, labels = [], []
    
    for bearing in bearings_in_group:
        b_path = os.path.join(uored_root, bearing)
        if not os.path.exists(b_path): continue
        
        for cls_name in os.listdir(b_path):
            if cls_name not in class_map: continue
            
            cls_path = os.path.join(b_path, cls_name)
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.png')]
            
            paths.extend(files)
            labels.extend([class_map[cls_name]] * len(files))
            
    return paths, labels

# --- ETAPA 1: TREINO FONTE (Source) ---
def train_vibnet_source(target_dataset="UORED"):
    print(f"\n>>> Treinando VibNet Source (Excluindo {target_dataset})...")
    
    # Treina em CWRU, HUST, PU
    sources = ["CWRU_12k", "CWRU_48k", "HUST", "PU"]
    
    all_paths, all_labels = [], []
    offset = 0

    for src in sources:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        if len(p) > 0:
            # Shift nos labels para não sobrepor classes de datasets diferentes no pré-treino
            l_adjusted = [x + offset for x in l]
            all_paths.extend(p)
            all_labels.extend(l_adjusted)
            offset += n_cls
            print(f"    + {src}: {len(p)} imagens")

    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)
    
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, offset)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Source Ep {epoch+1}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Ep {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    w_path = "vibnet_source_no_uored.pth"
    torch.save(model.state_dict(), w_path)
    return w_path

# --- ETAPA 2: EXPERIMENTO ALVO (UORED) ---
def run_uored_experiment():
    print(f"\n{'#'*50}")
    print(f"INICIANDO EXPERIMENTO: UORED (Virtual Groups)")
    print(f"{'#'*50}")

    uored_root = os.path.join(DATASET_FINAL, "UORED")
    if not os.path.exists(uored_root):
        print("Dataset UORED não encontrado.")
        return

    # 1. Pesos Fonte
    weights_path = "vibnet_source_no_uored.pth"
    if not os.path.exists(weights_path):
        weights_path = train_vibnet_source()
    else:
        print(f"Pesos pré-treinados encontrados: {weights_path}")

    # Mapa de classes local do UORED
    class_map = {
        "Class_Normal": 0, "Class_47": 1, "Class_48": 2, 
        "Class_49": 3, "Class_50": 4, "Class_51": 5
    }
    
    strategies = ["Scratch", "ImageNet", "VibNet"]
    results = []
    
    groups = list(VIRTUAL_CONDITIONS.keys()) # ['Group_A', 'Group_B', ...]

    # LODO-CV Loop (Leave-One-GROUP-Out)
    for test_group in groups:
        train_groups = [g for g in groups if g != test_group]
        
        print(f"\n--- Fold: Testando em {test_group} ({VIRTUAL_CONDITIONS[test_group]}) ---")

        # Preparar dados de Treino (Juntando os grupos de treino)
        train_x, train_y = [], []
        for g in train_groups:
            p, l = get_data_from_virtual_group(uored_root, g, class_map)
            train_x.extend(p)
            train_y.extend(l)

        # Preparar dados de Teste (Grupo isolado)
        test_x, test_y = get_data_from_virtual_group(uored_root, test_group, class_map)
        
        # Validação (20% do treino)
        if len(train_x) == 0: 
            print("AVISO: Dados de treino vazios. Verifique nomes das pastas.")
            continue
            
        tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)

        loaders = {
            'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
            'test': DataLoader(VibDataset(test_x, test_y, data_transforms['val']), batch_size=BATCH_SIZE)
        }

        # Rodar Estratégias
        for strat in strategies:
            model = models.densenet121(weights=None)
            n_feat = model.classifier.in_features
            
            if strat == "ImageNet":
                temp = models.densenet121(weights='IMAGENET1K_V1')
                model.features = temp.features
            elif strat == "VibNet":
                state = torch.load(weights_path)
                curr = model.state_dict()
                pt_dict = {k: v for k, v in state.items() if k in curr and 'classifier' not in k}
                curr.update(pt_dict)
                model.load_state_dict(curr)

            model.classifier = nn.Linear(n_feat, len(class_map))
            model = model.to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3 if strat=="Scratch" else 1e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_f1 = 0
            best_w = copy.deepcopy(model.state_dict())

            # Fine-Tuning
            for ep in range(8):
                model.train()
                for x, y in loaders['train']:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                
                # Validação
                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for x, y in loaders['val']:
                        out = model(x.to(DEVICE))
                        _, p = torch.max(out, 1)
                        preds.extend(p.cpu().numpy()); targs.extend(y.numpy())
                f1 = f1_score(targs, preds, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = copy.deepcopy(model.state_dict())

            # Teste Final
            model.load_state_dict(best_w)
            model.eval()
            final_p, final_t = [], []
            with torch.no_grad():
                for x, y in loaders['test']:
                    _, p = torch.max(model(x.to(DEVICE)), 1)
                    final_p.extend(p.cpu().numpy()); final_t.extend(y.numpy())
            
            bal_acc = balanced_accuracy_score(final_t, final_p)
            macro_f1 = f1_score(final_t, final_p, average='macro')
            
            print(f"   [{strat}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f}")
            results.append({
                "Dataset": "UORED",
                "Test Condition": test_group,
                "Strategy": strat,
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1
            })

    return results

if __name__ == "__main__":
    results = run_uored_experiment()
    if results:
        df = pd.DataFrame(results)
        print("\n\n" + "="*50)
        print("RELATÓRIO FINAL UORED")
        print("="*50)
        print(df.to_string())
        
        print("\n--- RESUMO ---")
        summary = df.groupby(["Dataset", "Strategy"])[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
        print(summary.to_string())
