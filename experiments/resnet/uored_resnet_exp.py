import os
import sys
import glob
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
BASE_DRIVE = "../"
DATASET_FINAL = os.path.join(BASE_DRIVE, "dataset_final")
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = "uored_resnet_results.txt"

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

# --- TRANSFORMAÇÕES ---
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

# --- DATASET GENÉRICO (PARA PRÉ-TREINO) ---
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

# --- DATASET ESPECÍFICO UORED (GRUPOS VIRTUAIS) ---
# Mapeamento: Bearing X -> Classe de Falha Y
# 1-5: Class 48 | 6-10: Class 49 | 11-15: Class 50 | 16-20: Class 51
BEARING_FAULT_MAP = {}
for i in range(1, 6):   BEARING_FAULT_MAP[i] = 1 # Class 48
for i in range(6, 11):  BEARING_FAULT_MAP[i] = 2 # Class 49
for i in range(11, 16): BEARING_FAULT_MAP[i] = 3 # Class 50
for i in range(16, 21): BEARING_FAULT_MAP[i] = 4 # Class 51

class UORED_VirtualGroupDataset(Dataset):
    def __init__(self, root_dir, bearing_list, transform=None):
        self.transform = transform
        self.samples = []
        
        # O UORED geralmente está em: UORED/Bearing_X/Normal/*.png
        
        for b_id in bearing_list:
            b_path = os.path.join(root_dir, f"Bearing_{b_id}")
            if not os.path.exists(b_path):
                 # Tenta procurar sem o prefixo "Bearing_" se não achar
                 b_path = os.path.join(root_dir, str(b_id))
            
            if not os.path.exists(b_path): continue
            
            # Busca recursiva
            files = glob.glob(os.path.join(b_path, "**/*.png"), recursive=True)
            
            for f in files:
                fname = os.path.basename(f).lower()
                # Define Label
                if "normal" in fname:
                    label = 0
                else:
                    # Se não é normal, é a falha específica deste rolamento
                    label = BEARING_FAULT_MAP.get(b_id, -1)
                
                if label != -1:
                    self.samples.append((f, label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, label
        except: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label

# --- PRÉ-TREINO (SOURCE) ---
def load_dataset_data(ds_name, root_dir):
    ds_path = os.path.join(root_dir, ds_name)
    paths, labels = [], []
    label_map = {}
    if not os.path.exists(ds_path): return [], [], 0
    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                parts = root.split(os.sep)
                cls_name = parts[-1]
                if cls_name not in label_map: label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
    return paths, labels, len(label_map)

def train_vibnet_source_resnet(target_dataset="UORED"):
    print(f"\n>>> Treinando VibNet Source (ResNet-18) excluindo {target_dataset}...")
    all_datasets = ["CWRU_12k", "CWRU_48k", "HUST", "UORED", "PU"]
    sources = [d for d in all_datasets if d != target_dataset]
    
    all_paths, all_labels = [], []
    offset = 0
    for src in sources:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        if len(p) > 0:
            l_adjusted = [x + offset for x in l]
            all_paths.extend(p)
            all_labels.extend(l_adjusted)
            offset += n_cls
            print(f"    + {src}: {len(p)} imagens")

    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, offset)
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

    w_path = f"vibnet_resnet_source_no_{target_dataset}.pth"
    torch.save(model.state_dict(), w_path)
    return w_path

# --- EXPERIMENTO PRINCIPAL ---
def run_uored_resnet_experiment():
    print(f"\n{'#'*50}")
    print(f"INICIANDO EXPERIMENTO RESNET: UORED (Virtual Groups)")
    print(f"{'#'*50}")

    uored_root = os.path.join(DATASET_FINAL, "UORED")
    if not os.path.exists(uored_root):
        print("Dataset UORED não encontrado.")
        return

    # 1. Pesos VibNet
    weights_path = "vibnet_resnet_source_no_UORED.pth"
    if not os.path.exists(weights_path):
        weights_path = train_vibnet_source_resnet("UORED")
    else:
        print(f"Pesos VibNet encontrados: {weights_path}")

    # 2. Definição dos Grupos Virtuais
    # Grupos que garantem todas as classes (Normal + 4 Falhas) em cada fold
    VIRTUAL_GROUPS = [
        [1, 6, 11, 16],  # Group A
        [2, 7, 12, 17],  # Group B
        [3, 8, 13, 18],  # Group C
        [4, 9, 14, 19],  # Group D
        [5, 10, 15, 20]  # Group E
    ]
    GROUP_NAMES = ["Group_A", "Group_B", "Group_C", "Group_D", "Group_E"]

    strategies = ["Scratch", "ImageNet", "VibNet"]
    results = []

    # 3. Loop LODO-CV (Leave-One-Group-Out)
    for i, test_group_ids in enumerate(VIRTUAL_GROUPS):
        test_group_name = GROUP_NAMES[i]
        print(f"\n--- Fold: Testando em {test_group_name} {test_group_ids} ---")
        
        # Treino = Todos os rolamentos que NÃO estão no grupo de teste
        train_bearings = []
        for g in VIRTUAL_GROUPS:
            if g != test_group_ids:
                train_bearings.extend(g)
        
        # Datasets e Loaders
        train_ds = UORED_VirtualGroupDataset(uored_root, train_bearings, transform=data_transforms['train'])
        # Validação: Split 20% do dataset de treino
        tr_len = int(0.8 * len(train_ds))
        val_len = len(train_ds) - tr_len
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [tr_len, val_len])
        # Aplica transform de validação no set de validação (workaround simples ou manter train transform)
        
        test_ds = UORED_VirtualGroupDataset(uored_root, test_group_ids, transform=data_transforms['val'])

        if len(train_ds) == 0: 
            print("Erro: Dataset de treino vazio.")
            continue

        loaders = {
            'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(val_ds, batch_size=BATCH_SIZE),
            'test': DataLoader(test_ds, batch_size=BATCH_SIZE)
        }

        for strat in strategies:
            # Modelo ResNet-18
            model = models.resnet18(weights=None)
            n_feat = model.fc.in_features
            
            if strat == "ImageNet":
                temp = models.resnet18(weights='IMAGENET1K_V1')
                state = temp.state_dict()
                curr = model.state_dict()
                state = {k: v for k, v in state.items() if 'fc' not in k}
                curr.update(state)
                model.load_state_dict(curr)
                
            elif strat == "VibNet":
                state = torch.load(weights_path)
                curr = model.state_dict()
                pt_dict = {k: v for k, v in state.items() if k in curr and 'fc' not in k}
                curr.update(pt_dict)
                model.load_state_dict(curr)

            # 5 Classes (Normal + 4 Faults)
            model.fc = nn.Linear(n_feat, 5)
            model = model.to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3 if strat=="Scratch" else 1e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_f1 = 0
            best_w = copy.deepcopy(model.state_dict())

            # Fine-Tuning
            for ep in range(10):
                model.train()
                for x, y in loaders['train']:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for x, y in loaders['val']:
                        out = model(x.to(DEVICE))
                        _, p = torch.max(out, 1)
                        preds.extend(p.cpu().numpy()); targs.extend(y.numpy())
                
                # Evita erro se batch for pequeno e faltar classe
                try:
                    f1 = f1_score(targs, preds, average='macro')
                except: f1 = 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = copy.deepcopy(model.state_dict())

            # Teste
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
                "Test Condition": test_group_name,
                "Strategy": strat,
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1
            })

    # Relatório
    df = pd.DataFrame(results)
    print("\n\n" + "="*50)
    print("RELATÓRIO FINAL UORED - RESNET18")
    print("="*50)
    print(df.to_string())
    
    print("\n--- RESUMO ---")
    summary = df.groupby(["Strategy"])[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
    print(summary.to_string())

if __name__ == "__main__":
    run_uored_resnet_experiment()
