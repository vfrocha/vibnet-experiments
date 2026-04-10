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
BASE_DRIVE = "../"
DATASET_FINAL = os.path.join(BASE_DRIVE, "dataset_final")
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = "pu_resnet_results.txt"

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

# --- FUNÇÕES AUXILIARES ---
def load_dataset_data(ds_name, root_dir):
    ds_path = os.path.join(root_dir, ds_name)
    paths, labels = [], []
    label_map = {}
    
    if not os.path.exists(ds_path):
        return [], [], 0

    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                parts = root.split(os.sep)
                cls_name = parts[-1]
                if cls_name not in label_map: label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
    
    return paths, labels, len(label_map)

# --- PRÉ-TREINO (SOURCE) ---
def train_vibnet_source_resnet(target_dataset="PU"):
    """Treina ResNet-18 em todas as bases MENOS a target (PU)."""
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
    
    # Modelo ResNet-18
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

# --- CONFIGURAÇÃO PU ---
# A estrutura do PU geralmente é: PU / Setting_X / Class_Y / Image.png
def get_pu_splits(root_dir, test_cond):
    train_x, train_y, test_x, test_y = [], [], [], []
    label_map = {} # Dinâmico, baseado nas pastas encontradas

    # Identificar todas as condições
    all_conds = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for cond in all_conds:
        cond_path = os.path.join(root_dir, cond)
        is_test = (cond == test_cond)
        
        for cls_name in os.listdir(cond_path):
            cls_path = os.path.join(cond_path, cls_name)
            if not os.path.isdir(cls_path): continue
            
            if cls_name not in label_map: label_map[cls_name] = len(label_map)
            
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.png')]
            lbls = [label_map[cls_name]] * len(files)
            
            if is_test:
                test_x.extend(files); test_y.extend(lbls)
            else:
                train_x.extend(files); train_y.extend(lbls)
                
    return (train_x, train_y), (test_x, test_y), len(label_map)

# --- EXPERIMENTO PRINCIPAL ---
def run_pu_resnet_experiment():
    print(f"\n{'#'*50}")
    print(f"INICIANDO EXPERIMENTO RESNET: PU (Paderborn)")
    print(f"{'#'*50}")

    pu_root = os.path.join(DATASET_FINAL, "PU")
    if not os.path.exists(pu_root):
        print("Dataset PU não encontrado.")
        return

    # 1. Garante Pesos VibNet (ResNet)
    weights_path = "vibnet_resnet_source_no_PU.pth"
    if not os.path.exists(weights_path):
        weights_path = train_vibnet_source_resnet("PU")
    else:
        print(f"Pesos VibNet encontrados: {weights_path}")

    # Condições de Operação (Settings)
    conditions = sorted([d for d in os.listdir(pu_root) if os.path.isdir(os.path.join(pu_root, d))])
    print(f"Condições encontradas: {conditions}")

    strategies = ["Scratch", "ImageNet", "VibNet"]
    results = []

    # 2. Loop LODO-CV (Leave-One-Setting-Out)
    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} ---")

        (tr_x, tr_y), (te_x, te_y), num_classes = get_pu_splits(pu_root, test_cond)
        
        if len(tr_x) == 0: continue

        # Validação (20% do Treino)
        tr_x, val_x, tr_y, val_y = train_test_split(tr_x, tr_y, test_size=0.2, stratify=tr_y, random_state=42)

        loaders = {
            'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
            'test': DataLoader(VibDataset(te_x, te_y, data_transforms['val']), batch_size=BATCH_SIZE)
        }

        for strat in strategies:
            # Configuração do Modelo
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

            # Nova camada final
            model.fc = nn.Linear(n_feat, num_classes)
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
                "Dataset": "PU",
                "Test Condition": test_cond,
                "Strategy": strat,
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1
            })

    # Relatório
    df = pd.DataFrame(results)
    print("\n\n" + "="*50)
    print("RELATÓRIO FINAL PU - RESNET18")
    print("="*50)
    print(df.to_string())
    
    print("\n--- RESUMO ---")
    summary = df.groupby(["Strategy"])[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
    print(summary.to_string())

if __name__ == "__main__":
    run_pu_resnet_experiment()
