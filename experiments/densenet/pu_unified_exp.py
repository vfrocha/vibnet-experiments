import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, roc_auc_score
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
OUTPUT_FILE = os.path.join(BASE_DRIVE, "results", "pu_unified_results.txt")

os.makedirs(os.path.join(BASE_DRIVE, "results"), exist_ok=True)
os.makedirs(os.path.join(BASE_DRIVE, "weights"), exist_ok=True)

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
    if not os.path.exists(ds_path): return [], [], 0
    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                cls_name = root.split(os.sep)[-1]
                if cls_name not in label_map: label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
    return paths, labels, len(label_map)

# --- PRÉ-TREINO FONTE (EXCLUINDO PU) ---
def train_vibnet_source_no_pu(start_with_imagenet=True):
    strat_name = "ImageNet" if start_with_imagenet else "Scratch"
    print(f"\n>>> Treinando VibNet Source (Excluindo PU) | Iniciando de: {strat_name}...")
    
    # PU é o alvo, então usamos as outras bases como fonte
    sources = ["CWRU_12k", "CWRU_48k", "HUST", "UORED"]
    all_paths, all_labels = [], []
    offset = 0

    for src in sources:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        if len(p) > 0:
            all_paths.extend(p)
            all_labels.extend([x + offset for x in l])
            offset += n_cls

    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    
    # Define o caminho do peso com base na estratégia
    w_filename = f"vibnet_source_no_pu_{'imagenet' if start_with_imagenet else 'scratch'}.pth"
    w_path = os.path.join(BASE_DRIVE, "weights", w_filename)

    model = models.densenet121(weights='IMAGENET1K_V1' if start_with_imagenet else None)
    model.classifier = nn.Linear(model.classifier.in_features, offset)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Source Ep {epoch+1}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), w_path)
    return w_path

# --- EXPERIMENTO PU ---
def run_pu_unified_experiment():
    print(f"\n{'#'*50}\nINICIANDO EXPERIMENTO: PU UNIFICADO\n{'#'*50}")

    w_scratch_path = os.path.join(BASE_DRIVE, "weights", "vibnet_source_no_pu_scratch.pth")
    w_imagenet_path = os.path.join(BASE_DRIVE, "weights", "vibnet_source_no_pu_imagenet.pth")

    if not os.path.exists(w_scratch_path): train_vibnet_source_no_pu(start_with_imagenet=False)
    if not os.path.exists(w_imagenet_path): train_vibnet_source_no_pu(start_with_imagenet=True)

    pu_root = os.path.join(DATASET_FINAL, "PU")
    # Detecta condições de PU (C1, C2, C3, C4)
    conditions = sorted([d for d in os.listdir(pu_root) if os.path.isdir(os.path.join(pu_root, d))])
    
    # Mapeamento de classes
    sample_cond_path = os.path.join(pu_root, conditions[0])
    classes = sorted([d for d in os.listdir(sample_cond_path) if os.path.isdir(os.path.join(sample_cond_path, d))])
    cls_map = {c: i for i, c in enumerate(classes)}
    
    strategies = ["Scratch", "ImageNet", "VibNet_from_Scratch", "VibNet_from_ImageNet"]
    results = []

    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} ---")
        train_x, train_y, test_x, test_y = [], [], [], []
        
        # Coleta dados para LODO-CV
        for cond in conditions:
            c_path = os.path.join(pu_root, cond)
            is_test = (cond == test_cond)
            for cls in classes:
                p_cls = os.path.join(c_path, cls)
                if not os.path.exists(p_cls): continue
                files = [os.path.join(p_cls, f) for f in os.listdir(p_cls) if f.endswith('.png')]
                if is_test:
                    test_x.extend(files); test_y.extend([cls_map[cls]] * len(files))
                else:
                    train_x.extend(files); train_y.extend([cls_map[cls]] * len(files))

        tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)
        loaders = {
            'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
            'test': DataLoader(VibDataset(test_x, test_y, data_transforms['val']), batch_size=BATCH_SIZE)
        }

        for strat in strategies:
            model = models.densenet121(weights=None)
            n_feat = model.classifier.in_features
            
            # Carregamento de pesos conforme estratégia
            if strat == "ImageNet":
                model.features = models.densenet121(weights='IMAGENET1K_V1').features
            elif "VibNet" in strat:
                path = w_imagenet_path if "ImageNet" in strat else w_scratch_path
                state = torch.load(path)
                model.load_state_dict({k: v for k, v in state.items() if 'classifier' not in k}, strict=False)

            model.classifier = nn.Linear(n_feat, len(cls_map))
            model = model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=1e-3 if strat=="Scratch" else 1e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_f1, best_w = 0, None
            for ep in range(8):
                model.train()
                for x, y in loaders['train']:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
                
                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for x, y in loaders['val']:
                        _, p = torch.max(model(x.to(DEVICE)), 1)
                        preds.extend(p.cpu().numpy()); targs.extend(y.numpy())
                f1 = f1_score(targs, preds, average='macro')
                if f1 > best_f1:
                    best_f1 = f1; best_w = copy.deepcopy(model.state_dict())

            # Teste com Softmax para AUC
            model.load_state_dict(best_w); model.eval()
            f_p, f_t, f_probs = [], [], []
            with torch.no_grad():
                for x, y in loaders['test']:
                    out = model(x.to(DEVICE))
                    f_probs.extend(F.softmax(out, dim=1).cpu().numpy())
                    _, p = torch.max(out, 1); f_p.extend(p.cpu().numpy()); f_t.extend(y.numpy())
            
            acc, b_acc, m_f1 = accuracy_score(f_t, f_p), balanced_accuracy_score(f_t, f_p), f1_score(f_t, f_p, average='macro')
            auc = roc_auc_score(f_t, np.array(f_probs), multi_class='ovr', average='macro', labels=list(range(len(cls_map))))
            
            print(f"   [{strat}] Acc: {acc:.4f} | Bal Acc: {b_acc:.4f} | F1: {m_f1:.4f} | AUC: {auc:.4f}")
            results.append({"Condition": test_cond, "Strategy": strat, "Accuracy": acc, "Bal Accuracy": b_acc, "Macro F1": m_f1, "Macro AUC": auc})

    return results

if __name__ == "__main__":
    res = run_pu_unified_experiment()
    if res:
        df = pd.DataFrame(res)
        print("\n" + "="*50 + "\nRESUMO PU\n" + "="*50)
        print(df.groupby("Strategy")[["Accuracy", "Bal Accuracy", "Macro F1", "Macro AUC"]].agg(['mean', 'std']).to_string())
