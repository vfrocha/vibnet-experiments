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

# --- CONFIGURAÇÕES DE CAMINHO ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DRIVE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../")) 
DATASET_FINAL = os.path.join(BASE_DRIVE, "dataset_final")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = os.path.join(BASE_DRIVE, "results", "hust_unified_results.txt")

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

# --- DATASET & TRANSFORMS ---
class VibDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list, self.labels, self.transform = file_list, labels, transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_list[idx]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, self.labels[idx]
        except: return torch.zeros((3, 224, 224)), self.labels[idx]

data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}

def load_dataset_data(ds_name, root_dir):
    ds_path = os.path.join(root_dir, ds_name)
    paths, labels, label_map = [], [], {}
    if not os.path.exists(ds_path): return [], [], 0
    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                cls_name = root.split(os.sep)[-1]
                if cls_name not in label_map: label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
    return paths, labels, len(label_map)

# --- TREINO FONTE (EXCLUI HUST) ---
def train_source(start_with_imagenet=True):
    strat = "imagenet" if start_with_imagenet else "scratch"
    print(f"\n>>> Treinando VibNet Source (Excluindo HUST) | Start: {strat}")
    sources = ["CWRU_12k", "CWRU_48k", "PU", "UORED"]
    all_paths, all_labels, offset = [], [], 0
    for src in sources:
        p, l, n = load_dataset_data(src, DATASET_FINAL)
        if n > 0:
            all_paths.extend(p); all_labels.extend([x + offset for x in l]); offset += n
    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)
    loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=32, shuffle=True)
    model = models.densenet121(weights='IMAGENET1K_V1' if start_with_imagenet else None)
    model.classifier = nn.Linear(model.classifier.in_features, offset)
    model, opt, crit = model.to(DEVICE), optim.Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss()
    for ep in range(5):
        model.train()
        for x, y in tqdm(loader, desc=f"Ep {ep+1}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    w_path = os.path.join(BASE_DRIVE, "weights", f"vibnet_source_no_hust_{strat}.pth")
    torch.save(model.state_dict(), w_path); return w_path

# --- EXPERIMENTO HUST ---
def run_experiment():
    w_scratch = os.path.join(BASE_DRIVE, "weights", "vibnet_source_no_hust_scratch.pth")
    w_imagenet = os.path.join(BASE_DRIVE, "weights", "vibnet_source_no_hust_imagenet.pth")
    if not os.path.exists(w_scratch): train_source(False)
    if not os.path.exists(w_imagenet): train_source(True)
    hust_root = os.path.join(DATASET_FINAL, "HUST")
    conditions = sorted([d for d in os.listdir(hust_root) if os.path.isdir(os.path.join(hust_root, d))])
    classes = sorted([d for d in os.listdir(os.path.join(hust_root, conditions[0])) if os.path.isdir(os.path.join(hust_root, conditions[0], d))])
    cls_map = {c: i for i, c in enumerate(classes)}
    strategies, results = ["Scratch", "ImageNet", "VibNet_from_Scratch", "VibNet_from_ImageNet"], []

    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} ---")
        train_x, train_y, test_x, test_y = [], [], [], []
        for cond in conditions:
            c_p = os.path.join(hust_root, cond)
            for cls in classes:
                p_c = os.path.join(c_p, cls)
                if not os.path.exists(p_c): continue
                f = [os.path.join(p_c, file) for file in os.listdir(p_c) if file.endswith('.png')]
                if cond == test_cond: test_x.extend(f); test_y.extend([cls_map[cls]]*len(f))
                else: train_x.extend(f); train_y.extend([cls_map[cls]]*len(f))
        tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)
        lds = {'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=32, shuffle=True),
               'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=32),
               'test': DataLoader(VibDataset(test_x, test_y, data_transforms['val']), batch_size=32)}

        for strat in strategies:
            model = models.densenet121(weights=None)
            if strat == "ImageNet": model.features = models.densenet121(weights='IMAGENET1K_V1').features
            elif "VibNet" in strat:
                state = torch.load(w_imagenet if "ImageNet" in strat else w_scratch)
                model.load_state_dict({k: v for k, v in state.items() if 'classifier' not in k}, strict=False)
            model.classifier = nn.Linear(model.classifier.in_features, len(cls_map))
            model, opt, crit = model.to(DEVICE), optim.Adam(model.parameters(), lr=1e-3 if strat=="Scratch" else 1e-4), nn.CrossEntropyLoss()
            best_f1, best_w = 0, None
            for ep in range(8):
                model.train()
                for x, y in lds['train']:
                    x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad(); crit(model(x), y).backward(); opt.step()
                model.eval(); pds, tgs = [], []
                with torch.no_grad():
                    for x, y in lds['val']:
                        _, p = torch.max(model(x.to(DEVICE)), 1); pds.extend(p.cpu().numpy()); tgs.extend(y.numpy())
                f1 = f1_score(tgs, pds, average='macro')
                if f1 > best_f1: best_f1, best_w = f1, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_w); model.eval(); f_p, f_t, f_probs = [], [], []
            with torch.no_grad():
                for x, y in lds['test']:
                    o = model(x.to(DEVICE)); f_probs.extend(F.softmax(o, dim=1).cpu().numpy()); _, p = torch.max(o, 1); f_p.extend(p.cpu().numpy()); f_t.extend(y.numpy())
            acc, b_acc, m_f1, auc = accuracy_score(f_t, f_p), balanced_accuracy_score(f_t, f_p), f1_score(f_t, f_p, average='macro'), roc_auc_score(f_t, np.array(f_probs), multi_class='ovr', average='macro', labels=list(range(len(cls_map))))
            print(f"   [{strat}] Acc: {acc:.4f} | Bal Acc: {b_acc:.4f} | F1: {m_f1:.4f} | AUC: {auc:.4f}")
            results.append({"Condition": test_cond, "Strategy": strat, "Accuracy": acc, "Bal Accuracy": b_acc, "Macro F1": m_f1, "Macro AUC": auc})
    return results

if __name__ == "__main__":
    res = run_experiment()
    if res:
        df = pd.DataFrame(res)
        print("\nRESUMO HUST\n" + df.groupby("Strategy")[["Accuracy", "Bal Accuracy", "Macro F1", "Macro AUC"]].agg(['mean', 'std']).to_string())
