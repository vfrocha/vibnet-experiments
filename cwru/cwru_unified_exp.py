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
OUTPUT_FILE = "cwru_unified_results.txt"

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
                parts = root.split(os.sep)
                cls_name = parts[-1]
                if cls_name not in label_map: label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
    return paths, labels, len(label_map)

# --- PRÉ-TREINO UNIFICADO (SOURCE) ---
def train_vibnet_source_unified():
    """
    Treina VibNet (DenseNet-121) excluindo AMBAS as bases CWRU (12k e 48k).
    O modelo vai aprender de HUST, PU e UORED.
    """
    print(f"\n>>> Treinando VibNet Source (Excluindo CWRU Completo)...")
    
    # Exclui 12k e 48k pois agora são nosso alvo unificado
    all_datasets = ["HUST", "UORED", "PU"]
    
    all_paths, all_labels = [], []
    offset = 0

    for src in all_datasets:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        if len(p) > 0:
            l_adjusted = [x + offset for x in l]
            all_paths.extend(p)
            all_labels.extend(l_adjusted)
            offset += n_cls
            print(f"    + {src}: {len(p)} imagens")

    # Split de validação para monitorar overfitting no source
    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)
    
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    
    # Modelo DenseNet-121
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

    w_path = "vibnet_source_no_CWRU_Unified.pth"
    torch.save(model.state_dict(), w_path)
    return w_path

# --- EXPERIMENTO UNIFICADO ---
def run_cwru_unified_experiment():
    print(f"\n{'#'*50}")
    print(f"INICIANDO EXPERIMENTO: CWRU UNIFICADO (12k + 48k)")
    print(f"{'#'*50}")

    # Lista de sub-datasets que compõem o CWRU Unificado
    cwru_subsets = ["CWRU_12k", "CWRU_48k"]
    
    # 1. Pesos Fonte
    weights_path = "vibnet_source_no_CWRU_Unified.pth"
    if not os.path.exists(weights_path):
        weights_path = train_vibnet_source_unified()
    else:
        print(f"Pesos pré-treinados encontrados: {weights_path}")

    # 2. Mapeamento de Condições e Classes
    # Assume-se que ambos os datasets têm a mesma estrutura de pastas: Load_XHP/Class_Y
    conditions = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    
    # Descobre classes olhando para uma pasta qualquer
    sample_dir = os.path.join(DATASET_FINAL, "CWRU_12k", "Load_0HP")
    if not os.path.exists(sample_dir): # Tenta 48k se 12k falhar
        sample_dir = os.path.join(DATASET_FINAL, "CWRU_48k", "Load_0HP")
        
    classes = sorted([d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))])
    cls_map = {c: i for i, c in enumerate(classes)}
    
    print(f"Condições Unificadas: {conditions}")
    print(f"Classes Mapeadas: {cls_map}")

    strategies = ["Scratch", "ImageNet", "VibNet"]
    results = []

    # 3. LODO-CV Loop (Leave-One-Load-Out)
    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} (Dados de 12k e 48k misturados) ---")

        train_x, train_y, test_x, test_y = [], [], [], []

        # Itera sobre os sub-datasets (12k e 48k) para coletar dados
        for subset in cwru_subsets:
            subset_root = os.path.join(DATASET_FINAL, subset)
            if not os.path.exists(subset_root): continue

            # Se for a condição de teste, adiciona ao conjunto de teste
            # Se não, adiciona ao conjunto de treino
            
            # Carrega Teste (Load atual)
            test_path = os.path.join(subset_root, test_cond)
            if os.path.exists(test_path):
                for cls in classes:
                    p = os.path.join(test_path, cls)
                    if os.path.exists(p):
                        files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.png')]
                        test_x.extend(files)
                        test_y.extend([cls_map[cls]] * len(files))

            # Carrega Treino (Outros Loads)
            train_conds = [c for c in conditions if c != test_cond]
            for tr_cond in train_conds:
                tr_path = os.path.join(subset_root, tr_cond)
                if os.path.exists(tr_path):
                    for cls in classes:
                        p = os.path.join(tr_path, cls)
                        if os.path.exists(p):
                            files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.png')]
                            train_x.extend(files)
                            train_y.extend([cls_map[cls]] * len(files))

        print(f"    Treino: {len(train_x)} imagens | Teste: {len(test_x)} imagens")
        
        # Split de Validação
        if len(train_x) == 0: continue
        tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)

        loaders = {
            'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
            'test': DataLoader(VibDataset(test_x, test_y, data_transforms['val']), batch_size=BATCH_SIZE)
        }

        for strat in strategies:
            # DenseNet-121
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

            model.classifier = nn.Linear(n_feat, len(cls_map))
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
                "Dataset": "CWRU_Unified",
                "Test Condition": test_cond,
                "Strategy": strat,
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1
            })

    return results

if __name__ == "__main__":
    results = run_cwru_unified_experiment()
    if results:
        df = pd.DataFrame(results)
        print("\n\n" + "="*50)
        print("RELATÓRIO FINAL CWRU UNIFICADO")
        print("="*50)
        print(df.to_string())
        
        print("\n--- RESUMO ---")
        summary = df.groupby(["Strategy"])[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
        print(summary.to_string())
