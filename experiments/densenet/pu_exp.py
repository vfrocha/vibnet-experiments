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
# Ajuste este caminho se necessário
BASE_DRIVE = os.getcwd()  # Usa diretório atual ou defina "/home/vfrocha/VibNet_Project"
DATASET_FINAL = os.path.join(BASE_DRIVE, "dataset_final")
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = "pu_results.txt"

# --- CLASSE DE LOGGING (Terminal + Arquivo) ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Garante gravação imediata

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redireciona print para o Logger
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
        print(f"AVISO: Pasta {ds_path} não encontrada. Pulando...")
        return [], [], 0

    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                parts = root.split(os.sep)
                cls_name = parts[-1]
                if cls_name not in label_map: label_map[cls_name] = len(label_map)
                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])
    
    print(f"[{ds_name}] Carregado: {len(paths)} imagens, {len(label_map)} classes.")
    return paths, labels, len(label_map)

# --- ETAPA 1: TREINO FONTE (Source) ---
def train_vibnet_source():
    print(f"\n{'='*40}")
    print("ETAPA 1: Pré-Treinamento VibNet (Source Domain)")
    print(f"{'='*40}")
    
    # AGORA PU É O ALVO, ENTÃO ELE SAI DA LISTA DE FONTE
    # HUST ENTRA COMO FONTE
    sources = ["CWRU_12k", "CWRU_48k", "HUST", "UORED"]
    
    all_paths, all_labels = [], []
    offset = 0

    for src in sources:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        l_adjusted = [x + offset for x in l]
        all_paths.extend(p)
        all_labels.extend(l_adjusted)
        offset += n_cls

    if len(all_paths) == 0:
        print("ERRO CRÍTICO: Nenhum dado de treino encontrado. Verifique os caminhos.")
        return None

    print(f"Total Source Data: {len(all_paths)} imagens, {offset} classes globais.")

    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)
    
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE)

    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, offset)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Treinando Source Model (5 Epochs)...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Source Ep {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Loss média: {total_loss/len(train_loader):.4f}")

    save_path = "vibnet_source_weights_pu_exp.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Pesos salvos em: {save_path}")
    return save_path

# --- ETAPA 2: EXPERIMENTO ALVO (PU) ---
def run_pu_experiments(weights_path):
    print(f"\n{'='*40}")
    print("ETAPA 2: Experimentos no PU (Target Domain)")
    print(f"{'='*40}")

    pu_root = os.path.join(DATASET_FINAL, "PU")
    if not os.path.exists(pu_root):
        print(f"ERRO: Pasta PU não encontrada em {pu_root}")
        return

    # 1. Detectar Condições e Classes Automaticamente
    conditions = sorted([d for d in os.listdir(pu_root) if os.path.isdir(os.path.join(pu_root, d))])
    if not conditions:
        print("Nenhuma condição encontrada na pasta PU.")
        return
    
    # Pega classes da primeira condição para criar o mapa
    first_cond_path = os.path.join(pu_root, conditions[0])
    classes = sorted([d for d in os.listdir(first_cond_path) if os.path.isdir(os.path.join(first_cond_path, d))])
    pu_class_map = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Condições Detectadas ({len(conditions)}): {conditions}")
    print(f"Classes Detectadas ({len(classes)}): {pu_class_map}")

    strategies = ["Scratch", "ImageNet", "VibNet"]
    results = []

    # Loop Leave-One-Condition-Out
    for test_cond in conditions:
        train_conds = [c for c in conditions if c != test_cond]
        print(f"\n>>> FOLD: Testando em '{test_cond}'")
        print(f"    Treino em: {train_conds}")

        # Preparar dados deste Fold
        train_x, train_y, test_x, test_y = [], [], [], []
        
        # Coleta dados de treino
        for cond in train_conds:
            c_path = os.path.join(pu_root, cond)
            for cls in classes:
                if cls not in pu_class_map: continue
                p_cls = os.path.join(c_path, cls)
                if not os.path.exists(p_cls): continue
                files = [os.path.join(p_cls, f) for f in os.listdir(p_cls) if f.endswith('.png')]
                train_x.extend(files)
                train_y.extend([pu_class_map[cls]] * len(files))

        # Coleta dados de teste
        c_path = os.path.join(pu_root, test_cond)
        for cls in classes:
            if cls not in pu_class_map: continue
            p_cls = os.path.join(c_path, cls)
            if not os.path.exists(p_cls): continue
            files = [os.path.join(p_cls, f) for f in os.listdir(p_cls) if f.endswith('.png')]
            test_x.extend(files)
            test_y.extend([pu_class_map[cls]] * len(files))

        if not train_x or not test_x:
            print("    AVISO: Dados insuficientes para este fold. Pulando.")
            continue

        # Split Validação
        tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y, random_state=42)

        dls = {
            'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
            'test': DataLoader(VibDataset(test_x, test_y, data_transforms['val']), batch_size=BATCH_SIZE)
        }

        for strat in strategies:
            print(f"   -> Estratégia: {strat}")
            
            # Setup Modelo
            model = models.densenet121(weights=None)
            num_ftrs = model.classifier.in_features
            
            if strat == "ImageNet":
                temp = models.densenet121(weights='IMAGENET1K_V1')
                model.features = temp.features
            elif strat == "VibNet":
                state = torch.load(weights_path)
                m_dict = model.state_dict()
                # Carrega pesos ignorando a última camada
                pt_dict = {k: v for k, v in state.items() if k in m_dict and 'classifier' not in k}
                m_dict.update(pt_dict)
                model.load_state_dict(m_dict)
            
            model.classifier = nn.Linear(num_ftrs, len(pu_class_map))
            model = model.to(DEVICE)
            
            lr = 0.001 if strat == "Scratch" else 0.0001
            opt = optim.Adam(model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()
            
            best_f1 = 0
            best_w = copy.deepcopy(model.state_dict())

            # Treino Curto (Fine-Tuning)
            for ep in range(8): # 8 Epocas por fold
                model.train()
                for inp, lbl in dls['train']:
                    inp, lbl = inp.to(DEVICE), lbl.to(DEVICE)
                    opt.zero_grad()
                    loss = crit(model(inp), lbl)
                    loss.backward()
                    opt.step()
                
                # Validação
                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for inp, lbl in dls['val']:
                        out = model(inp.to(DEVICE))
                        _, p = torch.max(out, 1)
                        preds.extend(p.cpu().numpy()); targs.extend(lbl.numpy())
                
                v_f1 = f1_score(targs, preds, average='macro')
                if v_f1 > best_f1:
                    best_f1 = v_f1
                    best_w = copy.deepcopy(model.state_dict())
            
            # Teste Final
            model.load_state_dict(best_w)
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for inp, lbl in dls['test']:
                    out = model(inp.to(DEVICE))
                    _, p = torch.max(out, 1)
                    all_p.extend(p.cpu().numpy()); all_t.extend(lbl.numpy())
            
            bal_acc = balanced_accuracy_score(all_t, all_p)
            macro_f1 = f1_score(all_t, all_p, average='macro')
            
            results.append({
                "Fold Condition": test_cond,
                "Strategy": strat,
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1
            })
            print(f"      Resultado: Bal Acc {bal_acc:.4f} | Macro F1 {macro_f1:.4f}")

    # Resumo Final
    print(f"\n{'='*40}")
    print("RESUMO DOS EXPERIMENTOS (PU)")
    print(f"{'='*40}")
    
    df = pd.DataFrame(results)
    print(df.to_string())
    
    print("\n--- MÉDIAS ---")
    summary = df.groupby("Strategy")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
    print(summary.to_string())
    print("\nResultados salvos em:", OUTPUT_FILE)

# --- EXECUÇÃO ---
if __name__ == "__main__":
    # Verifica se já temos o peso da fonte treinado, senão treina
    w_path = "vibnet_source_weights_pu_exp.pth"
    
    # Se quiser forçar re-treino do VibNet, apague o arquivo .pth ou comente o if
    if not os.path.exists(w_path):
        w_path = train_vibnet_source()
    else:
        print(f"Pesos VibNet encontrados ({w_path}), pulando Etapa 1.")

    if w_path:
        run_pu_experiments(w_path)
