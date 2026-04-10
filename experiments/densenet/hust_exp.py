import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

# --- CONFIGURAÇÕES ---
BASE_DRIVE = "/home/vfrocha/VibNet_Project"
DATASET_FINAL = os.path.join(BASE_DRIVE, "dataset_final")
IMG_SIZE = 224 # Atualizado conforme pedido
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET GENÉRICO ---
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
        except:
            # Retorna preto em caso de erro de leitura
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), self.labels[idx]

# --- TRANSFORMAÇÕES ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # Augmentation leve
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- FUNÇÃO PARA CARREGAR DADOS DE UMA BASE ---
def load_dataset_data(ds_name, root_dir):
    """Lê todas as imagens de um dataset específico recursivamente."""
    ds_path = os.path.join(root_dir, ds_name)
    paths = []
    labels = []
    label_map = {} # Mapeia nomes de classe para inteiros locais

    # Caminha por todas as pastas (Conditions/Classes)
    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".png"):
                # Estrutura esperada: .../Condition/Class_Name/img.png
                parts = root.split(os.sep)
                cls_name = parts[-1]

                # Cria ID numérico para a classe se não existir
                if cls_name not in label_map:
                    label_map[cls_name] = len(label_map)

                paths.append(os.path.join(root, file))
                labels.append(label_map[cls_name])

    print(f"[{ds_name}] Carregado: {len(paths)} imagens, {len(label_map)} classes.")
    return paths, labels, len(label_map)

def train_vibnet_source():
    print(f"\n{'='*40}")
    print("ETAPA 1: Pré-Treinamento VibNet (Source Domain)")
    print(f"{'='*40}")

    # 1. Carregar Source Domains (Todos menos HUST)
    sources = ["CWRU_12k", "CWRU_48k", "PU", "UORED"]
    all_paths, all_labels = [], []
    offset = 0

    for src in sources:
        p, l, n_cls = load_dataset_data(src, DATASET_FINAL)
        # Ajusta labels para não sobrepor (Ex: CWRU 0-9, PU 10-14...)
        l_adjusted = [x + offset for x in l]
        all_paths.extend(p)
        all_labels.extend(l_adjusted)
        offset += n_cls

    num_classes_total = offset
    print(f"Total Source Data: {len(all_paths)} imagens, {num_classes_total} classes globais.")

    # 2. Split Simples (Treino/Val) para o Source
    # Aqui não precisamos de LOO-CV rigoroso, queremos apenas aprender features
    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels)

    train_ds = VibDataset(tr_x, tr_y, data_transforms['train'])
    val_ds = VibDataset(val_x, val_y, data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3. Modelo
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, num_classes_total)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 4. Loop de Treino Rápido (5 Epochs costuma bastar para pré-treino robusto)
    print("Treinando Source Model...")
    for epoch in range(5):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Source Ep {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # 5. Salvar Pesos VibNet (Sem a última camada, pois ela muda no target)
    print("Salvando backbone da VibNet...")
    torch.save(model.state_dict(), os.path.join(BASE_DRIVE, "vibnet_source_weights.pth"))
    return model.state_dict()

# DESCOMENTE PARA RODAR O PRÉ-TREINO SE AINDA NÃO TIVER O ARQUIVO
#vibnet_weights = train_vibnet_source()

# Mapeamento do HUST (Fixo)
HUST_CLASS_MAP = {
    "Class_40": 0, "Class_41": 1, "Class_42": 2, "Class_43": 3,
    "Class_44": 4, "Class_45": 5, "Class_46": 6
}
HUST_CONDS = ["Load_0W", "Load_200W", "Load_400W"]

def get_hust_splits(root_dir, train_conds, test_conds):
    """Função de split específica para o HUST LODO-CV"""
    train_x, train_y, test_x, test_y = [], [], [], []

    # Varre as condições (Pastas Load_XW)
    for cond in os.listdir(root_dir):
        cond_path = os.path.join(root_dir, cond)
        if not os.path.isdir(cond_path): continue

        is_train = cond in train_conds
        is_test = cond in test_conds
        if not is_train and not is_test: continue

        # Varre as classes
        for cls_name in os.listdir(cond_path):
            if cls_name not in HUST_CLASS_MAP: continue

            cls_path = os.path.join(cond_path, cls_name)
            files = sorted([f for f in os.listdir(cls_path) if f.endswith('.png')])
            paths = [os.path.join(cls_path, f) for f in files]
            lbls = [HUST_CLASS_MAP[cls_name]] * len(files)

            if is_test:
                test_x.extend(paths); test_y.extend(lbls)
            elif is_train:
                train_x.extend(paths); train_y.extend(lbls)

    return (train_x, train_y), (test_x, test_y)

def run_target_experiments():
    print(f"\n{'='*40}")
    print("ETAPA 2: Experimentos no HUST (Target Domain)")
    print(f"{'='*40}")

    hust_root = os.path.join(DATASET_FINAL, "HUST")
    strategies = ["Scratch", "ImageNet", "VibNet"]
    results = []

    # Loop LODO-CV (Leave-One-Load-Out)
    for test_cond in HUST_CONDS:
        train_conds = [c for c in HUST_CONDS if c != test_cond]
        print(f"\n>>> FOLD: Testando em {test_cond} (Treino em {train_conds})")

        (tr_x, tr_y), (te_x, te_y) = get_hust_splits(hust_root, train_conds, [test_cond])

        # Criação de Validação interna (20% do treino) para Early Stopping
        tr_x, val_x, tr_y, val_y = train_test_split(tr_x, tr_y, test_size=0.2, stratify=tr_y, random_state=42)

        dataloaders = {
            'train': DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(VibDataset(val_x, val_y, data_transforms['val']), batch_size=BATCH_SIZE),
            'test': DataLoader(VibDataset(te_x, te_y, data_transforms['val']), batch_size=BATCH_SIZE)
        }

        for strat in strategies:
            print(f"   -> Estratégia: {strat}")

            # 1. Inicialização do Modelo
            model = models.densenet121(weights=None) # Base limpa
            num_ftrs = model.classifier.in_features

            if strat == "Scratch":
                # Pesos aleatórios (já está assim)
                model.classifier = nn.Linear(num_ftrs, len(HUST_CLASS_MAP))

            elif strat == "ImageNet":
                # Carrega pesos ImageNet
                temp_model = models.densenet121(weights='IMAGENET1K_V1')
                model.features = temp_model.features
                model.classifier = nn.Linear(num_ftrs, len(HUST_CLASS_MAP))

            elif strat == "VibNet":
                # Carrega pesos da Etapa 1
                weights_path = os.path.join(BASE_DRIVE, "vibnet_source_weights.pth")
                if not os.path.exists(weights_path):
                    print("ERRO: Pesos VibNet não encontrados. Rode a Etapa 1 primeiro.")
                    continue

                # Carrega state_dict com cuidado (ignorando a última camada classifier que tem tamanho diferente)
                state_dict = torch.load(weights_path)
                model_dict = model.state_dict()
                # Filtra pesos que não batem (classifier)
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                model.classifier = nn.Linear(num_ftrs, len(HUST_CLASS_MAP))

            model = model.to(DEVICE)

            # 2. Treino (Fine-Tuning)
            # LR menor para transfer learning, maior para scratch
            lr = 0.001 if strat == "Scratch" else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            best_f1 = 0.0
            best_w = copy.deepcopy(model.state_dict())

            # Epochs reduzidos para demonstração, use 15-20 para artigo
            for epoch in range(10):
                model.train()
                for inp, lbl in dataloaders['train']:
                    inp, lbl = inp.to(DEVICE), lbl.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(inp), lbl)
                    loss.backward()
                    optimizer.step()

                # Validação
                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for inp, lbl in dataloaders['val']:
                        out = model(inp.to(DEVICE))
                        _, p = torch.max(out, 1)
                        preds.extend(p.cpu().numpy()); targs.extend(lbl.numpy())

                val_f1 = f1_score(targs, preds, average='macro')
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_w = copy.deepcopy(model.state_dict())

            # 3. Teste Final (No Load Inédito)
            model.load_state_dict(best_w)
            model.eval()
            all_preds, all_lbls = [], []
            with torch.no_grad():
                for inp, lbl in dataloaders['test']:
                    out = model(inp.to(DEVICE))
                    _, p = torch.max(out, 1)
                    all_preds.extend(p.cpu().numpy()); all_lbls.extend(lbl.numpy())

            acc = balanced_accuracy_score(all_lbls, all_preds)
            f1 = f1_score(all_lbls, all_preds, average='macro')

            results.append({
                "Test Condition": test_cond,
                "Strategy": strat,
                "Bal Accuracy": acc,
                "Macro F1": f1
            })
            print(f"      Result: Bal Acc {acc:.4f} | F1 {f1:.4f}")

    # Relatório Final
    df = pd.DataFrame(results)
    print("\n\n=== RESULTADOS CONSOLIDADOS ===")
    print(df)

    print("\n=== MÉDIA POR ESTRATÉGIA ===")
    summary = df.groupby("Strategy")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
    print(summary)

# --- EXECUÇÃO ---
# Passo 1: Se não tiver os pesos, descomente a linha abaixo:
train_vibnet_source()

# Passo 2: Rodar experimentos
run_target_experiments()
