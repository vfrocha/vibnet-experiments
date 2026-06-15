import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

from .models import VibNetAutoencoder
from .config import DEVICE, WEIGHTS_DIR, BATCH_SIZE
from .data import VibDataset, data_transforms, load_source_data

def pre_train_source(target_to_exclude, start_with_imagenet=True):
    """
    Executa o pré-treinamento da VibNet usando os domínios fonte.
    Retorna o caminho do arquivo .pth gerado.
    """
    strat_name = "imagenet" if start_with_imagenet else "scratch"
    
    # Trata o nome caso seja uma lista (útil para exclusões duplas como CWRU 12k e 48k)
    if isinstance(target_to_exclude, list):
        target_name = "_".join(target_to_exclude)
    else:
        target_name = target_to_exclude
        
    w_filename = f"vibnet_source_no_{target_name}_{strat_name}.pth"
    w_path = os.path.join(WEIGHTS_DIR, w_filename)

    # Se já existe, evita re-treino
    if os.path.exists(w_path):
        print(f"Pesos encontrados: {w_path}. Pulando pré-treino.")
        return w_path

    print(f"\n>>> Treinando VibNet Source (Excluindo {target_name}) | Start: {strat_name}")
    
    all_paths, all_labels, offset = load_source_data(target_to_exclude)
    
    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, stratify=all_labels, random_state=42)
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    
    model = models.densenet121(weights='IMAGENET1K_V1' if start_with_imagenet else None)
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
        print(f"    Loss média: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), w_path)
    return w_path


def train_target_fold(model, dataloaders, optimizer, criterion, epochs=8):
    """
    Executa o fine-tuning e a avaliação para um fold de LODO-CV no domínio alvo.
    Salva o melhor modelo baseado no Macro F1 de validação.
    """
    best_f1 = 0.0
    best_w = copy.deepcopy(model.state_dict())
    
    # --- LOOP DE TREINO E VALIDAÇÃO ---
    for ep in range(epochs):
        model.train()
        for x, y in dataloaders['train']:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for x, y in dataloaders['val']:
                out = model(x.to(DEVICE))
                _, p = torch.max(out, 1)
                val_preds.extend(p.cpu().numpy())
                val_targs.extend(y.numpy())
                
        val_f1 = f1_score(val_targs, val_preds, average='macro')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_w = copy.deepcopy(model.state_dict())

    # --- LOOP DE TESTE ---
    model.load_state_dict(best_w)
    model.eval()
    test_preds, test_targs = [], []
    
    with torch.no_grad():
        for x, y in dataloaders['test']:
            out = model(x.to(DEVICE))
            _, p = torch.max(out, 1)
            test_preds.extend(p.cpu().numpy())
            test_targs.extend(y.numpy())
            
    # --- MÉTRICAS GERAIS ---
    metrics = {
        "Accuracy": accuracy_score(test_targs, test_preds),
        "Bal Accuracy": balanced_accuracy_score(test_targs, test_preds),
        "Macro F1": f1_score(test_targs, test_preds, average='macro')
    }
    
    return metrics

def pre_train_autoencoder_source(target_to_exclude):
    """
    Treina o Autoencoder nos domínios fonte usando MSE (Não-supervisionado).
    """
    w_filename = f"vibnet_ae_source_no_{target_to_exclude}.pth"
    w_path = os.path.join(WEIGHTS_DIR, w_filename)

    if os.path.exists(w_path):
        print(f"Pesos do Autoencoder encontrados: {w_path}. Pulando pré-treino.")
        return w_path

    print(f"\n>>> Pré-treinando Autoencoder Source (Excluindo {target_to_exclude})")
    
    all_paths, all_labels, _ = load_source_data(target_to_exclude)
    
    # Prepara os dados (reaproveitando as funções do core)
    tr_x, val_x, tr_y, val_y = train_test_split(all_paths, all_labels, test_size=0.1, random_state=42)
    train_loader = DataLoader(VibDataset(tr_x, tr_y, data_transforms['train']), batch_size=BATCH_SIZE, shuffle=True)
    
    # Instancia o Autoencoder
    model = VibNetAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # <-- A MÁGICA ESTÁ AQUI (Erro de reconstrução)

    for epoch in range(10): # AEs costumam precisar de 10 a 15 épocas para convergir bem
        model.train()
        total_loss = 0
        for inputs, _ in tqdm(train_loader, desc=f"AE Source Ep {epoch+1}", leave=False):
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            
            # O modelo retorna (imagem_reconstruida, espaco_latente)
            reconstruction, _ = model(inputs)
            
            # A perda é calculada comparando a reconstrução com a imagem ORIGINAL
            loss = criterion(reconstruction, inputs) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"    MSE Loss média: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), w_path)
    return w_path
