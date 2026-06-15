import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import DEVICE
from core.data import get_target_splits, get_dataloaders
from core.models import VibNetAutoencoder, VibNetFeatureExtractor
from core.engine import pre_train_autoencoder_source, train_target_fold

def run_ae_feature_extraction():
    # 1. Pré-treina o Autoencoder (Não-supervisionado) nos fontes
    ae_w_path = pre_train_autoencoder_source(target_to_exclude="UORED")
    
    # 2. Carrega as condições do alvo (LODO)
    conditions = ["Bearing_1", "Bearing_2"] # Exemplo do UORED
    
    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} ---")
        
        train_x, train_y, test_x, test_y, num_classes = get_target_splits("UORED", test_cond)
        dataloaders = get_dataloaders(train_x, train_y, test_x, test_y)

        # 3. Instancia o AE e carrega os pesos aprendidos
        base_ae = VibNetAutoencoder()
        base_ae.load_state_dict(torch.load(ae_w_path, map_location=DEVICE))
        
        # 4. Transforma o AE em um Classificador (Corta o Decoder, adiciona Camada Linear)
        # O freeze_encoder=True garante que estamos treinando APENAS o classificador (TL Puro)
        model = VibNetFeatureExtractor(base_ae, num_classes, freeze_encoder=True).to(DEVICE)
        
        # 5. Otimizador apenas para os parâmetros que requerem gradiente (a camada Linear)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 6. Roda a avaliação usando o MESMO motor da DenseNet!
        metrics = train_target_fold(model, dataloaders, optimizer, criterion, epochs=8)
        
        print(f"      Resultado AE Extractor: Bal Acc: {metrics['Bal Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")

if __name__ == "__main__":
    run_ae_feature_extraction()
