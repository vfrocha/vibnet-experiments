import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Garante que o Python encontre a pasta core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import DEVICE, RESULTS_DIR
from core.utils import Logger
from core.data import get_target_splits, get_dataloaders
from core.models import VibNetAutoencoder, VibNetFeatureExtractor
from core.engine import pre_train_autoencoder_source, train_target_fold

# --- GRUPOS VIRTUAIS UORED E MAPA DE CLASSES ---
VIRTUAL_CONDITIONS = {
    "Group_A": ["Bearing_1", "Bearing_6", "Bearing_11", "Bearing_16"],
    "Group_B": ["Bearing_2", "Bearing_7", "Bearing_12", "Bearing_17"],
    "Group_C": ["Bearing_3", "Bearing_8", "Bearing_13", "Bearing_18"],
    "Group_D": ["Bearing_4", "Bearing_9", "Bearing_14", "Bearing_19"],
    "Group_E": ["Bearing_5", "Bearing_10", "Bearing_15", "Bearing_20"]
}

CLASS_MAP_UORED = {
    "Class_Normal": 0, "Class_47": 1, "Class_48": 2, 
    "Class_49": 3, "Class_50": 4, "Class_51": 5
}

# Configura o Logger para este experimento específico
sys.stdout = Logger(os.path.join(RESULTS_DIR, "ae_uored_results.txt"))

def run_ae_uored_experiment():
    # 1. PRÉ-TREINO (Não-supervisionado) nos domínios fonte
    print("Verificando/Iniciando pré-treinamento do Autoencoder (Source)...")
    ae_w_path = pre_train_autoencoder_source(target_to_exclude="UORED")
    
    results = []

    # 2. LOOP SOBRE OS GRUPOS VIRTUAIS (LODO-CV)
    for group_name, bearings_list in VIRTUAL_CONDITIONS.items():
        print(f"\n{'='*40}\nFold: Testando no {group_name} (Autoencoder)\n{'='*40}")
        
        # Pega os splits usando a lista de rolamentos e o mapa forçado
        train_x, train_y, test_x, test_y, num_classes = get_target_splits(
            dataset_name="UORE
