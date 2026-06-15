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
            dataset_name="UORED", 
            test_condition=bearings_list, 
            predefined_class_map=CLASS_MAP_UORED
        )
        
        if len(test_x) == 0 or len(train_x) == 0:
            print(f"   [AVISO] Dados insuficientes no {group_name}. Treino: {len(train_x)} | Teste: {len(test_x)}")
            continue

        print(f"   [Dados] Treino: {len(train_x)} imgs | Teste: {len(test_x)} imgs")
        dataloaders = get_dataloaders(train_x, train_y, test_x, test_y)

        # 3. Instancia o AE base e carrega os pesos aprendidos (reconstrução)
        base_ae = VibNetAutoencoder()
        base_ae.load_state_dict(torch.load(ae_w_path, map_location=DEVICE))
        
        # 4. Transforma o AE num Classificador
        # freeze_encoder=True -> O encoder não muda, treinamos APENAS a camada linear final
        model = VibNetFeatureExtractor(base_ae, num_classes, freeze_encoder=True).to(DEVICE)
        
        # 5. Otimizador apenas para os parâmetros que requerem gradiente (camada final)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 6. Roda a avaliação usando o mesmo motor de treino
        metrics = train_target_fold(model, dataloaders, optimizer, criterion, epochs=8)
        
        print(f"      Resultado AE Extractor: Bal Acc: {metrics['Bal Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")
        
        results.append({
            "Virtual Group": group_name,
            "Strategy": "AE_Feature_Extraction",
            **metrics
        })
            
    return results

if __name__ == "__main__":
    print("\n" + "="*50)
    print("INICIANDO EXPERIMENTO AUTOENCODER UORED (VIRTUAL GROUPS)")
    print("="*50)
    
    res = run_ae_uored_experiment()
    
    if res:
        df = pd.DataFrame(res)
        print("\n--- RESUMO FINAL UORED (AUTOENCODER) ---")
        print(df.groupby("Strategy")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std']).to_string())
