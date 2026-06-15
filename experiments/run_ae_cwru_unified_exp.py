import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Silencia o aviso inofensivo de classes ausentes do Scikit-Learn
warnings.filterwarnings("ignore", category=UserWarning)

# Garante que o Python encontre a pasta core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import DEVICE, RESULTS_DIR
from core.utils import Logger
from core.data import get_target_splits, get_dataloaders
from core.models import VibNetAutoencoder, VibNetFeatureExtractor
from core.engine import pre_train_autoencoder_source, train_target_fold

# Configura o Logger
sys.stdout = Logger(os.path.join(RESULTS_DIR, "ae_cwru_unified_results.txt"))

def run_ae_cwru_experiment():
    # 1. PRÉ-TREINO (Não-supervisionado)
    # Excluímos as duas sub-bases do CWRU para não haver Data Leakage
    targets_to_exclude = ["CWRU_12k", "CWRU_48k"]
    
    print("Verificando/Iniciando pré-treinamento do Autoencoder (Source)...")
    ae_w_path = pre_train_autoencoder_source(target_to_exclude=targets_to_exclude)
    
    # 2. DEFINIÇÃO DE CARGAS DO CWRU (LODO-CV)
    conditions = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    datasets = ["CWRU_12k", "CWRU_48k"]
    results = []

    # 3. LOOP SOBRE OS DATASETS E CARGAS
    for ds_name in datasets:
        print(f"\n{'='*50}\nAVALIANDO DATASET: {ds_name}\n{'='*50}")
        
        for test_cond in conditions:
            print(f"\n--- Fold: Testando em {test_cond} (Autoencoder) ---")
            
            # Pega os splits da carga atual
            train_x, train_y, test_x, test_y, num_classes = get_target_splits(
                dataset_name=ds_name, 
                test_condition=test_cond
            )
            
            if len(test_x) == 0 or len(train_x) == 0:
                print(f"   [AVISO] Dados insuficientes em {test_cond}. Treino: {len(train_x)} | Teste: {len(test_x)}")
                continue

            print(f"   [Dados] Treino: {len(train_x)} imgs | Teste: {len(test_x)} imgs")
            dataloaders = get_dataloaders(train_x, train_y, test_x, test_y)

            # Instancia AE base e carrega pesos de reconstrução
            base_ae = VibNetAutoencoder()
            base_ae.load_state_dict(torch.load(ae_w_path, map_location=DEVICE))
            
            # Transforma em classificador. 
            # freeze_encoder=True -> Feature Extraction puro
            model = VibNetFeatureExtractor(base_ae, num_classes, freeze_encoder=True).to(DEVICE)
            
            # Otimizador apenas para o classificador linear
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            metrics = train_target_fold(model, dataloaders, optimizer, criterion, epochs=8)
            
            print(f"      Resultado AE Extractor: Bal Acc: {metrics['Bal Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")
            
            results.append({
                "Dataset": ds_name,
                "Test Condition": test_cond,
                "Strategy": "AE_Feature_Extraction",
                **metrics
            })
            
    return results

if __name__ == "__main__":
    print("\nINICIANDO EXPERIMENTO AUTOENCODER CWRU UNIFICADO")
    res = run_ae_cwru_experiment()
    
    if res:
        df = pd.DataFrame(res)
        print("\n--- RESUMO FINAL CWRU (AUTOENCODER) ---")
        print(df.groupby(["Dataset", "Strategy"])[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std']).to_string())
