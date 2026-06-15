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
    targets_to_exclude = ["CWRU_12k", "CWRU_48k"]
    print("Verificando/Iniciando pré-treinamento do Autoencoder (Source)...")
    ae_w_path = pre_train_autoencoder_source(target_to_exclude=targets_to_exclude)
    
    conditions = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    datasets = ["CWRU_12k", "CWRU_48k"]
    
    # ---> AS TRÊS ESTRATÉGIAS DO AUTOENCODER <---
    strategies = ["AE_Scratch", "AE_FineTuning", "AE_Feature_Extraction"]
    results = []

    for ds_name in datasets:
        print(f"\n{'='*50}\nAVALIANDO DATASET: {ds_name}\n{'='*50}")
        
        for test_cond in conditions:
            print(f"\n--- Fold: Testando em {test_cond} ---")
            
            train_x, train_y, test_x, test_y, num_classes = get_target_splits(dataset_name=ds_name, test_condition=test_cond)
            
            if len(test_x) == 0 or len(train_x) == 0:
                print(f"   [AVISO] Dados insuficientes em {test_cond}.")
                continue

            dataloaders = get_dataloaders(train_x, train_y, test_x, test_y)

            for strat in strategies:
                print(f"   -> Estratégia: {strat}")
                
                # 1. Instancia o AE Base
                base_ae = VibNetAutoencoder()
                
                # 2. Carrega pesos pré-treinados APENAS se não for Scratch
                if strat in ["AE_FineTuning", "AE_Feature_Extraction"]:
                    base_ae.load_state_dict(torch.load(ae_w_path, map_location=DEVICE))
                
                # 3. Define se vai congelar o encoder
                is_frozen = (strat == "AE_Feature_Extraction")
                model = VibNetFeatureExtractor(base_ae, num_classes, freeze_encoder=is_frozen).to(DEVICE)
                
                # 4. Taxa de aprendizado: menor para Fine-Tuning para não destruir os pesos pré-treinados
                lr = 0.001 if strat == "AE_Scratch" else 0.0001
                
                # O filter garante que o otimizador só atualize o que não está congelado
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                criterion = nn.CrossEntropyLoss()
                
                # 5. Roda o treino
                metrics = train_target_fold(model, dataloaders, optimizer, criterion, epochs=8)
                
                print(f"      Result: Bal Acc: {metrics['Bal Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")
                
                results.append({
                    "Dataset": ds_name,
                    "Test Condition": test_cond,
                    "Strategy": strat,
                    **metrics
                })
            
    return results

if __name__ == "__main__":
    print("\nINICIANDO EXPERIMENTO AUTOENCODER CWRU UNIFICADO (MULTI-STRATEGY)")
    res = run_ae_cwru_experiment()
    
    if res:
        df = pd.DataFrame(res)
        print("\n--- RESUMO FINAL CWRU (AUTOENCODER) ---")
        print(df.groupby(["Dataset", "Strategy"])[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std']).to_string())
