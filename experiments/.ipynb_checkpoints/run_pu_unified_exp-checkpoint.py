import os
import sys
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Garante que o Python ache a pasta core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import DEVICE, RESULTS_DIR, DATASET_FINAL
from core.utils import Logger
from core.data import get_target_splits, get_dataloaders
from core.models import get_vibnet_model
from core.engine import pre_train_source, train_target_fold

# Configura Logger
sys.stdout = Logger(os.path.join(RESULTS_DIR, "pu_unified_results.txt"))

def run_pu_experiment():
    w_imagenet = pre_train_source(target_to_exclude="PU", start_with_imagenet=True)
    w_scratch = pre_train_source(target_to_exclude="PU", start_with_imagenet=False)

    # LÊ AS CONDIÇÕES DINAMICAMENTE DIRETO DA PASTA
    pu_root = os.path.join(DATASET_FINAL, "PU")
    conditions = sorted([d for d in os.listdir(pu_root) if os.path.isdir(os.path.join(pu_root, d))])
    
    strategies = ["Scratch", "ImageNet", "VibNet_from_Scratch", "VibNet_from_ImageNet"]
    results = []

    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} ---")
        
        train_x, train_y, test_x, test_y, num_classes = get_target_splits(dataset_name="PU", test_condition=test_cond)
        
        # Verificação de caminho incorreto
        if len(test_x) == 0 or len(train_x) == 0:
            print(f"   [AVISO] Faltando dados na condição {test_cond}. Treino: {len(train_x)} | Teste: {len(test_x)}")
            continue

        dataloaders = get_dataloaders(train_x, train_y, test_x, test_y)

        for strat in strategies:
            print(f"   -> Estratégia: {strat}")
            
            w_path = w_imagenet if "ImageNet" in strat else w_scratch
            model = get_vibnet_model(num_classes, strat, w_path)
            
            lr = 0.001 if strat == "Scratch" else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            metrics = train_target_fold(model, dataloaders, optimizer, criterion, epochs=8)
            
            print(f"      Resultado: Bal Acc: {metrics['Bal Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")
            
            results.append({
                "Condition": test_cond, "Strategy": strat, **metrics
            })
            
    return results

if __name__ == "__main__":
    print("\n" + "="*50 + "\nINICIANDO EXPERIMENTO PU UNIFICADO\n" + "="*50)
    res = run_pu_experiment()
    
    if res:
        df = pd.DataFrame(res)
        print("\n--- RESUMO FINAL ---")
        print(df.groupby("Strategy")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std']).to_string())
