import os
import sys
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Garante que o Python ache a pasta core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import DEVICE, RESULTS_DIR
from core.utils import Logger
from core.data import get_target_splits, get_dataloaders
from core.models import get_vibnet_model
from core.engine import pre_train_source, train_target_fold

# Configura Logger
sys.stdout = Logger(os.path.join(RESULTS_DIR, "pu_unified_results.txt"))

def run_pu_experiment():
    # 1. PRÉ-TREINO (Vai pular automaticamente pois já rodou e salvou!)
    w_imagenet = pre_train_source(target_to_exclude="PU", start_with_imagenet=True)
    w_scratch = pre_train_source(target_to_exclude="PU", start_with_imagenet=False)

    # 2. DEFINIR ESTRUTURA DO ALVO
    conditions = ["C1", "C2", "C3", "C4"] 
    strategies = ["Scratch", "ImageNet", "VibNet_from_Scratch", "VibNet_from_ImageNet"]
    results = []

    # 3. LOOP DE FOLDS (LODO-CV)
    for test_cond in conditions:
        print(f"\n--- Fold: Testando em {test_cond} ---")
        
        # Coleta os splits corretamente
        train_x, train_y, test_x, test_y, num_classes = get_target_splits(dataset_name="PU", test_condition=test_cond)
        
        if len(train_x) == 0:
            print(f"AVISO: Dados vazios para a condição {test_cond}. Verifique os caminhos.")
            continue

        # Monta os tensores
        dataloaders = get_dataloaders(train_x, train_y, test_x, test_y)

        for strat in strategies:
            print(f"   -> Estratégia: {strat}")
            
            # Pega o peso correto baseado no nome da estratégia
            w_path = w_imagenet if "ImageNet" in strat else w_scratch
            
            # Instancia o modelo
            model = get_vibnet_model(num_classes, strat, w_path)
            
            # Configura otimizador (LR menor para transfer learning)
            lr = 0.001 if strat == "Scratch" else 0.0001
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Roda o motor de treino
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
