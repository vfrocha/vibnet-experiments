import os
import sys
import pandas as pd
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import RESULTS_DIR
from core.utils import Logger
from core.data import get_target_splits, get_dataloaders
from core.models import get_vibnet_model
from core.engine import pre_train_source, train_target_fold

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

sys.stdout = Logger(os.path.join(RESULTS_DIR, "uored_unified_results.txt"))

def run_uored_experiment():
    # 1. PRÉ-TREINO (Excluindo UORED da base de conhecimentos)
    w_imagenet = pre_train_source(target_to_exclude="UORED", start_with_imagenet=True)
    w_scratch = pre_train_source(target_to_exclude="UORED", start_with_imagenet=False)

    strategies = ["Scratch", "ImageNet", "VibNet_from_Scratch", "VibNet_from_ImageNet"]
    results = []

    # 2. LOOP SOBRE OS GRUPOS VIRTUAIS
    for group_name, bearings_list in VIRTUAL_CONDITIONS.items():
        print(f"\n{'='*40}\nFold: Testando no {group_name}\n{'='*40}")
        
        # Passamos a lista de rolamentos e o mapa forçado
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
                "Virtual Group": group_name, "Strategy": strat, **metrics
            })
            
    return results

if __name__ == "__main__":
    print("\nINICIANDO EXPERIMENTO UORED (VIRTUAL GROUPS)")
    res = run_uored_experiment()
    
    if res:
        df = pd.DataFrame(res)
        print("\n--- RESUMO FINAL UORED ---")
        print(df.groupby("Strategy")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std']).to_string())
