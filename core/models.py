import torch
import torch.nn as nn
from torchvision import models
from .config import DEVICE

def get_vibnet_model(num_classes, strategy, weights_path=None):
    """
    Constrói a DenseNet121 baseada na estratégia (Scratch, ImageNet, ou VibNet).
    """
    model = models.densenet121(weights=None)
    n_feat = model.classifier.in_features
    
    # Carrega base ImageNet se a estratégia exigir
    if strategy in ["ImageNet", "VibNet_from_ImageNet"]:
        temp_model = models.densenet121(weights='IMAGENET1K_V1')
        model.features = temp_model.features

    # Carrega pesos pré-treinados do Source Domain (VibNet)
    if "VibNet" in strategy:
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=DEVICE)
            curr_dict = model.state_dict()
            
            # Filtra os pesos da última camada (classifier) que terá tamanho diferente
            pretrained_dict = {k: v for k, v in state_dict.items() if k in curr_dict and 'classifier' not in k}
            curr_dict.update(pretrained_dict)
            model.load_state_dict(curr_dict)
        else:
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")

    # Ajusta a última camada para o Target Domain atual
    model.classifier = nn.Linear(n_feat, num_classes)
    return model.to(DEVICE)
