import os
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

class VibNetAutoencoder(nn.Module):
    def __init__(self):
        super(VibNetAutoencoder, self).__init__()
        # ENCODER (Comprime a imagem para o espaço latente)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # DECODER (Reconstrói a imagem a partir do espaço latente)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Saída entre 0 e 1 (assumindo imagens normalizadas)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
        
class VibNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained_ae, num_classes, freeze_encoder=True):
        super(VibNetFeatureExtractor, self).__init__()
        
        # 1. Copia APENAS o Encoder do Autoencoder pré-treinado
        self.encoder = pretrained_ae.encoder
        
        # 2. Congela os pesos do Encoder (Feature Extraction puro)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # 3. Cria o cabeçote de classificação
        # Usamos AdaptiveAvgPool2d para esmagar a dimensão espacial (A x L) para 1x1,
        # mantendo apenas os 128 canais de profundidade independentemente do tamanho da imagem.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3), # Evita overfitting no domínio alvo
            nn.Linear(128, num_classes) # 128 é a saída da última Conv2d do seu encoder
        )

    def forward(self, x):
        # Passa a imagem pelo encoder congelado
        features = self.encoder(x)
        # Esmaga espacialmente e passa pelo classificador novo
        pooled = self.pool(features)
        out = self.classifier(pooled)
        return out
