import os
import torch

# --- CAMINHOS ABSOLUTOS ---
# Sobe um nível em relação à pasta 'core' para chegar na raiz do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_FINAL = os.path.join(BASE_DIR, "dataset_final")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Garante que as pastas de saída existam
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- HIPERPARÂMETROS GLOBAIS ---
IMG_SIZE = 224
BATCH_SIZE = 32

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
