import os
import cv2
import zipfile
import gdown
import shutil
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm
import vibdata.raw as raw_datasets
from vibdata.deep.signal.transforms import Sequential, Transform
from scipy.signal import stft, detrend

# --- CLASSES AUXILIARES (MANTIDAS) ---
class SimpleSplit(Transform):
    def __init__(self, window_size=2048, overlap=0):
        super().__init__()
        self.window_size = window_size
        self.step = window_size - overlap
    def transform(self, data):
        data = data.copy()
        sig = data['signal']
        if isinstance(sig, list): sig = sig[0]
        if isinstance(sig, np.ndarray): sig = sig.flatten()
        windows = []
        if len(sig) >= self.window_size:
            for i in range(0, len(sig) - self.window_size + 1, self.step):
                windows.append(sig[i : i + self.window_size])
        data['signal'] = windows
        return data

class Detrend(Transform):
    def transform(self, data):
        data = data.copy()
        sig = data['signal']
        if isinstance(sig, np.ndarray):
            sig = sig.flatten()
            data['signal'] = detrend(sig, type='linear')
        elif isinstance(sig, list):
            data['signal'] = [detrend(s.flatten(), type='linear') if isinstance(s, np.ndarray) else s for s in sig]
        return data

class OttawaSpectrogram(Transform):
    def __init__(self, window="hann", nperseg=200, noverlap=None, nfft=1600):
        super().__init__()
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else int(nperseg * 0.96)
        self.nfft = nfft
    def transform(self, data):
        data = data.copy()
        signals = data["signal"]
        metainfo = data["metainfo"]
        if not isinstance(signals, list) or len(signals) == 0: return data
        if len(metainfo) == 1 and len(signals) > 1:
            metainfo = pd.concat([metainfo]*len(signals), ignore_index=True)
        ret, new_metainfo = [], []
        for i, sig in enumerate(signals):
            try:
                entry = metainfo.iloc[0]
                fs = entry.get("sample_rate", 12000)
                if np.max(sig) == np.min(sig): continue
                f, t, Sxx = stft(sig, fs=fs, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
                distances = np.abs(f - 10000)
                max_bound = np.argmin(distances)
                Sxx_seg = np.abs(Sxx[: max_bound+1, :])**2
                float_mat = np.log(Sxx_seg + 1e-10)
                min_val, max_val = float_mat.min(), float_mat.max()
                denom = max_val - min_val
                if denom > 1e-5: gray = ((float_mat - min_val) / denom) * 255
                else: gray = np.zeros_like(float_mat)
                gray = cv2.resize(gray, (512, 256), interpolation=cv2.INTER_CUBIC)
                bgr = cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_JET)
                ret.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                new_metainfo.append(entry)
            except: continue
        data["signal"] = ret
        data["metainfo"] = pd.DataFrame(new_metainfo)
        return data

# --- PIPELINES ATUALIZADOS (CWRU DIVIDIDO) ---
PIPELINES = {
    # CWRU 12k: ~0.25s = 3000 pontos (12000 Hz * 0.25)
    "CWRU_12k": Sequential([
        Detrend(), SimpleSplit(window_size=3000),
        OttawaSpectrogram(nperseg=200, noverlap=int(200*0.96), nfft=1600)
    ]),

    # CWRU 48k: ~0.25s = 12000 pontos (48000 Hz * 0.25)
    "CWRU_48k": Sequential([
        Detrend(), SimpleSplit(window_size=12000),
        OttawaSpectrogram(nperseg=200, noverlap=int(200*0.96), nfft=1600)
    ]),

    "HUST": Sequential([
        Detrend(), SimpleSplit(window_size=12800),
        OttawaSpectrogram(nperseg=200, noverlap=int(200*0.96), nfft=1600)
    ]),
    "UORED": Sequential([
        Detrend(), SimpleSplit(window_size=10500),
        OttawaSpectrogram(nperseg=180, noverlap=int(180*0.96), nfft=1600)
    ]),
    "PU": Sequential([
        Detrend(), SimpleSplit(window_size=16000),
        OttawaSpectrogram(nperseg=180, noverlap=int(180*0.96), nfft=1600)
    ])
}

# --- FUNÇÃO DE NOMES (CWRU e UORED Ajustados) ---
def get_names(ds_name, meta):
    # Nota: ds_name pode vir como "CWRU_12k" ou "CWRU_48k", tratamos igual
    if "CWRU" in ds_name:
        # --- CORREÇÃO PARA REFLETIR O ARTIGO ---
        # Artigo: "Split by motor load (0, 1, 2, and 3 HP)"
        # Ignoramos a severidade na criação da PASTA DE CONDIÇÃO.
        # A severidade será implícita na classe (label) ou misturada dentro da pasta.
        
        load = meta.get('load', 0)
        # Garante inteiro para ficar bonito no nome da pasta
        try:
            load = int(load)
        except:
            load = 0
            
        cond = f"Load_{load}HP"

    elif ds_name == "PU":
        fname = str(meta.get('file_name', ''))
        speed_code = fname[:3]
        torque = meta.get('load_nm', 0)
        radial = meta.get('radial_force_n', 0)
        if speed_code == "N15" and torque == 0.7 and radial == 1000: cond = "C1_1500rpm_0.7Nm_1000N"
        elif speed_code == "N09" and torque == 0.7 and radial == 1000: cond = "C2_900rpm_0.7Nm_1000N"
        elif speed_code == "N15" and torque == 0.1 and radial == 1000: cond = "C3_1500rpm_0.1Nm_1000N"
        elif speed_code == "N15" and torque == 0.7 and radial == 400:  cond = "C4_1500rpm_0.7Nm_400N"
        else: cond = f"Cx_Other_{speed_code}_{torque}Nm"

    elif ds_name == "HUST":
        load_w = meta.get('load_W', 0)
        cond = f"Load_{load_w}W"

    elif ds_name == "UORED":
        # Estratégia Cross-Domain: Condição = Bearing ID
        bid = meta.get('bearing_id', meta.get('bearing.id', 'Unknown'))
        cond = f"Bearing_{bid}"
        stage = meta.get('stage', 'unknown')
        if stage == 'healthy':
            return cond, "Class_Normal"

    else:
        val = meta.get('load', meta.get('rotation_hz', '0'))
        cond = f"Cond_{str(val).replace('.', '')}"

    orig_label = meta.get('label')
    if isinstance(orig_label, pd.Series): orig_label = orig_label.item()
    label_name = f"Class_{orig_label}"
    return cond, label_name

# Pega automaticamente o diretório base do usuário atual (ex: /home/vfrocha ou /home/flavio)
USER_HOME = os.path.expanduser("~")

# Constrói o caminho relativo a esse usuário
BASE_DRIVE = os.path.join(USER_HOME, "VibNet_Project")
RAW_DATA_DIR = os.path.join(BASE_DRIVE, "raw_data")
FINAL_IMG_DIR = os.path.join(BASE_DRIVE, "dataset_final")

def extract_signal(item):
    raw = item.get('signal')
    if isinstance(raw, np.ndarray) and raw.dtype == 'O' and raw.size > 0: return raw[0]
    if isinstance(raw, np.ndarray): return raw
    return None

datasets = ["CWRU", "HUST", "PU", "UORED"]

# 1. NOVO LINK DA SUA PASTA 'zip' UNIFICADA
# Substitua o link abaixo pelo link de compartilhamento da pasta "zip" do seu drive
MASTER_ZIP_LINK = "https://drive.google.com/drive/folders/1QTzuAWcyKtTjHFU9o2OfK2Gp1x9Swihe?usp=drive_link"

# Pasta temporária onde todos os zips serão baixados juntos
PASTA_DOWNLOAD_ZIPS = os.path.join(RAW_DATA_DIR, "zips_baixados")
os.makedirs(PASTA_DOWNLOAD_ZIPS, exist_ok=True)

# =====================================================================
# ETAPA A: DOWNLOAD ÚNICO DA PASTA MASTER (FORA DO LOOP)
# =====================================================================
if len(os.listdir(PASTA_DOWNLOAD_ZIPS)) == 0:
    print("📥 Pasta de zips vazia. Baixando TODOS os arquivos do Google Drive de uma vez...")
    try:
        gdown.download_folder(
            url=MASTER_ZIP_LINK, 
            output=PASTA_DOWNLOAD_ZIPS, 
            quiet=False
            # remaining_ok=True  <-- Removido pois sua versão do gdown não suporta
        )
        print("✅ Download de todos os zips concluído.")
    except Exception as e:
        print(f"❌ Falha crítica no download da pasta Master: {e}")
else:
    print("💾 Arquivos zip já detectados localmente. Pulando download da nuvem.")

# =====================================================================
# ETAPA B: PROCESSAMENTO POR DATASET
# =====================================================================
for ds_name in datasets:
    print(f"\n=== Processando {ds_name} ===")
    
    pasta_destino = os.path.join(RAW_DATA_DIR, f"{ds_name}_raw")
    os.makedirs(pasta_destino, exist_ok=True)

    # 2. EXTRAÇÃO INTELIGENTE DOS ARQUIVOS COMPACTADOS
    # Procura na pasta de downloads apenas os zips que pertencem a este dataset (ex: contém "CWRU")
    arquivos_baixados = os.listdir(PASTA_DOWNLOAD_ZIPS)
    arquivos_zip_dataset = [f for f in arquivos_baixados if f.lower().endswith('.zip') and ds_name in f]

    if arquivos_zip_dataset:
        if len(os.listdir(pasta_destino)) == 0:
            print(f"📦 Extraindo {len(arquivos_zip_dataset)} arquivos compactados para {ds_name}...")
            for zip_name in arquivos_zip_dataset:
                caminho_zip = os.path.join(PASTA_DOWNLOAD_ZIPS, zip_name)
                try:
                    with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                        zip_ref.extractall(pasta_destino)
                except Exception as e:
                    print(f"⚠️ Erro ao extrair o arquivo {zip_name}: {e}")
            
            # --- CORREÇÃO DA PASTA DUPLA DO GOOGLE DRIVE ---
            # Verifica se ao extrair, foi criada uma pasta com o mesmo nome dentro do destino
            subpasta_provavel = os.path.join(pasta_destino, f"{ds_name}_raw")
            if os.path.isdir(subpasta_provavel):
                print(f"🔧 Corrigindo estrutura de pastas duplas para {ds_name}...")
                # Move todos os arquivos de dentro da subpasta para a pasta raiz correta
                for arquivo in os.listdir(subpasta_provavel):
                    shutil.move(os.path.join(subpasta_provavel, arquivo), pasta_destino)
                # Remove a subpasta vazia para manter a limpeza
                os.rmdir(subpasta_provavel)
                
            print(f"✨ Estrutura de dados descompactada com sucesso.")
        else:
            print(f"📦 Dados de {ds_name} já estavam descompactados.")
    else:
        print(f"⚠️ Nenhum arquivo zip encontrado para {ds_name} na pasta de downloads.")

    # 3. CARREGAMENTO DOS DADOS PELA BIBLIOTECA VIBDATA
    try:
        raw_cls = getattr(raw_datasets, f"{ds_name}_raw")
        ds = raw_cls(RAW_DATA_DIR, download=False)
    except Exception as e: 
        print(f"❌ ERRO GRAVE ao inicializar a base {ds_name} no vibdata: {e}")
        continue

    # Contadores separados para CWRU
    saved_count = {}
    
    for i in tqdm(range(len(ds))):
        try:
            item = ds[i]
            if not isinstance(item, dict): continue

            sig_array = extract_signal(item)
            if sig_array is None: continue

            meta = item['metainfo']
            if isinstance(meta, pd.DataFrame): meta = meta.iloc[0]

            # --- LÓGICA DE SEPARAÇÃO CWRU ---
            target_ds_name = ds_name
            if ds_name == "CWRU":
                sr = meta.get('sample_rate', 12000)
                # Define qual sub-base estamos processando
                if sr > 20000:
                    target_ds_name = "CWRU_48k"
                else:
                    target_ds_name = "CWRU_12k"

            # Pega o pipeline correto
            current_transform = PIPELINES.get(target_ds_name)
            if not current_transform: continue # Segurança

            # Cria a pasta de destino (CWRU_12k ou CWRU_48k ou HUST...)
            save_path = os.path.join(FINAL_IMG_DIR, target_ds_name)
            os.makedirs(save_path, exist_ok=True)

            # Aplica Transformação
            sample = {"signal": sig_array, "metainfo": pd.DataFrame([meta])}
            processed = current_transform(sample)

            imgs = processed["signal"]
            if isinstance(imgs, list) and len(imgs) > 0:
                cond, lbl = get_names(target_ds_name, meta)
                final_dir = os.path.join(save_path, cond, lbl)
                os.makedirs(final_dir, exist_ok=True)

                for idx, img in enumerate(imgs):
                    if isinstance(img, np.ndarray):
                        fname = f"s{i:05d}_w{idx:02d}.png"
                        cv2.imwrite(os.path.join(final_dir, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Atualiza contagem
                saved_count[target_ds_name] = saved_count.get(target_ds_name, 0) + len(imgs)

        except Exception: continue

    print(f"--> Status: {saved_count}")
