import os
import sys
import cv2
import zipfile
import gdown
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import vibdata.raw as raw_datasets
from vibdata.deep.signal.transforms import Sequential

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import BASE_DIR
from core.transforms import Detrend, SimpleSplit, OttawaSpectrogram

# --- CAMINHOS DE DIRETÓRIO ---
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
FINAL_IMG_DIR = os.path.join(BASE_DIR, "dataset_final")

# --- PIPELINES ---
PIPELINES = {
    "CWRU_12k": Sequential([Detrend(), SimpleSplit(window_size=3000), OttawaSpectrogram(nperseg=200, noverlap=int(200*0.96), nfft=1600)]),
    "CWRU_48k": Sequential([Detrend(), SimpleSplit(window_size=12000), OttawaSpectrogram(nperseg=200, noverlap=int(200*0.96), nfft=1600)]),
    "HUST": Sequential([Detrend(), SimpleSplit(window_size=12800), OttawaSpectrogram(nperseg=200, noverlap=int(200*0.96), nfft=1600)]),
    "UORED": Sequential([Detrend(), SimpleSplit(window_size=10500), OttawaSpectrogram(nperseg=180, noverlap=int(180*0.96), nfft=1600)]),
    "PU": Sequential([Detrend(), SimpleSplit(window_size=16000), OttawaSpectrogram(nperseg=180, noverlap=int(180*0.96), nfft=1600)])
}

# --- FUNÇÕES DE ROTULAGEM ---
def get_names(ds_name, meta):
    if "CWRU" in ds_name:
        load = meta.get('load', 0)
        try: load = int(load)
        except: load = 0
        cond = f"Load_{load}HP"
    elif ds_name == "PU":
        fname = str(meta.get('file_name', ''))
        speed_code = fname[:3]
        torque, radial = meta.get('load_nm', 0), meta.get('radial_force_n', 0)
        if speed_code == "N15" and torque == 0.7 and radial == 1000: cond = "C1_1500rpm_0.7Nm_1000N"
        elif speed_code == "N09" and torque == 0.7 and radial == 1000: cond = "C2_900rpm_0.7Nm_1000N"
        elif speed_code == "N15" and torque == 0.1 and radial == 1000: cond = "C3_1500rpm_0.1Nm_1000N"
        elif speed_code == "N15" and torque == 0.7 and radial == 400: cond = "C4_1500rpm_0.7Nm_400N"
        else: cond = f"Cx_Other_{speed_code}_{torque}Nm"
    elif ds_name == "HUST":
        load_w = meta.get('load_W', 0)
        cond = f"Load_{load_w}W"
    elif ds_name == "UORED":
        bid = meta.get('bearing_id', meta.get('bearing.id', 'Unknown'))
        cond = f"Bearing_{bid}"
        if meta.get('stage', 'unknown') == 'healthy':
            return cond, "Class_Normal"
    else:
        val = meta.get('load', meta.get('rotation_hz', '0'))
        cond = f"Cond_{str(val).replace('.', '')}"

    orig_label = meta.get('label')
    if isinstance(orig_label, pd.Series): orig_label = orig_label.item()
    return cond, f"Class_{orig_label}"

def extract_signal(item):
    raw = item.get('signal')
    if isinstance(raw, np.ndarray) and raw.dtype == 'O' and raw.size > 0: return raw[0]
    if isinstance(raw, np.ndarray): return raw
    return None

# =====================================================================
# ETAPA DE EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    MASTER_ZIP_LINK = "https://drive.google.com/drive/folders/1QTzuAWcyKtTjHFU9o2OfK2Gp1x9Swihe?usp=drive_link"
    PASTA_DOWNLOAD_ZIPS = os.path.join(RAW_DATA_DIR, "zips_baixados")
    os.makedirs(PASTA_DOWNLOAD_ZIPS, exist_ok=True)

    # A: Download da nuvem
    if len(os.listdir(PASTA_DOWNLOAD_ZIPS)) == 0:
        print("📥 Baixando arquivos do Google Drive...")
        try:
            gdown.download_folder(url=MASTER_ZIP_LINK, output=PASTA_DOWNLOAD_ZIPS, quiet=False)
            print("✅ Download concluído.")
        except Exception as e: print(f"❌ Falha no download: {e}")
    else:
        print("💾 Arquivos zip detectados localmente.")

    # B: Processamento
    for ds_name in ["CWRU", "HUST", "PU", "UORED"]:
        print(f"\n=== Processando {ds_name} ===")
        pasta_destino = os.path.join(RAW_DATA_DIR, f"{ds_name}_raw")
        os.makedirs(pasta_destino, exist_ok=True)

        arquivos_zip = [f for f in os.listdir(PASTA_DOWNLOAD_ZIPS) if f.lower().endswith('.zip') and ds_name in f]
        if arquivos_zip and len(os.listdir(pasta_destino)) == 0:
            print(f"📦 Extraindo {len(arquivos_zip)} arquivos...")
            for zip_name in arquivos_zip:
                try:
                    with zipfile.ZipFile(os.path.join(PASTA_DOWNLOAD_ZIPS, zip_name), 'r') as zip_ref:
                        zip_ref.extractall(pasta_destino)
                except Exception as e: print(f"⚠️ Erro ao extrair {zip_name}: {e}")
            
            # Corrige pasta dupla
            subpasta = os.path.join(pasta_destino, f"{ds_name}_raw")
            if os.path.isdir(subpasta):
                for arquivo in os.listdir(subpasta):
                    shutil.move(os.path.join(subpasta, arquivo), pasta_destino)
                os.rmdir(subpasta)
        
        try:
            raw_cls = getattr(raw_datasets, f"{ds_name}_raw")
            ds = raw_cls(RAW_DATA_DIR, download=False)
        except Exception as e: 
            print(f"❌ Erro ao inicializar {ds_name}: {e}"); continue

        saved_count = {}
        for i in tqdm(range(len(ds))):
            try:
                item = ds[i]
                if not isinstance(item, dict): continue
                sig_array = extract_signal(item)
                if sig_array is None: continue

                meta = item['metainfo']
                if isinstance(meta, pd.DataFrame): meta = meta.iloc[0]

                target_ds_name = ds_name
                if ds_name == "CWRU":
                    target_ds_name = "CWRU_48k" if meta.get('sample_rate', 12000) > 20000 else "CWRU_12k"

                current_transform = PIPELINES.get(target_ds_name)
                if not current_transform: continue

                save_path = os.path.join(FINAL_IMG_DIR, target_ds_name)
                processed = current_transform({"signal": sig_array, "metainfo": pd.DataFrame([meta])})
                imgs = processed["signal"]

                if isinstance(imgs, list) and len(imgs) > 0:
                    cond, lbl = get_names(target_ds_name, meta)
                    final_dir = os.path.join(save_path, cond, lbl)
                    os.makedirs(final_dir, exist_ok=True)

                    for idx, img in enumerate(imgs):
                        if isinstance(img, np.ndarray):
                            fname = f"s{i:05d}_w{idx:02d}.png"
                            cv2.imwrite(os.path.join(final_dir, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    saved_count[target_ds_name] = saved_count.get(target_ds_name, 0) + len(imgs)
            except Exception: continue

        print(f"--> Status: {saved_count}")
