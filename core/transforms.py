import numpy as np
import pandas as pd
import cv2
from scipy.signal import stft, detrend
from vibdata.deep.signal.transforms import Transform

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
