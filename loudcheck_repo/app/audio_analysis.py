import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.signal import resample

def analyze_audio(file_path: str):
    # Load audio (mono)
    y, sr = librosa.load(file_path, sr=None, mono=True)

    if len(y) < sr:
        raise ValueError("Audio too short for analysis")

    # Loudness meter
    meter = pyln.Meter(sr)

    # Integrated LUFS
    loudness = meter.integrated_loudness(y)

    # Loudness Range (LRA)
    lra = meter.loudness_range(y)

    # True Peak: upsample by 4x (robust)
    upsample_factor = 4
    y_upsampled = resample(y, len(y) * upsample_factor)

    # Ensure 1D float array
    y_upsampled = np.asarray(y_upsampled).flatten().astype(float)

    # Compute True Peak in dBTP
    true_peak_db = 20 * np.log10(np.max(np.abs(y_upsampled)) + 1e-9)

    # RMS
    rms = float(np.sqrt(np.mean(y**2)))
    rms_db = float(20 * np.log10(max(rms, 1e-9)))

    # Frequency bands
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    low = float(stft[freqs < 80].mean())
    mid = float(stft[(freqs >= 80) & (freqs < 2000)].mean())
    high = float(stft[freqs >= 2000].mean())

    total = max(low + mid + high, 1e-9)

    return {
        "integrated_lufs": round(loudness, 2),
        "true_peak_db": round(true_peak_db, 2),
        "rms_db": round(rms_db, 2),
        "loudness_range": round(lra, 2),
        "frequency_balance": {
            "low": round((low / total) * 100, 1),
            "mid": round((mid / total) * 100, 1),
            "high": round((high / total) * 100, 1),
        }
    }
