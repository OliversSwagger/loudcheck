# audio_analysis.py
import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.signal import resample

def analyze_audio(file_path: str):
    # Load audio (stereo if available)
    y, sr = librosa.load(file_path, sr=None, mono=False)

    if y.shape[-1] < sr:
        raise ValueError("Audio too short for analysis")

    # Mono 2D for stereo processing
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    # -----------------------
    # Stereo balance (L-R)
    # -----------------------
    if y.shape[0] == 2:
        left_rms = np.sqrt(np.mean(y[0] ** 2))
        right_rms = np.sqrt(np.mean(y[1] ** 2))
        stereo_balance = float((right_rms - left_rms) / max(left_rms, right_rms))
        y_mono = np.mean(y, axis=0)
    else:
        stereo_balance = None
        y_mono = y[0]

    # -----------------------
    # Loudness meter
    # -----------------------
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_mono)
    lra = meter.loudness_range(y_mono)

    # -----------------------
    # True Peak: upsample AES-recommended ≥192 kHz
    # -----------------------
    upsample_factor = max(4, int(192000 / sr))
    y_upsampled = resample(y_mono, len(y_mono) * upsample_factor)
    y_upsampled = np.asarray(y_upsampled).flatten()
    true_peak_db = 20 * np.log10(np.max(np.abs(y_upsampled)) + 1e-9)

    # -----------------------
    # RMS
    # -----------------------
    rms = float(np.sqrt(np.mean(y_mono ** 2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))

    # -----------------------
    # Frequency bands
    # -----------------------
    stft = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    low = float(stft[freqs < 80].mean())
    mid = float(stft[(freqs >= 80) & (freqs < 2000)].mean())
    high = float(stft[freqs >= 2000].mean())
    total = max(low + mid + high, 1e-9)

    # -----------------------
    # Clipping detection
    # -----------------------
    clipping = bool(np.max(np.abs(y_mono)) >= 1.0)

    # -----------------------
    # Short-term dynamic range (1-second LUFS windows)
    # -----------------------
    hop_len = max(1, int(sr))
    window_lufs = []
    for start in range(0, len(y_mono), hop_len):
        segment = y_mono[start:start + hop_len]
        if len(segment) == 0:
            continue
        try:
            l = meter.integrated_loudness(segment)
            if np.isfinite(l):
                window_lufs.append(float(l))
        except Exception:
            continue

    short_term_dr = float(max(window_lufs) - min(window_lufs)) if len(window_lufs) >= 2 else None

    # -----------------------
    # PLR (Peak-to-Loudness Ratio)
    # -----------------------
    plr = float(true_peak_db - loudness)

    # -----------------------
    # Frequency balance %
    # -----------------------
    freq_balance = {
        "low": round((low / total) * 100, 1),
        "mid": round((mid / total) * 100, 1),
        "high": round((high / total) * 100, 1),
    }

    # -----------------------
    # Content classification
    # -----------------------
    content_type = classify_content(loudness, lra, freq_balance)

    # -----------------------
    # Distribution simulation
    # -----------------------
    distribution_simulation = simulate_distribution(loudness, true_peak_db)

    return {
        "integrated_lufs": round(float(loudness), 2),
        "true_peak_db": round(float(true_peak_db), 2),
        "rms_db": round(float(rms_db), 2),
        "loudness_range": round(float(lra), 2),
        "plr": round(plr, 2),
        "frequency_balance": freq_balance,
        "clipping": clipping,
        "stereo_balance": None if stereo_balance is None else round(stereo_balance, 2),
        "short_term_dynamic_range": None if short_term_dr is None else round(short_term_dr, 2),
        "content_type": content_type,
        "distribution_simulation": distribution_simulation
    }


# -----------------------
# AES Helper functions
# -----------------------
def classify_content(loudness, lra, freq_balance):
    """AES-inspired heuristic classification"""
    if lra > 14:
        return "wide_dynamic_range"
    if freq_balance["mid"] > 45 and freq_balance["low"] < 25:
        return "speech_dominant"
    if loudness > -16 and lra < 8:
        return "dense_music"
    return "mixed"

def simulate_distribution(lufs, true_peak):
    """Predict what platforms will do (AES-style)"""
    platforms = {}

    # Spotify / Apple Music (-14 LUFS reference)
    delta = -14 - lufs
    platforms["music_streaming"] = {
        "gain_change_db": round(delta, 2),
        "will_limit": bool(delta > 0 and (true_peak + delta) > -1.0)
    }

    # Radio / speech (-18 LUFS)
    delta_radio = -18 - lufs
    platforms["speech_streaming"] = {
        "gain_change_db": round(delta_radio, 2),
        "will_limit": bool(delta_radio > 0 and (true_peak + delta_radio) > -1.0)
    }

    return platforms
