# audio_analysis.py

import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.signal import resample


def analyze_audio(file_path: str):
    """
    LoudCheck audio analysis engine.

    Handles:
    - Mono audio
    - Stereo audio
    - Multichannel audio (3+ channels)
    - Integrated LUFS
    - Loudness Range
    - True Peak
    - RMS
    - PLR
    - Frequency balance
    - Clipping detection
    - Short-term dynamic range
    - Content classification
    - Distribution simulation

    Audio files are analyzed from the supplied file path.
    """

    # =========================================================
    # LOAD AUDIO
    # =========================================================

    y, sr = librosa.load(
        file_path,
        sr=None,
        mono=False
    )

    if y is None:
        raise ValueError("Unable to load audio.")

    y = np.asarray(y, dtype=np.float64)

    if y.size == 0:
        raise ValueError("Audio file contains no audio data.")

    # ---------------------------------------------------------
    # Ensure audio is always represented as:
    #
    # channels x samples
    # ---------------------------------------------------------

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    if y.ndim != 2:
        raise ValueError(
            f"Unsupported audio shape: {y.shape}"
        )

    channel_count = int(y.shape[0])
    print("Channel count:", channel_count) # remove this later
    sample_count = int(y.shape[1])

    if sample_count < sr:
        raise ValueError(
            "Audio too short for analysis. "
            "At least one second of audio is required."
        )

    # =========================================================
    # CLEAN INVALID AUDIO VALUES
    # =========================================================

    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # =========================================================
    # CHANNEL HANDLING
    # =========================================================

    if channel_count == 1:

        # -----------------------------------------------------
        # MONO
        # -----------------------------------------------------

        stereo_balance = None

        y_mono = y[0]

    elif channel_count == 2:

        # -----------------------------------------------------
        # STEREO
        # -----------------------------------------------------

        left = y[0]
        right = y[1]

        left_rms = float(
            np.sqrt(np.mean(left ** 2))
        )

        right_rms = float(
            np.sqrt(np.mean(right ** 2))
        )

        max_rms = max(left_rms, right_rms)

        if max_rms > 1e-12:

            stereo_balance = float(
                (right_rms - left_rms) / max_rms
            )

        else:

            stereo_balance = 0.0

        # Combine stereo channels for general analysis
        y_mono = np.mean(y, axis=0)

    else:

        # -----------------------------------------------------
        # MULTICHANNEL
        # -----------------------------------------------------
        #
        # Examples:
        # 5.1
        # 7.1
        # immersive / multichannel files
        #
        # Do NOT simply use channel 1.
        # Combine all available channels.
        # -----------------------------------------------------

        stereo_balance = None

        y_mono = np.mean(y, axis=0)
    print("y_mono type:", type(y_mono))
    print("y_mono shape:", y_mono.shape if y_mono is not None else "NONE")
    print("y_mono is None:", y_mono is None)
    # Make sure the analysis signal is valid
    y_mono = np.asarray(
        y_mono,
        dtype=np.float64
    )

    y_mono = np.nan_to_num(
        y_mono,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # =========================================================
    # CHECK FOR SILENCE
    # =========================================================
    print("DEBUG: before peak", flush=True)
    peak_linear = float(
        np.max(np.abs(y_mono))
    )
    print("DEBUG: after peak", flush=True)
    if peak_linear <= 1e-12:

        raise ValueError(
            "Audio contains no measurable signal. "
            "The file may be silent or empty."
        )

    # =========================================================
    # LOUDNESS METER
    # =========================================================

    meter = pyln.Meter(sr)

    # ---------------------------------------------------------
    # Integrated LUFS
    # ---------------------------------------------------------

    try:

        loudness = meter.integrated_loudness(
            y_mono
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate integrated loudness: {exc}"
        )

    if loudness is None or not np.isfinite(loudness):

        raise ValueError(
            "Unable to calculate a valid integrated loudness value."
        )

    loudness = float(loudness)

    # =========================================================
    # LOUDNESS RANGE
    # =========================================================

    try:

        lra = meter.loudness_range(
            y_mono
        )

        if lra is None or not np.isfinite(lra):

            lra = 0.0

        else:

            lra = float(lra)

    except Exception:

        # Some very short, sparse or unusual files may not
        # provide a meaningful LRA.
        lra = 0.0

    # =========================================================
    # TRUE PEAK
    # =========================================================
    #
    # Upsample toward >=192 kHz.
    #
    # ceil() is used instead of int() so the target does not
    # accidentally remain below 192 kHz.
    # =========================================================

    upsample_factor = max(
        4,
        int(np.ceil(192000 / sr))
    )

    try:

        target_length = int(
            len(y_mono) * upsample_factor
        )

        y_upsampled = resample(
            y_mono,
            target_length
        )

        y_upsampled = np.asarray(
            y_upsampled,
            dtype=np.float64
        )

        y_upsampled = np.nan_to_num(
            y_upsampled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        peak_linear_upsampled = float(
            np.max(np.abs(y_upsampled))
        )

        if peak_linear_upsampled <= 1e-12:

            true_peak_db = -120.0

        else:

            true_peak_db = float(
                20 * np.log10(
                    peak_linear_upsampled
                )
            )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate true peak: {exc}"
        )

    # =========================================================
    # RMS
    # =========================================================

    rms = float(
        np.sqrt(
            np.mean(
                y_mono ** 2
            )
        )
    )

    if rms <= 1e-12:

        rms_db = -120.0

    else:

        rms_db = float(
            20 * np.log10(rms)
        )

    # =========================================================
    # FREQUENCY ANALYSIS
    # =========================================================

    try:

        stft = np.abs(
            librosa.stft(y_mono)
        )

        freqs = librosa.fft_frequencies(
            sr=sr
        )

        # -----------------------------------------------------
        # Frequency masks
        # -----------------------------------------------------

        low_mask = freqs < 80

        mid_mask = (
            (freqs >= 80)
            &
            (freqs < 2000)
        )

        high_mask = freqs >= 2000

        # -----------------------------------------------------
        # Calculate energy/activity for each band
        # -----------------------------------------------------

        low = (
            float(np.mean(stft[low_mask]))
            if np.any(low_mask)
            else 0.0
        )

        mid = (
            float(np.mean(stft[mid_mask]))
            if np.any(mid_mask)
            else 0.0
        )

        high = (
            float(np.mean(stft[high_mask]))
            if np.any(high_mask)
            else 0.0
        )

        total = low + mid + high

        # -----------------------------------------------------
        # Prevent division by zero
        # -----------------------------------------------------

        if total <= 1e-12:

            freq_balance = {
                "low": 0.0,
                "mid": 0.0,
                "high": 0.0
            }

        else:

            freq_balance = {
                "low": round(
                    (low / total) * 100,
                    1
                ),

                "mid": round(
                    (mid / total) * 100,
                    1
                ),

                "high": round(
                    (high / total) * 100,
                    1
                )
            }

    except Exception:

        # Frequency analysis should not cause the entire
        # LoudCheck analysis to fail.
        freq_balance = {
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0
        }

    # =========================================================
    # CLIPPING DETECTION
    # =========================================================

    clipping = bool(
        np.max(
            np.abs(y_mono)
        ) >= 1.0
    )

    # =========================================================
    # SHORT-TERM DYNAMIC RANGE
    # =========================================================
    #
    # Approximation using 1-second LUFS windows.
    #
    # Windows that cannot produce valid LUFS values are
    # ignored.
    # =========================================================

    hop_len = max(
        1,
        int(sr)
    )

    window_lufs = []

    for start in range(
        0,
        len(y_mono),
        hop_len
    ):

        segment = y_mono[
            start:start + hop_len
        ]

        if len(segment) == 0:
            continue

        # Very small segments are not useful for loudness
        # calculations.
        if len(segment) < int(sr * 0.4):
            continue

        try:

            segment_lufs = (
                meter.integrated_loudness(
                    segment
                )
            )

            if (
                segment_lufs is not None
                and np.isfinite(segment_lufs)
            ):

                window_lufs.append(
                    float(segment_lufs)
                )

        except Exception:

            continue

    if len(window_lufs) >= 2:

        short_term_dr = float(
            max(window_lufs)
            -
            min(window_lufs)
        )

    else:

        short_term_dr = None

    # =========================================================
    # PLR
    # =========================================================
    #
    # Peak-to-Loudness Ratio
    # =========================================================

    if (
        np.isfinite(true_peak_db)
        and np.isfinite(loudness)
    ):

        plr = float(
            true_peak_db - loudness
        )

    else:

        plr = 0.0

    # =========================================================
    # CONTENT CLASSIFICATION
    # =========================================================

    content_type = classify_content(
        loudness,
        lra,
        freq_balance
    )

    # =========================================================
    # DISTRIBUTION SIMULATION
    # =========================================================

    distribution_simulation = simulate_distribution(
        loudness,
        true_peak_db
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    return {

        "integrated_lufs": round(
            loudness,
            2
        ),

        "true_peak_db": round(
            true_peak_db,
            2
        ),

        "rms_db": round(
            rms_db,
            2
        ),

        "loudness_range": round(
            lra,
            2
        ),

        "plr": round(
            plr,
            2
        ),

        "frequency_balance": freq_balance,

        "clipping": clipping,

        "stereo_balance": (
            None
            if stereo_balance is None
            else round(
                stereo_balance,
                2
            )
        ),

        "short_term_dynamic_range": (
            None
            if short_term_dr is None
            else round(
                short_term_dr,
                2
            )
        ),

        "content_type": content_type,

        "channel_count": channel_count,

        "distribution_simulation":
            distribution_simulation
    }


# =============================================================
# AES-INSPIRED CONTENT CLASSIFICATION
# =============================================================

def classify_content(
    loudness,
    lra,
    freq_balance
):

    """
    AES-inspired heuristic classification.

    This is an analytical heuristic and should not be
    presented as an official AES certification.
    """

    if lra > 14:

        return "wide_dynamic_range"

    if (
        freq_balance["mid"] > 45
        and freq_balance["low"] < 25
    ):

        return "speech_dominant"

    if (
        loudness > -16
        and lra < 8
    ):

        return "dense_music"

    return "mixed"


# =============================================================
# DISTRIBUTION SIMULATION
# =============================================================

def simulate_distribution(
    lufs,
    true_peak
):

    """
    Estimate gain changes relative to reference loudness
    targets.

    These are simulations, not guarantees of actual platform
    processing.
    """

    platforms = {}

    # =========================================================
    # MUSIC STREAMING
    # Reference: -14 LUFS
    # =========================================================

    delta = -14 - lufs

    predicted_peak = (
        true_peak + delta
    )

    will_limit = bool(
        delta > 0
        and predicted_peak > -1.0
    )

    platforms["music_streaming"] = {

        "gain_change_db": round(
            delta,
            2
        ),

        "will_limit": will_limit
    }

    # =========================================================
    # RADIO / SPEECH
    # Reference: -18 LUFS
    # =========================================================

    delta_radio = -18 - lufs

    predicted_radio_peak = (
        true_peak + delta_radio
    )

    will_limit_radio = bool(
        delta_radio > 0
        and predicted_radio_peak > -1.0
    )

    platforms["speech_streaming"] = {

        "gain_change_db": round(
            delta_radio,
            2
        ),

        "will_limit": will_limit_radio
    }

    return platforms
