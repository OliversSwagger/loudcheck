# audio_analysis.py

import time
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
    # TOTAL ANALYSIS TIMER
    # =========================================================

    analysis_start = time.perf_counter()

    def timer(label, start):
        elapsed = time.perf_counter() - start

        print(
            f"TIMER: {label}: {elapsed:.3f} seconds",
            flush=True
        )

        return time.perf_counter()

    # =========================================================
    # LOAD AUDIO
    # =========================================================

    timer_start = time.perf_counter()

    try:

        y, sr = librosa.load(
            file_path,
            sr=None,
            mono=False
        )

        timer(
            "LOAD AUDIO",
            timer_start
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to load audio: {exc}"
        )

    if y is None:

        raise ValueError(
            "Unable to load audio."
        )

    y = np.asarray(
        y,
        dtype=np.float64
    )

    if y.size == 0:

        raise ValueError(
            "Audio file contains no audio data."
        )

    # =========================================================
    # ENSURE CHANNELS x SAMPLES
    # =========================================================

    if y.ndim == 1:

        y = np.expand_dims(
            y,
            axis=0
        )

    if y.ndim != 2:

        raise ValueError(
            f"Unsupported audio shape: {y.shape}"
        )

    channel_count = int(
        y.shape[0]
    )

    sample_count = int(
        y.shape[1]
    )

    print(
        "Channel count:",
        channel_count,
        flush=True
    )

    print(
        "Sample count:",
        sample_count,
        flush=True
    )

    print(
        "Sample rate:",
        sr,
        flush=True
    )

    if sample_count < sr:

        raise ValueError(
            "Audio too short for analysis. "
            "At least one second of audio is required."
        )

    # =========================================================
    # CLEAN INVALID VALUES
    # =========================================================

    if not np.all(
        np.isfinite(y)
    ):

        print(
            "WARNING: Invalid audio samples detected. "
            "Replacing invalid values.",
            flush=True
        )

        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # =========================================================
    # CHANNEL HANDLING
    # =========================================================

    timer_start = time.perf_counter()

    if channel_count == 1:

        # -----------------------------------------------------
        # MONO
        # -----------------------------------------------------

        stereo_available = False

        # Always numeric because downstream code may
        # perform mathematical operations on this value.

        stereo_balance = 0.0

        y_mono = y[0]

    elif channel_count == 2:

        # -----------------------------------------------------
        # STEREO
        # -----------------------------------------------------

        stereo_available = True

        left = y[0]
        right = y[1]

        left_rms = float(
            np.sqrt(
                np.mean(
                    left ** 2
                )
            )
        )

        right_rms = float(
            np.sqrt(
                np.mean(
                    right ** 2
                )
            )
        )

        max_rms = max(
            left_rms,
            right_rms
        )

        if max_rms > 1e-12:

            stereo_balance = float(
                (
                    right_rms
                    -
                    left_rms
                )
                /
                max_rms
            )

        else:

            stereo_balance = 0.0

        # Combine stereo channels for
        # general loudness analysis.

        y_mono = np.mean(
            y,
            axis=0
        )

    else:

        # -----------------------------------------------------
        # MULTICHANNEL
        # -----------------------------------------------------
        #
        # Examples:
        # 5.1
        # 7.1
        # immersive / multichannel
        #
        # Stereo balance does not apply.
        #
        # We still return 0.0 so downstream numerical
        # calculations remain safe.
        # -----------------------------------------------------

        stereo_available = False

        stereo_balance = 0.0

        # Combine ALL channels.

        y_mono = np.mean(
            y,
            axis=0
        )

    timer(
        "CHANNEL HANDLING",
        timer_start
    )

    # =========================================================
    # VALIDATE ANALYSIS SIGNAL
    # =========================================================

    timer_start = time.perf_counter()

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

    print(
        "y_mono type:",
        type(y_mono),
        flush=True
    )

    print(
        "y_mono shape:",
        y_mono.shape,
        flush=True
    )

    print(
        "y_mono is None:",
        y_mono is None,
        flush=True
    )

    # =========================================================
    # CHECK FOR SILENCE
    # =========================================================

    print(
        "DEBUG: before peak",
        flush=True
    )

    peak_linear = float(
        np.max(
            np.abs(y_mono)
        )
    )

    print(
        "DEBUG: after peak:",
        peak_linear,
        flush=True
    )

    if peak_linear <= 1e-12:

        raise ValueError(
            "Audio contains no measurable signal. "
            "The file may be silent or empty."
        )

    timer(
        "VALIDATE + PEAK",
        timer_start
    )

    # =========================================================
    # LOUDNESS METER
    # =========================================================

    meter = pyln.Meter(
        sr
    )

    # =========================================================
    # INTEGRATED LUFS
    # =========================================================

    timer_start = time.perf_counter()

    try:

        print(
            "DEBUG: before LUFS",
            flush=True
        )

        loudness = meter.integrated_loudness(
            y_mono
        )

        print(
            "DEBUG: LUFS:",
            loudness,
            flush=True
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate integrated loudness: {exc}"
        )

    if (
        loudness is None
        or
        not np.isfinite(loudness)
    ):

        raise ValueError(
            "Unable to calculate a valid integrated loudness value."
        )

    loudness = float(
        loudness
    )

    timer(
        "INTEGRATED LUFS",
        timer_start
    )

    # =========================================================
    # LOUDNESS RANGE
    # =========================================================

    timer_start = time.perf_counter()

    try:

        print(
            "DEBUG: before LRA",
            flush=True
        )

        lra = meter.loudness_range(
            y_mono
        )

        print(
            "DEBUG: LRA:",
            lra,
            flush=True
        )

        if (
            lra is None
            or
            not np.isfinite(lra)
        ):

            lra = 0.0

        else:

            lra = float(
                lra
            )

    except Exception as exc:

        print(
            "WARNING: LRA calculation failed:",
            exc,
            flush=True
        )

        lra = 0.0

    timer(
        "LOUDNESS RANGE",
        timer_start
    )

    # =========================================================
    # TRUE PEAK
    # =========================================================
    #
    # Upsample toward >=192 kHz.
    # =========================================================

    timer_start = time.perf_counter()

    try:

        upsample_factor = max(
            4,
            int(
                np.ceil(
                    192000 / sr
                )
            )
        )

        target_length = int(
            len(y_mono)
            *
            upsample_factor
        )

        print(
            "DEBUG: True peak upsample factor:",
            upsample_factor,
            flush=True
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
            np.max(
                np.abs(
                    y_upsampled
                )
            )
        )

        if peak_linear_upsampled <= 1e-12:

            true_peak_db = -120.0

        else:

            true_peak_db = float(
                20
                *
                np.log10(
                    peak_linear_upsampled
                )
            )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate true peak: {exc}"
        )

    print(
        "DEBUG: True Peak:",
        true_peak_db,
        flush=True
    )

    timer(
        "TRUE PEAK",
        timer_start
    )

    # =========================================================
    # RMS
    # =========================================================

    timer_start = time.perf_counter()

    try:

        rms = float(
            np.sqrt(
                np.mean(
                    y_mono ** 2
                )
            )
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate RMS: {exc}"
        )

    if rms <= 1e-12:

        rms_db = -120.0

    else:

        rms_db = float(
            20
            *
            np.log10(
                rms
            )
        )

    timer(
        "RMS",
        timer_start
    )

    # =========================================================
    # FREQUENCY ANALYSIS
    # =========================================================

    timer_start = time.perf_counter()

    try:

        stft = np.abs(
            librosa.stft(
                y_mono
            )
        )

        freqs = librosa.fft_frequencies(
            sr=sr
        )

        low_mask = (
            freqs < 80
        )

        mid_mask = (
            (freqs >= 80)
            &
            (freqs < 2000)
        )

        high_mask = (
            freqs >= 2000
        )

        low = (
            float(
                np.mean(
                    stft[low_mask]
                )
            )
            if np.any(low_mask)
            else 0.0
        )

        mid = (
            float(
                np.mean(
                    stft[mid_mask]
                )
            )
            if np.any(mid_mask)
            else 0.0
        )

        high = (
            float(
                np.mean(
                    stft[high_mask]
                )
            )
            if np.any(high_mask)
            else 0.0
        )

        total = (
            low
            +
            mid
            +
            high
        )

        if total <= 1e-12:

            freq_balance = {
                "low": 0.0,
                "mid": 0.0,
                "high": 0.0
            }

        else:

            freq_balance = {

                "low": round(
                    (
                        low
                        /
                        total
                    )
                    *
                    100,
                    1
                ),

                "mid": round(
                    (
                        mid
                        /
                        total
                    )
                    *
                    100,
                    1
                ),

                "high": round(
                    (
                        high
                        /
                        total
                    )
                    *
                    100,
                    1
                )
            }

    except Exception as exc:

        print(
            "WARNING: Frequency analysis failed:",
            exc,
            flush=True
        )

        freq_balance = {
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0
        }

    timer(
        "FREQUENCY ANALYSIS",
        timer_start
    )

    # =========================================================
    # CLIPPING DETECTION
    # =========================================================

    timer_start = time.perf_counter()

    clipping = bool(
        np.max(
            np.abs(
                y_mono
            )
        )
        >= 1.0
    )

    timer(
        "CLIPPING DETECTION",
        timer_start
    )

    # =========================================================
    # SHORT-TERM DYNAMIC RANGE
    # =========================================================

    timer_start = time.perf_counter()

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
            start:
            start + hop_len
        ]

        if len(segment) == 0:

            continue

        if len(segment) < int(
            sr * 0.4
        ):

            continue

        try:

            segment_lufs = (
                meter.integrated_loudness(
                    segment
                )
            )

            if (
                segment_lufs is not None
                and
                np.isfinite(
                    segment_lufs
                )
            ):

                window_lufs.append(
                    float(
                        segment_lufs
                    )
                )

        except Exception:

            continue

    if len(window_lufs) >= 2:

        short_term_dr = float(
            max(
                window_lufs
            )
            -
            min(
                window_lufs
            )
        )

    else:

        short_term_dr = 0.0

    timer(
        "SHORT-TERM DYNAMIC RANGE",
        timer_start
    )

    # =========================================================
    # PLR
    # =========================================================

    timer_start = time.perf_counter()

    if (
        np.isfinite(
            true_peak_db
        )
        and
        np.isfinite(
            loudness
        )
    ):

        plr = float(
            true_peak_db
            -
            loudness
        )

    else:

        plr = 0.0

    timer(
        "PLR",
        timer_start
    )

    # =========================================================
    # CONTENT CLASSIFICATION
    # =========================================================

    timer_start = time.perf_counter()

    content_type = classify_content(
        loudness,
        lra,
        freq_balance
    )

    timer(
        "CONTENT CLASSIFICATION",
        timer_start
    )

    # =========================================================
    # DISTRIBUTION SIMULATION
    # =========================================================

    timer_start = time.perf_counter()

    distribution_simulation = (
        simulate_distribution(
            loudness,
            true_peak_db
        )
    )

    timer(
        "DISTRIBUTION SIMULATION",
        timer_start
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    timer_start = time.perf_counter()

    result = {

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

        "frequency_balance":
            freq_balance,

        "clipping":
            clipping,

        "stereo_balance":
            round(
                stereo_balance,
                2
            ),

        "stereo_available":
            stereo_available,

        "short_term_dynamic_range":
            round(
                short_term_dr,
                2
            ),

        "content_type":
            content_type,

        "channel_count":
            channel_count,

        "distribution_simulation":
            distribution_simulation
    }

    timer(
        "RESULT BUILD",
        timer_start
    )

    print(
        "DEBUG: Final analysis result:",
        result,
        flush=True
    )

    # =========================================================
    # TOTAL ANALYSIS TIME
    # =========================================================

    total_time = (
        time.perf_counter()
        -
        analysis_start
    )

    print(
        f"TIMER: TOTAL ANALYSIS: {total_time:.3f} seconds",
        flush=True
    )

    return result


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
    presented as official AES certification.
    """

    if lra > 14:

        return "wide_dynamic_range"

    if (
        freq_balance["mid"] > 45
        and
        freq_balance["low"] < 25
    ):

        return "speech_dominant"

    if (
        loudness > -16
        and
        lra < 8
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

    delta = (
        -14
        -
        lufs
    )

    predicted_peak = (
        true_peak
        +
        delta
    )

    will_limit = bool(
        delta > 0
        and
        predicted_peak > -1.0
    )

    platforms["music_streaming"] = {

        "gain_change_db":
            round(
                delta,
                2
            ),

        "will_limit":
            will_limit
    }

    # =========================================================
    # RADIO / SPEECH
    # Reference: -18 LUFS
    # =========================================================

    delta_radio = (
        -18
        -
        lufs
    )

    predicted_radio_peak = (
        true_peak
        +
        delta_radio
    )

    will_limit_radio = bool(
        delta_radio > 0
        and
        predicted_radio_peak > -1.0
    )

    platforms["speech_streaming"] = {

        "gain_change_db":
            round(
                delta_radio,
                2
            ),

        "will_limit":
            will_limit_radio
    }

    return platforms
