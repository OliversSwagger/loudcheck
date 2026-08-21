# audio_analysis.py

import json
import shutil
import subprocess

import librosa
import numpy as np

from scipy import signal


# =============================================================
# BS.1770 CONSTANTS
# =============================================================

LOUDNESS_OFFSET = -0.691

ABSOLUTE_GATE_LKFS = -70.0
RELATIVE_GATE_OFFSET_DB = -10.0

LOUDNESS_BLOCK_SECONDS = 0.400
LOUDNESS_HOP_SECONDS = 0.100

SHORT_TERM_WINDOW_SECONDS = 3.0

MIN_TRUE_PEAK_OVERSAMPLE = 4
MIN_TRUE_PEAK_SAMPLE_RATE = 192000


# =============================================================
# CHANNEL WEIGHTS
# =============================================================

CONVENTIONAL_CHANNEL_WEIGHTS = {

    1: np.array(
        [1.0],
        dtype=np.float64
    ),

    2: np.array(
        [1.0, 1.0],
        dtype=np.float64
    ),

    3: np.array(
        [1.0, 1.0, 1.0],
        dtype=np.float64
    ),

    4: np.array(
        [1.0, 1.0, 1.41, 1.41],
        dtype=np.float64
    ),

    5: np.array(
        [1.0, 1.0, 1.0, 1.41, 1.41],
        dtype=np.float64
    ),

    6: np.array(
        [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
        dtype=np.float64
    ),

    7: np.array(
        [1.0, 1.0, 1.0, 1.41, 1.41, 1.41, 1.41],
        dtype=np.float64
    ),

    8: np.array(
        [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41],
        dtype=np.float64
    )
}


# =============================================================
# ITU COMPLIANCE CHANNEL CONFIGURATIONS
# =============================================================

ITU_CHANNEL_WEIGHTS = {

    "itu_10": np.array(
        [
            1.00,
            1.00,
            1.00,
            0.00,
            1.41,
            1.41,
            1.00,
            1.00,
            1.00,
            1.00
        ],
        dtype=np.float64
    ),

    "itu_12": np.array(
        [
            1.00,
            1.00,
            1.00,
            0.00,
            1.41,
            1.41,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00
        ],
        dtype=np.float64
    ),

    "itu_24": np.array(
        [
            1.41,
            1.41,
            1.00,
            0.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            0.00,
            1.41,
            1.41,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00
        ],
        dtype=np.float64
    )
}


# =============================================================
# ADVANCED LAYOUTS
# =============================================================

ADVANCED_CHANNEL_WEIGHTS = {

    "7.1.4": np.array(
        [
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.41,
            1.41,
            1.0,
            1.0,
            1.0,
            1.0
        ],
        dtype=np.float64
    )
}


# =============================================================
# NORMALISE CHANNEL LAYOUT
# =============================================================

def normalize_channel_layout(
    channel_layout: str | None
) -> str | None:

    if channel_layout is None:
        return None

    value = str(
        channel_layout
    ).strip().lower()

    aliases = {

        "mono": "mono",
        "1.0": "mono",

        "stereo": "stereo",
        "2.0": "stereo",

        "5.1": "5.1",

        "7.1": "7.1",

        "7.1.4": "7.1.4",
        "7_1_4": "7.1.4",
        "7-1-4": "7.1.4",

        "itu_10": "itu_10",
        "itu-10": "itu_10",

        "itu_12": "itu_12",
        "itu-12": "itu_12",

        "itu_24": "itu_24",
        "itu-24": "itu_24",

        "24ch": "itu_24",
        "24_channel": "itu_24",
        "24-channel": "itu_24"
    }

    return aliases.get(
        value,
        value
    )


# =============================================================
# FFPROBE CHANNEL LAYOUT DISCOVERY
# =============================================================

def detect_channel_layout(
    file_path: str
) -> str | None:

    ffprobe = shutil.which(
        "ffprobe"
    )

    if ffprobe is None:

        print(
            "DEBUG: ffprobe not available; "
            "channel layout metadata cannot be detected.",
            flush=True
        )

        return None

    command = [

        ffprobe,

        "-v",
        "error",

        "-select_streams",
        "a:0",

        "-show_entries",
        "stream=channel_layout",

        "-of",
        "json",

        file_path
    ]

    try:

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )

        data = json.loads(
            completed.stdout
        )

        streams = data.get(
            "streams",
            []
        )

        if not streams:
            return None

        layout = streams[0].get(
            "channel_layout"
        )

        if not layout:
            return None

        return normalize_channel_layout(
            layout
        )

    except Exception as exc:

        print(
            "WARNING: Unable to detect channel layout:",
            exc,
            flush=True
        )

        return None


# =============================================================
# CHANNEL WEIGHTS
# =============================================================

def get_channel_weights(
    channel_count: int,
    channel_layout: str | None = None
) -> np.ndarray:

    layout = normalize_channel_layout(
        channel_layout
    )

    if layout in ADVANCED_CHANNEL_WEIGHTS:

        weights = ADVANCED_CHANNEL_WEIGHTS[
            layout
        ]

        if len(weights) != channel_count:

            raise ValueError(
                f"Channel layout '{layout}' requires "
                f"{len(weights)} channels, but the file "
                f"contains {channel_count}."
            )

        return weights.copy()

    if layout in ITU_CHANNEL_WEIGHTS:

        weights = ITU_CHANNEL_WEIGHTS[
            layout
        ]

        if len(weights) != channel_count:

            raise ValueError(
                f"Channel layout '{layout}' requires "
                f"{len(weights)} channels, but the file "
                f"contains {channel_count}."
            )

        return weights.copy()

    if layout == "mono":

        if channel_count != 1:

            raise ValueError(
                "Mono layout requires exactly 1 channel."
            )

        return np.array(
            [1.0],
            dtype=np.float64
        )

    if layout == "stereo":

        if channel_count != 2:

            raise ValueError(
                "Stereo layout requires exactly 2 channels."
            )

        return np.array(
            [1.0, 1.0],
            dtype=np.float64
        )

    if layout == "5.1":

        if channel_count != 6:

            raise ValueError(
                "5.1 layout requires exactly 6 channels."
            )

        return CONVENTIONAL_CHANNEL_WEIGHTS[
            6
        ].copy()

    if layout == "7.1":

        if channel_count != 8:

            raise ValueError(
                "7.1 layout requires exactly 8 channels."
            )

        return CONVENTIONAL_CHANNEL_WEIGHTS[
            8
        ].copy()

    if layout is None:

        if channel_count in CONVENTIONAL_CHANNEL_WEIGHTS:

            return CONVENTIONAL_CHANNEL_WEIGHTS[
                channel_count
            ].copy()

        if channel_count == 10:

            print(
                "DEBUG: Using ITU 10-channel compliance layout.",
                flush=True
            )

            return ITU_CHANNEL_WEIGHTS[
                "itu_10"
            ].copy()

        if channel_count == 12:

            print(
                "DEBUG: Using ITU 12-channel compliance layout.",
                flush=True
            )

            return ITU_CHANNEL_WEIGHTS[
                "itu_12"
            ].copy()

        if channel_count == 24:

            print(
                "DEBUG: Using ITU 24-channel compliance layout.",
                flush=True
            )

            return ITU_CHANNEL_WEIGHTS[
                "itu_24"
            ].copy()

        if channel_count > 8:

            return np.ones(
                channel_count,
                dtype=np.float64
            )

        raise ValueError(
            f"Unsupported channel count: {channel_count}."
        )

    raise ValueError(
        f"Unsupported channel layout '{channel_layout}' "
        f"for {channel_count} channels."
    )


# =============================================================
# K-WEIGHTING
# =============================================================

def _k_weighting_coefficients(
    sample_rate: int
):

    fs = float(
        sample_rate
    )

    if fs <= 0:

        raise ValueError(
            "Sample rate must be greater than zero."
        )

    f0 = 1681.974450955533

    q = 0.7071752369554196

    gain_db = 3.999843853973347

    w0 = 2.0 * np.pi * f0

    gain = (
        10.0
        **
        (
            gain_db
            /
            20.0
        )
    )

    sqrt_gain = np.sqrt(
        gain
    )

    b1_analog = np.array(
        [
            gain,
            gain
            *
            w0
            /
            (
                q
                *
                sqrt_gain
            ),
            gain
            *
            w0
            **
            2
        ],
        dtype=np.float64
    )

    a1_analog = np.array(
        [
            1.0,
            w0
            *
            sqrt_gain
            /
            q,
            gain
            *
            w0
            **
            2
        ],
        dtype=np.float64
    )

    b1, a1 = signal.bilinear(
        b1_analog,
        a1_analog,
        fs=fs
    )

    f0_rlb = 38.13547087613982

    q_rlb = 0.5003270373253953

    w0_rlb = 2.0 * np.pi * f0_rlb

    b2_analog = np.array(
        [
            1.0,
            0.0,
            0.0
        ],
        dtype=np.float64
    )

    a2_analog = np.array(
        [
            1.0,
            w0_rlb
            /
            q_rlb,
            w0_rlb
            **
            2
        ],
        dtype=np.float64
    )

    b2, a2 = signal.bilinear(
        b2_analog,
        a2_analog,
        fs=fs
    )

    return (
        b1,
        a1,
        b2,
        a2
    )


def _k_weight_channel(
    channel: np.ndarray,
    sample_rate: int
) -> np.ndarray:

    channel = np.asarray(
        channel,
        dtype=np.float64
    )

    if channel.size == 0:

        return channel.copy()

    b1, a1, b2, a2 = (
        _k_weighting_coefficients(
            sample_rate
        )
    )

    sos1 = signal.tf2sos(
        b1,
        a1
    )

    sos2 = signal.tf2sos(
        b2,
        a2
    )

    filtered = signal.sosfilt(
        sos1,
        channel
    )

    filtered = signal.sosfilt(
        sos2,
        filtered
    )

    return np.asarray(
        filtered,
        dtype=np.float64
    )


def k_weight_channels(
    channels: np.ndarray,
    sample_rate: int
) -> np.ndarray:

    channels = np.asarray(
        channels,
        dtype=np.float64
    )

    if channels.ndim != 2:

        raise ValueError(
            "Expected audio with shape "
            "(channels, samples)."
        )

    filtered = np.empty_like(
        channels,
        dtype=np.float64
    )

    for channel_index in range(
        channels.shape[0]
    ):

        filtered[channel_index] = (
            _k_weight_channel(
                channels[channel_index],
                sample_rate
            )
        )

    return filtered


# =============================================================
# LOUDNESS BLOCKS
# =============================================================

def _generate_loudness_blocks(
    sample_count: int,
    sample_rate: int
):

    block_length = max(
        1,
        int(
            round(
                sample_rate
                *
                LOUDNESS_BLOCK_SECONDS
            )
        )
    )

    hop_length = max(
        1,
        int(
            round(
                sample_rate
                *
                LOUDNESS_HOP_SECONDS
            )
        )
    )

    blocks = []

    start = 0

    while (
        start
        +
        block_length
        <=
        sample_count
    ):

        end = (
            start
            +
            block_length
        )

        blocks.append(
            (
                start,
                end
            )
        )

        start += hop_length

    return blocks


# =============================================================
# BLOCK ENERGY
# =============================================================

def _calculate_weighted_energy(
    block: np.ndarray,
    weights: np.ndarray
) -> float:

    mean_square = np.mean(
        block ** 2,
        axis=1
    )

    weighted_energy = float(
        np.sum(
            weights
            *
            mean_square
        )
    )

    return weighted_energy


def _energy_to_loudness(
    energy: float
) -> float:

    if energy <= 0.0:

        return -np.inf

    return float(
        LOUDNESS_OFFSET
        +
        10.0
        *
        np.log10(
            energy
        )
    )


# =============================================================
# LOUDNESS BLOCK CALCULATION
# =============================================================

def calculate_loudness_blocks(
    channels: np.ndarray,
    sample_rate: int,
    channel_layout: str | None = None
):

    channels = np.asarray(
        channels,
        dtype=np.float64
    )

    if channels.ndim != 2:

        raise ValueError(
            "Expected channels x samples."
        )

    weights = get_channel_weights(
        channels.shape[0],
        channel_layout
    )

    k_weighted = k_weight_channels(
        channels,
        sample_rate
    )

    blocks = _generate_loudness_blocks(
        channels.shape[1],
        sample_rate
    )

    loudness_values = []

    for start, end in blocks:

        block = k_weighted[
            :,
            start:end
        ]

        energy = _calculate_weighted_energy(
            block,
            weights
        )

        loudness_values.append(
            _energy_to_loudness(
                energy
            )
        )

    return loudness_values


# =============================================================
# INTEGRATED LOUDNESS
# =============================================================

def calculate_integrated_loudness(
    channels: np.ndarray,
    sample_rate: int,
    channel_layout: str | None = None
) -> float:

    channels = np.asarray(
        channels,
        dtype=np.float64
    )

    if channels.ndim != 2:

        raise ValueError(
            "Expected channels x samples."
        )

    weights = get_channel_weights(
        channels.shape[0],
        channel_layout
    )

    k_weighted = k_weight_channels(
        channels,
        sample_rate
    )

    blocks = _generate_loudness_blocks(
        channels.shape[1],
        sample_rate
    )

    if not blocks:

        raise ValueError(
            "Audio is too short for a 400 ms "
            "BS.1770 loudness block."
        )

    block_energies = []

    block_loudness = []

    for start, end in blocks:

        block = k_weighted[
            :,
            start:end
        ]

        energy = _calculate_weighted_energy(
            block,
            weights
        )

        block_energies.append(
            energy
        )

        block_loudness.append(
            _energy_to_loudness(
                energy
            )
        )

    block_energies = np.asarray(
        block_energies,
        dtype=np.float64
    )

    block_loudness = np.asarray(
        block_loudness,
        dtype=np.float64
    )

    absolute_mask = (
        block_loudness
        >=
        ABSOLUTE_GATE_LKFS
    )

    if not np.any(
        absolute_mask
    ):

        return -np.inf

    absolute_energies = (
        block_energies[
            absolute_mask
        ]
    )

    absolute_loudness = _energy_to_loudness(
        float(
            np.mean(
                absolute_energies
            )
        )
    )

    relative_threshold = (
        absolute_loudness
        +
        RELATIVE_GATE_OFFSET_DB
    )

    relative_mask = (
        absolute_mask
        &
        (
            block_loudness
            >=
            relative_threshold
        )
    )

    if not np.any(
        relative_mask
    ):

        return float(
            absolute_loudness
        )

    gated_energies = (
        block_energies[
            relative_mask
        ]
    )

    return _energy_to_loudness(
        float(
            np.mean(
                gated_energies
            )
        )
    )


# =============================================================
# MOMENTARY LOUDNESS
# =============================================================

def calculate_momentary_loudness(
    channels: np.ndarray,
    sample_rate: int,
    channel_layout: str | None = None
):

    channels = np.asarray(
        channels,
        dtype=np.float64
    )

    if channels.ndim != 2:

        raise ValueError(
            "Expected channels x samples."
        )

    weights = get_channel_weights(
        channels.shape[0],
        channel_layout
    )

    k_weighted = k_weight_channels(
        channels,
        sample_rate
    )

    block_length = max(
        1,
        int(
            round(
                sample_rate
                *
                LOUDNESS_BLOCK_SECONDS
            )
        )
    )

    hop_length = max(
        1,
        int(
            round(
                sample_rate
                *
                LOUDNESS_HOP_SECONDS
            )
        )
    )

    results = []

    start = 0

    while (
        start
        +
        block_length
        <=
        channels.shape[1]
    ):

        end = (
            start
            +
            block_length
        )

        block = k_weighted[
            :,
            start:end
        ]

        energy = _calculate_weighted_energy(
            block,
            weights
        )

        loudness = _energy_to_loudness(
            energy
        )

        results.append(
            {
                "time_seconds": round(
                    start
                    /
                    sample_rate,
                    3
                ),
                "lufs": float(
                    loudness
                )
            }
        )

        start += hop_length

    return results


# =============================================================
# SHORT-TERM LOUDNESS
# =============================================================

def calculate_short_term_loudness(
    channels: np.ndarray,
    sample_rate: int,
    channel_layout: str | None = None
):

    channels = np.asarray(
        channels,
        dtype=np.float64
    )

    if channels.ndim != 2:

        raise ValueError(
            "Expected channels x samples."
        )

    weights = get_channel_weights(
        channels.shape[0],
        channel_layout
    )

    k_weighted = k_weight_channels(
        channels,
        sample_rate
    )

    window_length = max(
        1,
        int(
            round(
                sample_rate
                *
                SHORT_TERM_WINDOW_SECONDS
            )
        )
    )

    hop_length = max(
        1,
        int(
            round(
                sample_rate
                *
                LOUDNESS_HOP_SECONDS
            )
        )
    )

    results = []

    start = 0

    while (
        start
        +
        window_length
        <=
        channels.shape[1]
    ):

        end = (
            start
            +
            window_length
        )

        block = k_weighted[
            :,
            start:end
        ]

        energy = _calculate_weighted_energy(
            block,
            weights
        )

        loudness = _energy_to_loudness(
            energy
        )

        results.append(
            {
                "time_seconds": round(
                    start
                    /
                    sample_rate,
                    3
                ),
                "lufs": float(
                    loudness
                )
            }
        )

        start += hop_length

    return results


# =============================================================
# LOUDNESS RANGE
# =============================================================

def calculate_loudness_range(
    channels: np.ndarray,
    sample_rate: int,
    channel_layout: str | None = None
) -> float:

    values = calculate_short_term_loudness(
        channels,
        sample_rate,
        channel_layout
    )

    values = np.asarray(
        [
            item["lufs"]
            for item in values
        ],
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    values = values[
        values >= ABSOLUTE_GATE_LKFS
    ]

    if values.size < 2:

        return 0.0

    energies = (
        10.0
        **
        (
            (
                values
                -
                LOUDNESS_OFFSET
            )
            /
            10.0
        )
    )

    absolute_loudness = _energy_to_loudness(
        float(
            np.mean(
                energies
            )
        )
    )

    relative_threshold = (
        absolute_loudness
        -
        10.0
    )

    values = values[
        values >= relative_threshold
    ]

    if values.size < 2:

        return 0.0

    low = float(
        np.percentile(
            values,
            10.0
        )
    )

    high = float(
        np.percentile(
            values,
            95.0
        )
    )

    return max(
        0.0,
        high - low
    )


# =============================================================
# TRUE PEAK
# =============================================================

def _true_peak_oversample_factor(
    sample_rate: int
) -> int:

    return max(
        MIN_TRUE_PEAK_OVERSAMPLE,
        int(
            np.ceil(
                MIN_TRUE_PEAK_SAMPLE_RATE
                /
                sample_rate
            )
        )
    )


def calculate_true_peak(
    channels: np.ndarray,
    sample_rate: int
) -> float:

    channels = np.asarray(
        channels,
        dtype=np.float64
    )

    if channels.ndim != 2:

        raise ValueError(
            "Expected channels x samples."
        )

    factor = _true_peak_oversample_factor(
        sample_rate
    )

    highest_peak = 0.0

    for channel_index in range(
        channels.shape[0]
    ):

        channel = channels[
            channel_index
        ]

        if channel.size == 0:
            continue

        upsampled = signal.resample_poly(
            channel,
            factor,
            1
        )

        upsampled = np.asarray(
            upsampled,
            dtype=np.float64
        )

        upsampled = np.nan_to_num(
            upsampled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        peak = float(
            np.max(
                np.abs(
                    upsampled
                )
            )
        )

        highest_peak = max(
            highest_peak,
            peak
        )

    if highest_peak <= 1e-12:

        return -120.0

    return float(
        20.0
        *
        np.log10(
            highest_peak
        )
    )


# =============================================================
# CHANNEL LAYOUT RESOLUTION
# =============================================================

def resolve_channel_layout(
    file_path: str,
    channel_count: int,
    requested_layout: str | None = None
) -> tuple[str, bool]:

    if requested_layout is not None:

        normalized = normalize_channel_layout(
            requested_layout
        )

        return normalized, True

    detected_layout = (
        detect_channel_layout(
            file_path
        )
    )

    if detected_layout:

        return detected_layout, True

    if channel_count == 1:

        return "mono", True

    if channel_count == 2:

        return "stereo", True

    if channel_count == 6:

        return "5.1", True

    if channel_count == 8:

        return "7.1", True

    if channel_count == 10:

        return "itu_10", True

    if channel_count == 12:

        return "itu_12", True

    if channel_count == 24:

        return "itu_24", True

    generic_layout = (
        f"generic_{channel_count}ch"
    )

    return generic_layout, False


# =============================================================
# AUDIO ANALYSIS
# =============================================================

def analyze_audio(
    file_path: str,
    channel_layout: str | None = None
):

    # =========================================================
    # LOAD
    # =========================================================

    try:

        y, sr = librosa.load(
            file_path,
            sr=None,
            mono=False
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
    # CHANNEL SHAPE
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

        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # =========================================================
    # RESOLVE CHANNEL LAYOUT
    # =========================================================

    resolved_layout, layout_is_known = (
        resolve_channel_layout(
            file_path,
            channel_count,
            channel_layout
        )
    )

    channel_layout = resolved_layout

    # =========================================================
    # VALIDATE CHANNEL WEIGHTS
    # =========================================================

    try:

        weights = get_channel_weights(
            channel_count,
            channel_layout
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to determine a valid channel layout "
            f"for loudness measurement: {exc}"
        )

    # =========================================================
    # LOUDCHECK MONO REPRESENTATION
    # =========================================================

    if channel_count == 1:

        stereo_available = False

        stereo_balance = 0.0

        y_mono = y[0]

    elif channel_count == 2:

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

        y_mono = np.mean(
            y,
            axis=0
        )

    else:

        stereo_available = False

        stereo_balance = 0.0

        y_mono = np.mean(
            y,
            axis=0
        )

    # =========================================================
    # ORIGINAL AUDIO PEAK
    # =========================================================

    peak_linear_all_channels = float(
        np.max(
            np.abs(
                y
            )
        )
    )

    if peak_linear_all_channels <= 1e-12:

        raise ValueError(
            "Audio contains no measurable signal. "
            "The file may be silent or empty."
        )

    # =========================================================
    # INTEGRATED LOUDNESS
    # =========================================================

    try:

        loudness = (
            calculate_integrated_loudness(
                y,
                sr,
                channel_layout
            )
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate integrated loudness: {exc}"
        )

    if (
        loudness is None
        or
        not np.isfinite(
            loudness
        )
    ):

        raise ValueError(
            "Unable to calculate a valid integrated "
            "loudness value."
        )

    loudness = float(
        loudness
    )

    # =========================================================
    # LRA
    # =========================================================

    try:

        lra = calculate_loudness_range(
            y,
            sr,
            channel_layout
        )

        if (
            lra is None
            or
            not np.isfinite(
                lra
            )
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

    # =========================================================
    # TRUE PEAK
    # =========================================================

    try:

        true_peak_db = calculate_true_peak(
            y,
            sr
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to calculate true peak: {exc}"
        )

    # =========================================================
    # RMS
    # =========================================================

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
            20.0
            *
            np.log10(
                rms
            )
        )

    # =========================================================
    # FREQUENCY ANALYSIS
    # =========================================================

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
            if np.any(
                low_mask
            )
            else 0.0
        )

        mid = (
            float(
                np.mean(
                    stft[mid_mask]
                )
            )
            if np.any(
                mid_mask
            )
            else 0.0
        )

        high = (
            float(
                np.mean(
                    stft[high_mask]
                )
            )
            if np.any(
                high_mask
            )
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
                    low
                    /
                    total
                    *
                    100.0,
                    1
                ),

                "mid": round(
                    mid
                    /
                    total
                    *
                    100.0,
                    1
                ),

                "high": round(
                    high
                    /
                    total
                    *
                    100.0,
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

    # =========================================================
    # CLIPPING
    # =========================================================

    clipping = bool(
        np.max(
            np.abs(
                y
            )
        )
        >=
        1.0
    )

    # =========================================================
    # SHORT-TERM DYNAMIC RANGE
    # =========================================================

    short_term_values = (
        calculate_short_term_loudness(
            y,
            sr,
            channel_layout
        )
    )

    window_lufs = [

        float(
            item["lufs"]
        )

        for item in short_term_values

        if np.isfinite(
            item["lufs"]
        )
    ]

    if len(
        window_lufs
    ) >= 2:

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

    # =========================================================
    # PLR
    # =========================================================

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

    distribution_simulation = (
        simulate_distribution(
            loudness,
            true_peak_db
        )
    )

    # =========================================================
    # RESULT
    # =========================================================

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

        "channel_layout":
            channel_layout,

        "channel_layout_known":
            layout_is_known,

        "distribution_simulation":
            distribution_simulation
    }

    print(
        "DEBUG: Final analysis result:",
        result,
        flush=True
    )

    return result


# =============================================================
# CONTENT CLASSIFICATION
# =============================================================

def classify_content(
    loudness,
    lra,
    freq_balance
):

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

    platforms = {}

    # ---------------------------------------------------------
    # MUSIC STREAMING
    # ---------------------------------------------------------

    delta = (
        -14.0
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

    # ---------------------------------------------------------
    # RADIO / SPEECH
    # ---------------------------------------------------------

    delta_radio = (
        -18.0
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
