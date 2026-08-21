def generate_consultation(result):
    """
    Generates automated LoudCheck consultation actions.

    AES/ITU standards inform the measurement methodology.
    Consultation recommendations are LoudCheck engineering
    guidance and should not be interpreted as official AES
    certification or mandatory mastering requirements.
    """

    actions = []

    # =========================================================
    # 1. LOUDNESS & STREAMING NORMALIZATION
    # =========================================================

    sim = (
        result
        .get("distribution_simulation", {})
        .get("music_streaming", {})
    )

    if sim.get("will_limit", False):

        actions.append({
            "area": "Loudness",
            "recommendation": (
                "Review integrated loudness and peak structure "
                "before distribution"
            ),
            "reason": (
                "The current LoudCheck distribution simulation "
                "predicts playback gain adjustment with possible "
                "limiter engagement."
            )
        })

    # =========================================================
    # 2. TRUE PEAK
    # =========================================================

    tp = result.get("true_peak_db")

    if tp is not None and tp > -1.0:

        actions.append({
            "area": "True Peak",
            "recommendation": (
                "Set the true-peak limiter ceiling to approximately "
                "−1.2 dBTP"
            ),
            "reason": (
                "Additional true-peak headroom can reduce the risk "
                "of inter-sample clipping during playback and "
                "encoding."
            )
        })

    # =========================================================
    # 3. PLR / DYNAMICS
    # =========================================================

    plr = result.get("plr")
    content_type = result.get("content_type", "")

    if plr is not None and plr < 8:

        if content_type == "speech_dominant":

            actions.append({
                "area": "Dynamics (PLR)",
                "recommendation": (
                    "Inspect limiting and compression"
                ),
                "reason": (
                    f"PLR = {plr:.2f} dB indicates limited "
                    "peak-to-loudness separation. Review compression, "
                    "limiting, and transient preservation."
                )
            })

        else:

            actions.append({
                "area": "Dynamics (PLR)",
                "recommendation": (
                    "Inspect limiting and compression"
                ),
                "reason": (
                    f"PLR = {plr:.2f} dB indicates limited "
                    "peak-to-loudness separation. Review peak "
                    "structure and transient preservation."
                )
            })

    # =========================================================
    # 4. CLIPPING
    # =========================================================

    if result.get("clipping", False):

        actions.append({
            "area": "Clipping",
            "recommendation": (
                "Review gain staging and peak control"
            ),
            "reason": (
                "Digital clipping can introduce non-linear "
                "distortion."
            )
        })

    # =========================================================
    # 5. LOUDNESS RANGE
    # =========================================================

    lra = result.get("loudness_range")

    if lra is not None:

        if lra < 7:

            actions.append({
                "area": "Dynamic Range",
                "recommendation": (
                    "Inspect compression, limiting, and "
                    "program-level dynamic variation"
                ),
                "reason": (
                    f"Loudness Range = {lra:.2f} dB. "
                    "A narrow measured loudness range may indicate "
                    "substantial dynamic processing or consistently "
                    "levelled program material."
                )
            })

        elif lra > 14:

            actions.append({
                "area": "Dynamic Range",
                "recommendation": (
                    "Review extreme level variation and peak control"
                ),
                "reason": (
                    f"Loudness Range = {lra:.2f} dB. "
                    "Large program-level variation may result in "
                    "greater playback level changes or normalization."
                )
            })

    # =========================================================
    # 6. FREQUENCY BALANCE
    # =========================================================

    freq = result.get(
        "frequency_balance",
        {}
    )

    low = freq.get("low", 0)
    mid = freq.get("mid", 0)
    high = freq.get("high", 0)

    if (
        low > 70
        or high > 70
        or mid < 20
    ):

        actions.append({
            "area": "Frequency Balance",
            "recommendation": (
                "Review spectral balance and tonal distribution"
            ),
            "reason": (
                "Measured spectral energy is concentrated unevenly "
                "across the analysed frequency bands. Review EQ and "
                "tonal balance where appropriate."
            )
        })

    # =========================================================
    # 7. STEREO BALANCE
    # =========================================================

    stereo_available = result.get(
        "stereo_available",
        False
    )

    stereo = result.get(
        "stereo_balance"
    )

    if stereo_available and stereo is not None:

        if abs(stereo) > 0.5:

            actions.append({
                "area": "Stereo Image",
                "recommendation": (
                    "Review stereo balance and channel levels"
                ),
                "reason": (
                    f"Stereo balance = {stereo:.2f}. "
                    "Significant channel imbalance may affect "
                    "stereo and mono playback."
                )
            })

    # =========================================================
    # 8. DISTRIBUTION-SPECIFIC NOTES
    # =========================================================

    distribution = result.get(
        "distribution_simulation",
        {}
    )

    for platform, info in distribution.items():

        if not isinstance(info, dict):
            continue

        if info.get("will_limit", False):

            actions.append({
                "area": f"Distribution – {platform}",
                "recommendation": (
                    "Inspect peak structure and limiting behavior"
                ),
                "reason": (
                    f"{platform} is predicted to apply gain "
                    "reduction or playback limiting."
                )
            })

    # =========================================================
    # 9. DISCLAIMER
    # =========================================================

    actions.append({
        "area": "Disclaimer",
        "recommendation": (
            "Automated guidance only"
        ),
        "reason": (
            "LoudCheck uses AES/ITU-informed measurement and "
            "engineering interpretation. Results do not constitute "
            "official AES certification or replace professional "
            "mastering review."
        )
    })

    return actions
