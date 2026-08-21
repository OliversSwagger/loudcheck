def generate_mobile_warnings(analysis):
    """
    Generates playback warnings for mobile devices and streaming.

    Measurement methodology is informed by AES/ITU loudness practices.
    LoudCheck-specific warning thresholds are content-aware application
    rules.

    PLR warnings are only applied to dense music content because a low
    PLR is not, by itself, sufficient evidence of a playback problem for
    speech-dominant material.
    """

    warnings = []

    tp = analysis.get("true_peak_db")
    lufs = analysis.get("integrated_lufs")
    plr = analysis.get("plr")
    content_type = analysis.get("content_type", "").lower().strip()

    distribution = analysis.get("distribution_simulation", {})
    sim = distribution.get("music_streaming", {})

    # ---------------------------------------------------------
    # TRUE PEAK
    # ---------------------------------------------------------

    if tp is not None and tp > -1.0:
        warnings.append({
            "severity": "high",
            "message": (
                f"True Peak = {tp:.2f} dBTP exceeds −1 dBTP. "
                "Inter-sample peaks may increase clipping risk on "
                "consumer playback and streaming systems."
            )
        })

    # ---------------------------------------------------------
    # STREAMING NORMALIZATION / LIMITING
    # ---------------------------------------------------------

    if sim.get("will_limit", False):
        warnings.append({
            "severity": "high",
            "message": (
                "Distribution normalization is expected to engage "
                "playback limiting, potentially altering transient content."
            )
        })

    # ---------------------------------------------------------
    # PEAK-TO-LOUDNESS RATIO
    # ---------------------------------------------------------
    #
    # PLR is interpreted differently depending on content.
    #
    # Dense music:
    #   Low PLR can indicate aggressive peak limiting/compression.
    #
    # Speech:
    #   Do not automatically classify low PLR as a playback risk.
    #   Speech can legitimately have a relatively small peak-to-loudness
    #   relationship.
    #
    # Therefore this warning is intentionally content-aware.
    # ---------------------------------------------------------

    if (
        plr is not None
        and content_type == "dense_music"
        and plr < 8
    ):
        warnings.append({
            "severity": "medium",
            "message": (
                f"PLR = {plr:.2f} dB. "
                "Low peak-to-loudness ratio may indicate aggressive "
                "compression or limiting in dense music content."
            )
        })

    # ---------------------------------------------------------
    # INTEGRATED LOUDNESS
    # ---------------------------------------------------------

    if lufs is not None and lufs > -13:
        warnings.append({
            "severity": "low",
            "message": (
                f"Integrated Loudness = {lufs:.2f} LUFS exceeds "
                "typical streaming targets; content may be attenuated "
                "during playback normalization."
            )
        })

    return warnings
