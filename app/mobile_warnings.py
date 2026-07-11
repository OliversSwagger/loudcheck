def generate_mobile_warnings(analysis):
    """
    Generates AES-compliant playback warnings for mobile devices and streaming.
    All language is technical and references measurable parameters.
    """
    warnings = []

    tp = analysis["true_peak_db"]
    lufs = analysis["integrated_lufs"]
    plr = analysis["plr"]
    sim = analysis["distribution_simulation"]["music_streaming"]

    # True Peak
    if tp > -1.0:
        warnings.append({
            "severity": "high",
            "message": (
                f"True Peak = {tp:.2f} dBTP exceeds −1 dBTP. "
                "Inter-sample peaks may be clipped on mobile DACs and streaming platforms."
            )
        })

    # Streaming normalization
    if sim["will_limit"]:
        warnings.append({
            "severity": "high",
            "message": (
                "Distribution normalization is expected to engage playback limiters, "
                "potentially attenuating transient content."
            )
        })

    # Peak-to-Loudness Ratio
    if plr < 8:
        warnings.append({
            "severity": "medium",
            "message": (
                f"PLR = {plr:.2f} dB. Transient content is likely to be attenuated under platform normalization."
            )
        })

    # Integrated Loudness
    if lufs > -13:
        warnings.append({
            "severity": "low",
            "message": (
                f"Integrated Loudness = {lufs:.2f} LUFS exceeds typical streaming targets; "
                "track may be attenuated during playback."
            )
        })

    return warnings
