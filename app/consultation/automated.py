def generate_consultation(result):
    """
    Generates automated AES-aligned consultation actions.
    All phrasing references measurable parameters and AES best practices.
    """
    actions = []

    # 1. Loudness & Streaming Normalization
    sim = result.get("distribution_simulation", {}).get("music_streaming", {})
    if sim.get("will_limit", False):
        actions.append({
            "area": "Loudness",
            "recommendation": "Reduce integrated loudness to approximately −14 to −14.5 LUFS",
            "reason": "Upward normalization may engage playback limiters, attenuating transient content."
        })

    # 2. True Peak
    tp = result.get("true_peak_db", 0)
    if tp > -1.0:
        actions.append({
            "area": "True Peak",
            "recommendation": "Set limiter ceiling to −1.2 dBTP",
            "reason": "Provides inter-sample peak safety across consumer DACs and mobile playback devices."
        })

    # 3. PLR / Dynamics
    plr = result.get("plr", 0)
    if plr < 8:
        actions.append({
            "area": "Dynamics (PLR)",
            "recommendation": "Ease limiting or reduce overall loudness",
            "reason": "Low PLR indicates transient signals may be attenuated under platform normalization."
        })

    # 4. Clipping
    if result.get("clipping", False):
        actions.append({
            "area": "Clipping",
            "recommendation": "Reduce gain staging or apply controlled limiting",
            "reason": "Digital clipping introduces non-linear distortion."
        })

    # 5. Loudness Range (LRA)
    lra = result.get("loudness_range", 0)
    if lra < 7:
        actions.append({
            "area": "Dynamic Range",
            "recommendation": "Increase dynamic range by reducing compression",
            "reason": "Overly narrow dynamics may collapse under streaming normalization."
        })
    elif lra > 14:
        actions.append({
            "area": "Dynamic Range",
            "recommendation": "Apply mild compression to control extreme peaks",
            "reason": "Excessive dynamics may be constrained by platform processing."
        })

    # 6. Frequency Balance
    freq = result.get("frequency_balance", {})
    low, mid, high = freq.get("low", 0), freq.get("mid", 0), freq.get("high", 0)
    if low > 70 or high > 70 or mid < 20:
        actions.append({
            "area": "Frequency Balance",
            "recommendation": "Adjust EQ to achieve more uniform spectral distribution",
            "reason": "Ensures tonal consistency across devices and platforms."
        })

    # 7. Stereo Balance
    #stereo = result.get("stereo_balance", 0)
    #if abs(stereo) > 0.5:
    #    actions.append({
    #        "area": "Stereo Image",
    #        "recommendation": "Center stereo image or adjust pan",
    #        "reason": "Excessive imbalance may collapse in mono or mobile playback."
    #    })
    # 7. Stereo Balance
    stereo = result.get("stereo_balance")
    
    if stereo is not None and abs(stereo) > 0.5:
        actions.append({
            "area": "Stereo Image",
            "recommendation": "Center stereo image or adjust pan",
            "reason": "Excessive imbalance may collapse in mono or mobile playback."
        })

    # 8. Distribution-specific notes
    for platform, info in result.get("distribution_simulation", {}).items():
        if info.get("will_limit"):
            actions.append({
                "area": f"Distribution – {platform}",
                "recommendation": "Inspect peak structure and limiting behavior",
                "reason": f"{platform} is likely to apply gain reduction or limiting."
            })

    # 9. Disclaimer
    actions.append({
        "area": "Disclaimer",
        "recommendation": "Automated guidance only",
        "reason": "AES-based interpretation; does not replace professional mastering review."
    })
    print("DEBUG ACTIONS:", actions)
    return actions
