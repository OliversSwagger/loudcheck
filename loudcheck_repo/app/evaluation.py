def evaluate_analysis(analysis: dict):
    results = {}

    # Integrated LUFS
    lufs = analysis["integrated_lufs"]
    if -16 <= lufs <= -13:
        results["lufs"] = ("green", f"Loudness is on target ({lufs:.1f} LUFS).")
    elif lufs < -16:
        results["lufs"] = ("yellow", f"Too quiet ({lufs:.1f} LUFS). Increase gain by ~{-14 - lufs:.1f} dB.")
    else:  # lufs > -13
        results["lufs"] = ("red", f"Too loud ({lufs:.1f} LUFS). Reduce limiter or overall gain by ~{lufs + 14:.1f} dB.")

    # True Peak
    tp = analysis["true_peak_db"]
    if tp <= -1.0:
        results["true_peak"] = ("green", f"True peak safe ({tp:.2f} dBTP).")
    else:
        results["true_peak"] = ("red", f"True peak too high ({tp:.2f} dBTP). Lower ceiling to −1.0 dBTP.")

    # Low-end balance
    low = analysis["frequency_balance"]["low"]
    if low < 35:
        results["low_end"] = ("green", f"Low-end balance good ({low:.1f}%).")
    elif 35 <= low <= 45:
        results["low_end"] = ("yellow", f"Low-end slightly dominant ({low:.1f}%). Consider subtle EQ reduction.")
    else:
        results["low_end"] = ("red", f"Low-end overpowering ({low:.1f}%). Reduce low frequencies or adjust EQ.")

    return results
