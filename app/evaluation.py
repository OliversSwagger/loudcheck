def strategy_explanation(strategy):
    explanations = {
        "preserve_dynamics": (
            "AES guidance recommends lower distribution loudness for wide dynamic range content "
            "to avoid playback limiter engagement."
        ),
        "reduce_loudness_war": (
            "Excessive loudness offers no advantage under normalization and may reduce clarity."
        ),
        "risk_of_downward_normalization": (
            "Streaming platforms will attenuate content, reducing perceived loudness."
        ),
        "risk_of_upward_limiting": (
            "Upward normalization may engage device limiters, increasing distortion risk."
        ),
        "balanced": (
            "Track parameters are within distribution-safe thresholds with minimal intervention expected."
        )
    }
    return explanations.get(strategy, "")


def recommend_strategy(analysis):
    lufs = analysis.get("integrated_lufs", 0)
    plr = analysis.get("plr", 0)
    content = analysis.get("content_type", "")

    if content == "wide_dynamic_range":
        return "preserve_dynamics"
    if plr < 8:
        return "reduce_loudness_war"
    if lufs > -13:
        return "risk_of_downward_normalization"
    if lufs < -16:
        return "risk_of_upward_limiting"
    return "balanced"


def calculate_confidence_score(analysis):
    """
    Returns Compliance Confidence (0-100%) based on AES targets.
    """
    lufs = analysis.get("integrated_lufs", -14)
    tp = analysis.get("true_peak_db", -1)
    plr = analysis.get("plr", 10)
    lra = analysis.get("loudness_range", 10)

    # Score calculations (normalized 0-1)
    lufs_score = max(0, min(1, 1 - abs(lufs + 14)/10))
    tp_score = max(0, min(1, 1 - max(0, tp - (-1.2))/5))
    plr_score = max(0, min(1, plr/12))
    lra_score = max(0, min(1, 1 - abs(lra - 10)/10))

    # Weighted confidence
    confidence = (0.35*lufs_score + 0.25*tp_score + 0.2*plr_score + 0.2*lra_score) * 100
    return round(confidence, 1)


def evaluate_analysis(analysis: dict):
    results = {}

    # Distribution
    sim = analysis.get("distribution_simulation", {}).get("music_streaming", {})
    if sim:
        if sim.get("gain_change_db", 0) < -1:
            results["distribution"] = [
                "yellow",
                f"Streaming platforms will attenuate track by {abs(sim['gain_change_db']):.1f} dB."
            ]
        elif sim.get("will_limit", False):
            results["distribution"] = [
                "red",
                "Upward normalization may trigger playback limiting."
            ]
        else:
            results["distribution"] = [
                "green",
                "Distribution-safe loudness with minimal platform intervention expected."
            ]

    # PLR
    plr = analysis.get("plr")
    if plr is not None:
        if plr < 8:
            results["plr"] = ["red", f"PLR = {plr:.1f} dB. Transients likely attenuated."]
        elif plr < 10:
            results["plr"] = ["yellow", f"PLR = {plr:.1f} dB. Monitor peak limiting."]
        else:
            results["plr"] = ["green", f"PLR = {plr:.1f} dB. Healthy dynamic balance."]

    # Strategy
    strategy = recommend_strategy(analysis)
    results["strategy"] = [
        "info",
        {
            "recommended_strategy": strategy,
            "aes_alignment": True,
            "explanation": strategy_explanation(strategy)
        }
    ]

    # Compliance Confidence Score
    results["confidence_score"] = calculate_confidence_score(analysis)

    return results
