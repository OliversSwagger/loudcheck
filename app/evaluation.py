def strategy_explanation(strategy):
    explanations = {

        "preserve_dynamics": (
            "AES-aligned guidance supports preserving dynamic range in "
            "wide-dynamic-range material and avoiding unnecessary playback "
            "limiting during distribution."
        ),

        "reduce_loudness_war": (
            "Excessive loudness provides limited benefit under loudness "
            "normalization and may reduce transient clarity and dynamic range."
        ),

        "risk_of_downward_normalization": (
            "Content significantly above the selected distribution reference "
            "is likely to be attenuated by normalized playback systems."
        ),

        "risk_of_upward_limiting": (
            "Content significantly below the selected distribution reference "
            "may be turned up during playback. If resulting peaks exceed the "
            "playback system's available headroom, playback limiting may occur."
        ),

        "balanced": (
            "Measured parameters are within LoudCheck's selected distribution "
            "assessment ranges, with no major intervention indicated by the "
            "current analysis."
        )
    }

    return explanations.get(strategy, "")


def recommend_strategy(analysis):
    lufs = analysis.get(
        "integrated_lufs",
        0
    )

    plr = analysis.get(
        "plr",
        0
    )

    content = analysis.get(
        "content_type",
        ""
    )

    # ---------------------------------------------------------
    # Wide dynamic range takes priority.
    # ---------------------------------------------------------

    if content == "wide_dynamic_range":
        return "preserve_dynamics"

    # ---------------------------------------------------------
    # Very low PLR is primarily relevant to dense music.
    #
    # Do NOT use the same PLR threshold for speech-dominant
    # material because speech can naturally have very different
    # peak-to-loudness characteristics.
    # ---------------------------------------------------------

    if content == "dense_music" and plr < 8:
        return "reduce_loudness_war"

    # ---------------------------------------------------------
    # Significantly loud relative to the selected reference.
    # ---------------------------------------------------------

    if lufs > -13:
        return "risk_of_downward_normalization"

    # ---------------------------------------------------------
    # Significantly quiet relative to the selected reference.
    # ---------------------------------------------------------

    if lufs < -16:
        return "risk_of_upward_limiting"

    return "balanced"


def calculate_confidence_score(analysis):
    """
    Returns LoudCheck Compliance Confidence (0-100%).

    This is a LoudCheck engineering confidence score based on
    selected loudness, true-peak, PLR and loudness-range criteria.

    It should not be interpreted as an official AES certification
    or AES compliance measurement.
    """

    lufs = analysis.get(
        "integrated_lufs",
        -14
    )

    tp = analysis.get(
        "true_peak_db",
        -1
    )

    plr = analysis.get(
        "plr",
        10
    )

    lra = analysis.get(
        "loudness_range",
        10
    )

    content = analysis.get(
        "content_type",
        ""
    )

    # ---------------------------------------------------------
    # LUFS score
    # ---------------------------------------------------------

    lufs_score = max(
        0.0,
        min(
            1.0,
            1.0 - abs(lufs + 14.0) / 10.0
        )
    )

    # ---------------------------------------------------------
    # True Peak score
    # ---------------------------------------------------------

    tp_score = max(
        0.0,
        min(
            1.0,
            1.0 - max(
                0.0,
                tp - (-1.2)
            ) / 5.0
        )
    )

    # ---------------------------------------------------------
    # PLR score
    #
    # PLR is weighted for music-oriented analysis.
    # Speech is not penalized using the music PLR model.
    # ---------------------------------------------------------

    if content == "dense_music":
        plr_score = max(
            0.0,
            min(
                1.0,
                plr / 12.0
            )
        )
    else:
        plr_score = 1.0

    # ---------------------------------------------------------
    # LRA score
    # ---------------------------------------------------------

    lra_score = max(
        0.0,
        min(
            1.0,
            1.0 - abs(lra - 10.0) / 10.0
        )
    )

    # ---------------------------------------------------------
    # Weighted LoudCheck confidence
    # ---------------------------------------------------------

    confidence = (
        0.35 * lufs_score
        +
        0.25 * tp_score
        +
        0.20 * plr_score
        +
        0.20 * lra_score
    ) * 100.0

    return round(
        confidence,
        1
    )


def evaluate_analysis(analysis: dict):
    results = {}

    content = analysis.get(
        "content_type",
        ""
    )

    # =========================================================
    # DISTRIBUTION
    # =========================================================

    sim = (
        analysis
        .get(
            "distribution_simulation",
            {}
        )
        .get(
            "music_streaming",
            {}
        )
    )

    if sim:

        gain_change = sim.get(
            "gain_change_db",
            0
        )

        will_limit = sim.get(
            "will_limit",
            False
        )

        if gain_change < -1:

            results["distribution"] = [
                "yellow",
                (
                    "Normalized playback is expected to attenuate "
                    f"the track by {abs(gain_change):.1f} dB."
                )
            ]

        elif will_limit:

            results["distribution"] = [
                "red",
                (
                    "Upward normalization may increase playback level "
                    "and may trigger playback limiting."
                )
            ]

        else:

            results["distribution"] = [
                "green",
                (
                    "No significant intervention is predicted by the "
                    "current LoudCheck distribution simulation."
                )
            ]

    # =========================================================
    # PLR
    #
    # Only apply the music-oriented PLR assessment to dense music.
    # =========================================================

    plr = analysis.get(
        "plr"
    )

    if plr is not None:

        if content == "dense_music":

            if plr < 8:

                results["plr"] = [
                    "red",
                    (
                        f"PLR = {plr:.1f} dB. "
                        "Very low peak-to-loudness separation may indicate "
                        "heavy limiting or reduced transient headroom."
                    )
                ]

            elif plr < 10:

                results["plr"] = [
                    "yellow",
                    (
                        f"PLR = {plr:.1f} dB. "
                        "Monitor peak limiting and transient preservation."
                    )
                ]

            else:

                results["plr"] = [
                    "green",
                    (
                        f"PLR = {plr:.1f} dB. "
                        "Peak-to-loudness separation is within the "
                        "current LoudCheck assessment range."
                    )
                ]

        else:

            # -------------------------------------------------
            # Speech / mixed / other material
            #
            # Report the measurement but do not classify it
            # using the dense-music PLR thresholds.
            # -------------------------------------------------

            results["plr"] = [
                "info",
                (
                    f"PLR = {plr:.1f} dB. "
                    "PLR is reported as an engineering diagnostic; "
                    "the dense-music PLR thresholds are not applied "
                    f"to {content or 'this'} content."
                )
            ]

    # =========================================================
    # STRATEGY
    # =========================================================

    strategy = recommend_strategy(
        analysis
    )

    results["strategy"] = [
        "info",
        {
            "recommended_strategy": strategy,

            # AES-aligned means LoudCheck's engineering strategy
            # is informed by AES-related principles.
            #
            # It does NOT mean AES has certified LoudCheck.
            "aes_alignment": True,

            "explanation": strategy_explanation(
                strategy
            )
        }
    ]

    # =========================================================
    # CONFIDENCE SCORE
    # =========================================================

    results["confidence_score"] = (
        calculate_confidence_score(
            analysis
        )
    )

    return results
