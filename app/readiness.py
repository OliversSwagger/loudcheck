from app.enums.readiness import ReadinessStatus
from app.enums.severity import SeverityLevel


def assess_readiness(analysis, evaluation):
    """
    AES-aligned mastering and distribution readiness assessment.

    Important:
    - AES/ITU standards inform the measurement methodology.
    - LoudCheck's readiness thresholds are application-level heuristics.
    - Content-aware rules are used so that speech is not evaluated
      using music-specific assumptions.
    - The function returns a deterministic readiness verdict.
    """

    issues = []
    actions = []

    # ---------------------------------------------------------
    # Extract analysis values safely
    # ---------------------------------------------------------

    integrated_lufs = analysis.get("integrated_lufs")
    true_peak = analysis.get("true_peak_db")
    plr = analysis.get("plr")
    content_type = analysis.get("content_type", "")
    lra = analysis.get("loudness_range")
    short_term_dynamic_range = analysis.get(
        "short_term_dynamic_range"
    )

    freq_balance = analysis.get(
        "frequency_balance",
        {"low": 0, "mid": 0, "high": 0}
    )

    stereo_balance = analysis.get("stereo_balance", 0)
    distribution = analysis.get(
        "distribution_simulation",
        {}
    )

    channel_count = analysis.get("channel_count")
    channel_layout = analysis.get("channel_layout")

    # ---------------------------------------------------------
    # Normalize values
    # ---------------------------------------------------------

    if not isinstance(content_type, str):
        content_type = ""

    content_type = content_type.lower().strip()

    if not isinstance(freq_balance, dict):
        freq_balance = {
            "low": 0,
            "mid": 0,
            "high": 0
        }

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    def add_issue(metric, severity, message):
        issues.append({
            "metric": metric,
            "severity": severity,
            "message": message
        })

    def add_action(action, recommendation):
        actions.append({
            "action": action,
            "recommendation": recommendation
        })

    # =========================================================
    # 1. TRUE PEAK SAFETY
    # =========================================================

    if true_peak is not None:

        if true_peak > -1.0:
            add_issue(
                "True Peak",
                SeverityLevel.HIGH,
                "True peak exceeds −1 dBTP inter-sample safety margin"
            )

            add_action(
                "Limiter ceiling",
                "Set true peak limiter ceiling to approximately −1.2 dBTP"
            )

        elif true_peak > -1.5:
            add_issue(
                "True Peak",
                SeverityLevel.MODERATE,
                "True peak is close to the inter-sample safety limit"
            )

            add_action(
                "Peak headroom",
                "Consider additional true-peak headroom to reduce playback risk"
            )

    # =========================================================
    # 2. DISTRIBUTION LIMITER ENGAGEMENT
    # =========================================================

    for platform, info in distribution.items():

        if not isinstance(info, dict):
            continue

        will_limit = info.get("will_limit", False)
        gain_change = info.get("gain_change_db")

        if will_limit:

            add_issue(
                f"Distribution – {platform}",
                SeverityLevel.HIGH,
                f"{platform} may apply gain reduction or playback limiting."
            )

            add_action(
                f"Inspect {platform} limiting",
                "Adjust track loudness and/or peak structure to reduce the likelihood of playback limiter engagement"
            )

        elif gain_change is not None:

            # Large upward normalization can still represent
            # a meaningful playback consideration even if the
            # simulator does not predict limiting.

            if gain_change > 6:

                add_issue(
                    f"Distribution – {platform}",
                    SeverityLevel.MODERATE,
                    f"{platform} may apply significant upward normalization."
                )

    # =========================================================
    # 3. INTEGRATED LOUDNESS
    # =========================================================

    if integrated_lufs is not None:

        # Dense music
        if content_type == "dense_music":

            if integrated_lufs < -16:

                add_issue(
                    "Integrated LUFS",
                    SeverityLevel.MODERATE,
                    "Integrated loudness is substantially below the typical streaming reference range for dense music"
                )

            elif integrated_lufs > -9:

                add_issue(
                    "Integrated LUFS",
                    SeverityLevel.HIGH,
                    "Very high integrated loudness may result in substantial normalization and increased playback processing"
                )

            elif integrated_lufs > -13:

                add_issue(
                    "Integrated LUFS",
                    SeverityLevel.MODERATE,
                    "Integrated loudness is relatively high and may result in significant normalization"
                )

        # Mixed material
        elif content_type == "mixed":

            if integrated_lufs > -9:

                add_issue(
                    "Integrated LUFS",
                    SeverityLevel.HIGH,
                    "Very high integrated loudness may result in substantial normalization and playback processing"
                )

            elif integrated_lufs > -13:

                add_issue(
                    "Integrated LUFS",
                    SeverityLevel.MODERATE,
                    "Integrated loudness is relatively high and may result in significant normalization"
                )

        # Speech
        elif content_type == "speech_dominant":

            # Do NOT impose the music target of approximately
            # -14 LUFS on speech.
            #
            # Instead, flag only unusually extreme values.

            if integrated_lufs > -9:

                add_issue(
                    "Integrated LUFS",
                    SeverityLevel.MODERATE,
                    "Speech content is unusually loud and may receive significant playback normalization"
                )

    # =========================================================
    # 4. PLR / PEAK-TO-LOUDNESS RELATIONSHIP
    # =========================================================

    if plr is not None:

        if content_type == "dense_music":

            if plr < 6:

                add_issue(
                    "PLR",
                    SeverityLevel.HIGH,
                    f"Very low PLR ({plr:.1f} dB); peak structure may be heavily limited or compressed"
                )

                add_action(
                    "Dynamic processing",
                    "Inspect limiting and compression to restore appropriate transient headroom"
                )

            elif plr < 8:

                add_issue(
                    "PLR",
                    SeverityLevel.MODERATE,
                    f"Low PLR ({plr:.1f} dB); dynamics may be overly compressed"
                )

        elif content_type == "speech_dominant":

            # Speech can legitimately have lower peak-to-loudness
            # relationships than music, so use a less aggressive
            # threshold.

            if plr < 3:

                add_issue(
                    "PLR",
                    SeverityLevel.MODERATE,
                    f"Very low PLR ({plr:.1f} dB); inspect speech dynamics processing and peak structure"
                )

        else:

            # Mixed / unknown material
            if plr < 6:

                add_issue(
                    "PLR",
                    SeverityLevel.MODERATE,
                    f"Low PLR ({plr:.1f} dB); inspect compression and limiting"
                )

    # =========================================================
    # 5. LOUDNESS RANGE
    # =========================================================

    if lra is not None:

        if content_type == "dense_music":

            if lra < 2:

                add_issue(
                    "Loudness Range",
                    SeverityLevel.MODERATE,
                    f"Very narrow loudness range ({lra:.1f} dB); inspect macro-dynamics"
                )

        elif content_type == "mixed":

            if lra < 1:

                add_issue(
                    "Loudness Range",
                    SeverityLevel.MODERATE,
                    f"Extremely narrow loudness range ({lra:.1f} dB); inspect dynamic processing"
                )

        elif content_type == "speech_dominant":

            if lra < 1:

                add_issue(
                    "Loudness Range",
                    SeverityLevel.MODERATE,
                    f"Extremely narrow loudness range ({lra:.1f} dB); inspect speech compression and gating"
                )

    # =========================================================
    # 6. SHORT-TERM DYNAMIC RANGE
    # =========================================================

    if short_term_dynamic_range is not None:

        if content_type == "dense_music":

            if short_term_dynamic_range < 2:

                add_issue(
                    "Short-term Dynamic Range",
                    SeverityLevel.MODERATE,
                    "Very low short-term dynamic variation detected; inspect compression and limiting"
                )

        elif content_type == "speech_dominant":

            if short_term_dynamic_range < 1:

                add_issue(
                    "Short-term Dynamic Range",
                    SeverityLevel.MODERATE,
                    "Very low short-term dynamic variation detected; inspect speech processing"
                )

    # =========================================================
    # 7. STEREO BALANCE
    # =========================================================

    # Only evaluate stereo balance when meaningful stereo
    # information exists.

    stereo_available = analysis.get(
        "stereo_available",
        False
    )

    if stereo_available and stereo_balance is not None:

        if abs(stereo_balance) > 1.0:

            add_issue(
                "Stereo Balance",
                SeverityLevel.MODERATE,
                "Significant stereo imbalance detected"
            )

        elif abs(stereo_balance) > 0.5:

            add_issue(
                "Stereo Balance",
                SeverityLevel.MODERATE,
                "Stereo balance is outside the preferred tolerance"
            )

    # =========================================================
    # 8. FREQUENCY BALANCE
    # =========================================================

    low = freq_balance.get("low", 0)
    mid = freq_balance.get("mid", 0)
    high = freq_balance.get("high", 0)

    if (
        low > 70
        or high > 70
        or mid < 20
    ):

        add_issue(
            "Frequency Balance",
            SeverityLevel.MODERATE,
            "Spectral energy is uneven and may produce inconsistent playback across systems"
        )

    # =========================================================
    # 9. CHANNEL INFORMATION
    # =========================================================

    # This is informational rather than a failure condition.
    # Multichannel material should not be treated as stereo.

    if channel_count is not None and channel_count > 2:

        # Do not add an issue merely because material is
        # multichannel. It is valid content.

        pass

    # =========================================================
    # 10. EXISTING EVALUATION FLAGS
    # =========================================================

    for key, value in evaluation.items():

        if isinstance(value, tuple):

            status, _ = value

            if status == "fail":

                add_issue(
                    key,
                    SeverityLevel.HIGH,
                    "Fails LoudCheck evaluation criteria"
                )

    # =========================================================
    # 11. DETERMINE READINESS
    # =========================================================

    high_count = sum(
        1
        for issue in issues
        if issue["severity"] == SeverityLevel.HIGH
    )

    moderate_count = sum(
        1
        for issue in issues
        if issue["severity"] == SeverityLevel.MODERATE
    )

    # ---------------------------------------------------------
    # Verdict logic
    # ---------------------------------------------------------
    #
    # LoudCheck distinguishes between:
    #
    #   HIGH      = significant technical/distribution risk
    #   MODERATE  = review recommended
    #   READY     = no significant detected risk
    #
    # The number of moderate diagnostic observations must NOT
    # automatically make a file NOT_READY.
    #
    # For example, speech material may legitimately have:
    #
    #   - very low PLR
    #   - very low LRA
    #   - low short-term variation
    #   - highly concentrated mid-band energy
    #
    # These are useful diagnostic findings, but they are not
    # necessarily distribution-blocking failures.
    # ---------------------------------------------------------

    if high_count > 0:

        verdict = ReadinessStatus.CONDITIONALLY_READY

    elif moderate_count > 0:

        verdict = ReadinessStatus.CONDITIONALLY_READY

    else:

        verdict = ReadinessStatus.READY

    # =========================================================
    # 12. CONFIDENCE CALCULATION
    # =========================================================

    confidence_penalty = (
        high_count * 15
        + moderate_count * 7
    )

    base_confidence = evaluation.get(
        "confidence_score",
        100
    )

    if not isinstance(base_confidence, (int, float)):

        base_confidence = 100

    estimated_post_fix_confidence = max(
        0,
        min(
            100,
            base_confidence - confidence_penalty
        )
    )

    # =========================================================
    # 13. WARNING CLASSIFICATION
    # =========================================================

    warnings = []
    mobile_warnings = []

    for issue in issues:

        metric = issue["metric"]

        warning = {
            "severity": issue["severity"].value.lower(),
            "message": issue["message"]
        }

        if (
            "Distribution" in metric
            or "True Peak" in metric
        ):

            mobile_warnings.append(warning)

        else:

            warnings.append(issue["message"])

    # =========================================================
    # 14. RETURN AUTHORITATIVE RESULT
    # =========================================================

    return {
        "verdict": verdict,
        "summary": _verdict_summary(verdict),
        "issues": issues,
        "required_actions": actions,
        "estimated_confidence_after_fixes": round(
            estimated_post_fix_confidence,
            1
        ),
        "warnings": warnings,
        "mobile_warnings": mobile_warnings
    }


def _verdict_summary(verdict: ReadinessStatus) -> str:

    return {

        ReadinessStatus.READY:
            "Meets LoudCheck's current distribution-readiness criteria with no significant detected risk factors",

        ReadinessStatus.CONDITIONALLY_READY:
            "Suitable for distribution after addressing identified risk factors",

        ReadinessStatus.NOT_READY:
            "Multiple significant playback, dynamics, or distribution risks detected and review is recommended before distribution"

    }[verdict]
