from app.enums.readiness import ReadinessStatus
from app.enums.severity import SeverityLevel

def assess_readiness(analysis, evaluation):
    """
    AES-aligned mastering & distribution readiness assessment.
    Returns a deterministic verdict with technical reasoning.
    """

    issues = []
    actions = []

    integrated_lufs = analysis.get("integrated_lufs")
    true_peak = analysis.get("true_peak_db")
    plr = analysis.get("plr")
    content_type = analysis.get("content_type")
    lra = analysis.get("loudness_range", 10)
    freq_balance = analysis.get("frequency_balance", {"low":0,"mid":0,"high":0})
    stereo_balance = analysis.get("stereo_balance", 0)
    distribution = analysis.get("distribution_simulation", {})

    # -------------------------
    # Check critical issues first
    # -------------------------

    # True Peak safety (critical)
    if true_peak is not None and true_peak > -1.0:
        issues.append({
            "metric": "True Peak",
            "severity": SeverityLevel.HIGH,
            "message": "True peak exceeds −1 dBTP inter-sample safety margin"
        })
        actions.append({
            "action": "Limiter ceiling",
            "recommendation": "Set true peak limiter ceiling to −1.2 dBTP"
        })

    # Distribution limiter engagement
    for platform, info in distribution.items():
        if info.get("will_limit"):
            issues.append({
                "metric": f"Distribution – {platform}",
                "severity": SeverityLevel.HIGH,
                "message": f"{platform} will apply gain reduction or limiting."
            })
            actions.append({
                "action": f"Inspect {platform} limiting",
                "recommendation": "Adjust track peaks or loudness to avoid limiter engagement"
            })

    # Loudness sanity (non-fatal)
    if integrated_lufs is not None:
        if content_type == "dense_music" and integrated_lufs < -16:
            issues.append({
                "metric": "Integrated LUFS",
                "severity": SeverityLevel.MODERATE,
                "message": "Loudness below typical music streaming target"
            })

    # Dynamic range
    if plr is not None and content_type == "dense_music":
        if plr < 8:
            issues.append({
                "metric": "PLR",
                "severity": SeverityLevel.MODERATE,
                "message": "Low punch-to-loudness ratio; dynamics may be overly compressed"
            })

    # Stereo balance
    if abs(stereo_balance) > 0.5:
        issues.append({
            "metric": "Stereo Balance",
            "severity": SeverityLevel.MODERATE,
            "message": "Excessive stereo imbalance may collapse on mono/mobile playback"
        })

    # Frequency balance
    if freq_balance.get("low", 0) > 70 or freq_balance.get("high",0) > 70 or freq_balance.get("mid",0) < 20:
        issues.append({
            "metric": "Frequency Balance",
            "severity": SeverityLevel.MODERATE,
            "message": "Spectral energy is uneven; may sound unbalanced across devices"
        })

    # Evaluation flags (existing AES checks)
    for key, value in evaluation.items():
        if isinstance(value, tuple):
            status, _ = value
            if status == "fail":
                issues.append({
                    "metric": key,
                    "severity": SeverityLevel.HIGH,
                    "message": "Fails AES evaluation criteria"
                })

    # -------------------------
    # Decide readiness verdict
    # -------------------------
    if any(i["severity"] == SeverityLevel.HIGH for i in issues):
        verdict = ReadinessStatus.CONDITIONALLY_READY
    else:
        verdict = ReadinessStatus.READY

    # Too many moderate issues → downgrade to NOT_READY
    moderate_count = sum(1 for i in issues if i["severity"] == SeverityLevel.MODERATE)
    if len(issues) >= 3 or moderate_count >= 4:
        verdict = ReadinessStatus.NOT_READY

    # -------------------------
    # Confidence calculation (penalize issues)
    # -------------------------
    confidence_penalty = sum(15 if i["severity"] == SeverityLevel.HIGH else 7 for i in issues)
    base_confidence = evaluation.get("confidence_score", 100)
    estimated_post_fix_confidence = max(0, min(100, base_confidence - confidence_penalty))

    # -------------------------
    # Filter warnings for consistency
    # -------------------------
    warnings = []
    mobile_warnings = []
    for i in issues:
        if "Distribution" in i["metric"] or "True Peak" in i["metric"]:
            mobile_warnings.append({
                "severity": i["severity"].value.lower(),
                "message": i["message"]
            })
        else:
            warnings.append(i["message"])

    # -------------------------
    # Return authoritative verdict
    # -------------------------
    return {
        "verdict": verdict,
        "summary": _verdict_summary(verdict),
        "issues": issues,
        "required_actions": actions,
        "estimated_confidence_after_fixes": round(estimated_post_fix_confidence, 1),
        "warnings": warnings,
        "mobile_warnings": mobile_warnings
    }


def _verdict_summary(verdict: ReadinessStatus) -> str:
    return {
        ReadinessStatus.READY:
            "Meets AES-based loudness, peak safety, and playback predictability criteria",
        ReadinessStatus.CONDITIONALLY_READY:
            "Suitable for distribution after addressing identified risk factors",
        ReadinessStatus.NOT_READY:
            "High risk of playback distortion or loudness penalties detected"
    }[verdict]
