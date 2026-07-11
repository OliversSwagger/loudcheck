from typing import List, Dict
from statistics import pstdev


def build_batch_summary(results: List[Dict]) -> Dict:
    summary = {
        "counts": {
            "ready": 0,
            "conditionally_ready": 0,
            "not_ready": 0,
        },
        "tracks_by_status": {
            "ready": [],
            "conditionally_ready": [],
            "not_ready": [],
        },
        "statistics": {
            "average_lufs": 0.0,
            "average_true_peak": 0.0,
            "average_plr": 0.0,
            "min_lufs": None,
            "max_lufs": None,
            "min_true_peak": None,
            "max_true_peak": None,
            "min_plr": None,
            "max_plr": None,
            "lufs_std_dev": 0.0,
        },
        "album_consistency": {},
        "release_status": {},
        "highest_risk_track": None,
    }

    lufs_vals = []
    tp_vals = []
    plr_vals = []

    lowest_confidence = 101
    risk_track = None

    for item in results:
        filename = item["filename"]
        analysis = item["analysis"]
        readiness = item["readiness"]
        verdict = readiness["verdict"].value

        summary["counts"][verdict] += 1
        summary["tracks_by_status"][verdict].append(filename)

        lufs = analysis.get("integrated_lufs")
        tp = analysis.get("true_peak_db")
        plr = analysis.get("plr")

        if lufs is not None:
            lufs_vals.append(lufs)
        if tp is not None:
            tp_vals.append(tp)
        if plr is not None:
            plr_vals.append(plr)

        confidence = item["evaluation"].get("confidence_score", 100)
        if confidence < lowest_confidence:
            lowest_confidence = confidence
            risk_track = {
                "filename": filename,
                "confidence_score": confidence,
                "reason": readiness["summary"]
            }

    # --- Aggregate statistics ---
    if lufs_vals:
        summary["statistics"].update({
            "average_lufs": round(sum(lufs_vals) / len(lufs_vals), 2),
            "min_lufs": round(min(lufs_vals), 2),
            "max_lufs": round(max(lufs_vals), 2),
            "lufs_std_dev": round(pstdev(lufs_vals), 2) if len(lufs_vals) > 1 else 0.0
        })

    if tp_vals:
        summary["statistics"].update({
            "average_true_peak": round(sum(tp_vals) / len(tp_vals), 2),
            "min_true_peak": round(min(tp_vals), 2),
            "max_true_peak": round(max(tp_vals), 2),
        })

    if plr_vals:
        summary["statistics"].update({
            "average_plr": round(sum(plr_vals) / len(plr_vals), 2),
            "min_plr": round(min(plr_vals), 2),
            "max_plr": round(max(plr_vals), 2),
        })

    # --- Album consistency evaluation ---
    summary["album_consistency"] = compute_album_consistency(summary["statistics"])

    # --- Release blocking rule ---
    release_blocked = summary["counts"]["not_ready"] > 0
    summary["release_status"] = {
        "blocked": release_blocked,
        "reason": (
            "One or more tracks are NOT READY"
            if release_blocked
            else "All tracks meet minimum AES safety thresholds"
        )
    }

    summary["highest_risk_track"] = risk_track

    return summary

def compute_album_consistency(stats: Dict) -> Dict:
    lufs_spread = (
        stats["max_lufs"] - stats["min_lufs"]
        if stats["max_lufs"] is not None
        else 0.0
    )
    tp_spread = (
        stats["max_true_peak"] - stats["min_true_peak"]
        if stats["max_true_peak"] is not None
        else 0.0
    )
    plr_spread = (
        stats["max_plr"] - stats["min_plr"]
        if stats["max_plr"] is not None
        else 0.0
    )

    # --- Consistency scoring (100 = ideal) ---
    score = 100
    if lufs_spread > 2.0:
        score -= 25
    if tp_spread > 0.5:
        score -= 25
    if plr_spread > 2.0:
        score -= 20
    if stats["lufs_std_dev"] > 1.5:
        score -= 15

    score = max(0, score)

    return {
        "lufs_spread": round(lufs_spread, 2),
        "true_peak_spread": round(tp_spread, 2),
        "plr_spread": round(plr_spread, 2),
        "lufs_std_dev": stats["lufs_std_dev"],
        "consistency_score": score,
        "assessment": (
            "Consistent album mastering"
            if score >= 80
            else "Noticeable loudness or dynamic inconsistencies detected"
        )
    }
