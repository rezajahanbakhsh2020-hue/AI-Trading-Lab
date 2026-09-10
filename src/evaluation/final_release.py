from __future__ import annotations

from typing import Any


REQUIRED_DECISION = "BUY"
REQUIRED_STRATEGY = "momentum"
MIN_STABILITY_SCORE = 0.50


def validate_final_release(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Perform the final production release validation."""

    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary.")

    required_sections = {
        "decision",
        "display",
        "overlay",
        "release_gate",
        "production_selection",
        "end_to_end_ready",
    }

    missing = required_sections.difference(result)

    if missing:
        raise ValueError(
            "Missing final release sections: "
            + ", ".join(sorted(missing))
        )

    decision = result["decision"]
    display = result["display"]
    overlay = result["overlay"]
    release_gate = result["release_gate"]
    selection = result["production_selection"]

    checks = {
        "end_to_end_ready": (
            result["end_to_end_ready"] is True
        ),
        "release_gate_ready": (
            release_gate.get("release_ready") is True
        ),
        "decision_present": (
            decision.get("decision") == REQUIRED_DECISION
        ),
        "strategy_stable": (
            selection.get("stable_strategy")
            == REQUIRED_STRATEGY
        ),
        "stability_score_valid": (
            float(selection.get("stability_score", 0.0))
            >= MIN_STABILITY_SCORE
        ),
        "display_matches_decision": (
            display.get("decision")
            == decision.get("decision")
        ),
        "overlay_matches_decision": (
            overlay.get("decision")
            == decision.get("decision")
        ),
    }

    if decision.get("decision") == REQUIRED_DECISION:
        levels = overlay.get("levels", {})

        checks["trade_levels_complete"] = all(
            levels.get(field) is not None
            for field in (
                "entry",
                "stop_loss",
                "tp1",
                "tp2",
                "tp3",
            )
        )

        checks["trade_levels_ordered"] = (
            checks["trade_levels_complete"]
            and levels["stop_loss"] < levels["entry"]
            and levels["entry"] < levels["tp1"]
            and levels["tp1"] < levels["tp2"]
            and levels["tp2"] < levels["tp3"]
        )
    else:
        checks["trade_levels_complete"] = False
        checks["trade_levels_ordered"] = False

    ready = all(checks.values())

    return {
        "final_release_ready": ready,
        "checks": checks,
        "status": (
            "RELEASE READY"
            if ready
            else "RELEASE BLOCKED"
        ),
        "strategy": selection.get("stable_strategy"),
        "stability_score": float(
            selection.get("stability_score", 0.0)
        ),
        "decision": decision.get("decision"),
    }
