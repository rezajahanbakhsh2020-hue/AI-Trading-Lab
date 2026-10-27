from __future__ import annotations

from typing import Any


VALID_RELEASE_DECISIONS = {"BUY", "SELL", "NO TRADE"}
MIN_STABILITY_SCORE = 0.50


def validate_final_release(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Perform the final production release validation.

    Contract:
    - BUY, SELL, and NO TRADE are first-class outcomes.
    - BUY / SELL are releasable only when every mandatory decision,
      authorization, stability, display-consistency, overlay-consistency,
      and trade-geometry gate passes.
    - NO TRADE is releasable only when every mandatory gate also passes
      AND the market data is fresh (a stale/invalid/provisional market
      state must never become releasable merely by emitting NO TRADE).
    - Stale market data (``quote_stale`` missing or True) fails closed
      for every decision outcome.
    """

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

    release_decision = str(decision.get("decision"))

    checks = {
        "end_to_end_ready": (
            result["end_to_end_ready"] is True
        ),
        "release_gate_ready": (
            release_gate.get("release_ready") is True
        ),
        "decision_supported": (
            release_decision in VALID_RELEASE_DECISIONS
        ),
        "market_data_fresh": (
            decision.get("quote_stale") is False
        ),
        "strategy_stable": (
            # Strategy identity must be bound and consistent across the
            # authorized selection, decision, and display — never hardcoded.
            bool(str(selection.get("stable_strategy") or "").strip())
            and selection.get("stable_strategy") == decision.get("stable_strategy")
            and selection.get("stable_strategy") == display.get("stable_strategy")
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

    if release_decision == "BUY":
        levels = overlay.get("levels", {})

        complete = all(
            levels.get(field) is not None
            for field in (
                "entry",
                "stop_loss",
                "tp1",
                "tp2",
                "tp3",
            )
        )

        checks["trade_levels_complete"] = complete
        checks["trade_levels_ordered"] = (
            complete
            and levels["stop_loss"] < levels["entry"]
            and levels["entry"] < levels["tp1"]
            and levels["tp1"] < levels["tp2"]
            and levels["tp2"] < levels["tp3"]
        )
    elif release_decision == "SELL":
        levels = overlay.get("levels", {})

        complete = all(
            levels.get(field) is not None
            for field in (
                "entry",
                "stop_loss",
                "tp1",
                "tp2",
                "tp3",
            )
        )

        checks["trade_levels_complete"] = complete
        checks["trade_levels_ordered"] = (
            complete
            and levels["tp3"] < levels["tp2"]
            and levels["tp2"] < levels["tp1"]
            and levels["tp1"] < levels["entry"]
            and levels["entry"] < levels["stop_loss"]
        )
    else:
        # NO TRADE: trade-level completeness and ordering do not apply.
        # The decision must not carry fabricated trade levels.
        levels = overlay.get("levels", {})

        checks["trade_levels_complete"] = all(
            levels.get(field) is None
            for field in (
                "entry",
                "stop_loss",
                "tp1",
                "tp2",
                "tp3",
            )
        )
        checks["trade_levels_ordered"] = True

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
