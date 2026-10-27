from __future__ import annotations

from typing import Any


VALID_RELEASE_DECISIONS = {"BUY", "SELL", "NO TRADE"}
DEFAULT_MIN_STABILITY_SCORE = 0.50
DEFAULT_REQUIRED_STRATEGY = "momentum"


def validate_live_release(
    *,
    decision: dict[str, Any],
    display: dict[str, Any],
    min_stability_score: float = DEFAULT_MIN_STABILITY_SCORE,
    required_strategy: str = DEFAULT_REQUIRED_STRATEGY,
) -> dict[str, Any]:
    """Validate the complete live system before release."""

    if not isinstance(decision, dict):
        raise TypeError("decision must be a dictionary.")

    if not isinstance(display, dict):
        raise TypeError("display must be a dictionary.")

    required_decision_fields = {
        "decision",
        "trend",
        "stable_strategy",
        "stability_score",
    }

    missing_decision = required_decision_fields.difference(
        decision
    )
    if missing_decision:
        raise ValueError(
            "Missing decision fields: "
            + ", ".join(sorted(missing_decision))
        )

    required_display_fields = {
        "decision",
        "stable_strategy",
        "stability_score",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "tp3",
    }

    missing_display = required_display_fields.difference(
        display
    )
    if missing_display:
        raise ValueError(
            "Missing display fields: "
            + ", ".join(sorted(missing_display))
        )

    release_decision = str(decision["decision"])

    checks: dict[str, bool] = {
        "decision_supported": (
            release_decision in VALID_RELEASE_DECISIONS
        ),
        "decision_display_match": (
            decision["decision"] == display["decision"]
        ),
        "required_strategy": (
            decision["stable_strategy"] == required_strategy
            and display["stable_strategy"] == required_strategy
        ),
        "stability_threshold": (
            float(decision["stability_score"])
            >= min_stability_score
            and float(display["stability_score"])
            >= min_stability_score
        ),
        # Stale or invalid market data blocks the release for every
        # decision outcome — NO TRADE must never be a vehicle for
        # stale/invalid input concealment (fail closed when absent
        # or the flag is missing).
        "market_data_fresh": (
            decision.get("quote_stale") is False
        ),
    }

    if release_decision == "BUY":
        levels_valid = (
            decision.get("entry_price") is not None
            and decision.get("stop_loss") is not None
            and decision.get("take_profit") is not None
            and display["entry_price"] is not None
            and display["stop_loss"] is not None
            and display["tp1"] is not None
            and display["tp2"] is not None
            and display["tp3"] is not None
            and display["stop_loss"] < display["entry_price"]
            and display["entry_price"] < display["tp1"]
            and display["tp1"] < display["tp2"]
            and display["tp2"] < display["tp3"]
        )
    elif release_decision == "SELL":
        levels_valid = (
            decision.get("entry_price") is not None
            and decision.get("stop_loss") is not None
            and decision.get("take_profit") is not None
            and display["entry_price"] is not None
            and display["stop_loss"] is not None
            and display["tp1"] is not None
            and display["tp2"] is not None
            and display["tp3"] is not None
            and display["tp3"] < display["tp2"]
            and display["tp2"] < display["tp1"]
            and display["tp1"] < display["entry_price"]
            and display["entry_price"] < display["stop_loss"]
        )
    else:
        # NO TRADE: no fabricated trade levels may be attached.
        levels_valid = all(
            display[field] is None
            for field in (
                "entry_price",
                "stop_loss",
                "tp1",
                "tp2",
                "tp3",
            )
        )

    checks["trade_levels_valid"] = levels_valid

    release_ready = all(checks.values())

    return {
        "release_ready": release_ready,
        "checks": checks,
        "decision": decision["decision"],
        "stable_strategy": decision["stable_strategy"],
        "stability_score": float(decision["stability_score"]),
    }
