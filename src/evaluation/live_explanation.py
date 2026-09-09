from __future__ import annotations

from typing import Any


def build_live_explanation(
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build a human-readable explanation from a live proof snapshot."""

    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a dictionary.")

    signal = snapshot.get("signal")
    signal_label = str(
        snapshot.get("signal_label", "UNKNOWN")
    )
    trend = str(
        snapshot.get("trend", "UNKNOWN")
    )
    momentum = snapshot.get("momentum")
    entry_price = snapshot.get("entry_price")
    stop_loss = snapshot.get("stop_loss")
    take_profit = snapshot.get("take_profit")
    risk_reward_ratio = snapshot.get(
        "risk_reward_ratio"
    )
    market_state = str(
        snapshot.get("market_state", "UNKNOWN")
    )
    quote_stale = bool(
        snapshot.get("quote_stale", False)
    )

    reasons: list[str] = []

    if signal == 1:
        reasons.append(
            "Momentum is positive over the configured window."
        )
        if trend == "UP":
            reasons.append(
                "The moving-average trend is UP."
            )
        elif trend == "DOWN":
            reasons.append(
                "The moving-average trend is DOWN."
            )
    elif signal == 0:
        reasons.append(
            "The current momentum condition does not "
            "produce a BUY signal."
        )
        if trend == "UP":
            reasons.append(
                "The broader moving-average trend is UP, "
                "but the entry condition is not confirmed."
            )
        elif trend == "DOWN":
            reasons.append(
                "The moving-average trend is DOWN, "
                "so a BUY entry is not confirmed."
            )

    if quote_stale:
        reasons.append(
            "The latest quote is marked as stale."
        )

    if market_state != "OPEN":
        reasons.append(
            f"The market state is {market_state}."
        )

    if signal == 1 and entry_price is not None:
        action = "BUY"
        summary = (
            "A BUY condition is currently detected."
        )
    else:
        action = "NO TRADE"
        summary = (
            "No trade is currently recommended "
            "by the existing live signal logic."
        )

    return {
        "action": action,
        "summary": summary,
        "signal_label": signal_label,
        "trend": trend,
        "momentum": momentum,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": risk_reward_ratio,
        "market_state": market_state,
        "quote_stale": quote_stale,
        "reasons": reasons,
        "human_text": " ".join(
            [summary, *reasons]
        ),
    }


def validate_live_explanation(
    explanation: dict[str, Any],
) -> bool:
    """Validate the required human-readable explanation fields."""

    if not isinstance(explanation, dict):
        return False

    required = {
        "action",
        "summary",
        "signal_label",
        "trend",
        "reasons",
        "human_text",
    }

    if not required.issubset(explanation.keys()):
        return False

    if explanation["action"] not in {
        "BUY",
        "NO TRADE",
    }:
        return False

    if not isinstance(
        explanation["reasons"],
        list,
    ):
        return False

    return True
