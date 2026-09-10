from __future__ import annotations

from collections.abc import Mapping
from typing import Any


VALID_DECISIONS = {"LONG", "SHORT", "NO TRADE"}
VALID_SIGNALS = {"BUY", "SELL", "NO TRADE"}
VALID_TRENDS = {
    "UP",
    "DOWN",
    "FLAT",
    "INSUFFICIENT DATA",
}

DEFAULT_MIN_RISK_REWARD = 1.0


def _normalize_signal(snapshot: Mapping[str, Any]) -> str:
    label = snapshot.get("signal_label")

    if label is not None:
        normalized = str(label).strip().upper()

        if normalized in VALID_SIGNALS:
            return normalized

    signal = snapshot.get("signal")

    if isinstance(signal, bool):
        return "BUY" if signal else "NO TRADE"

    if isinstance(signal, (int, float)):
        if signal > 0:
            return "BUY"
        if signal < 0:
            return "SELL"
        return "NO TRADE"

    return "NO TRADE"


def _normalize_trend(snapshot: Mapping[str, Any]) -> str:
    trend = str(
        snapshot.get("trend", "INSUFFICIENT DATA")
    ).strip().upper()

    if trend not in VALID_TRENDS:
        return "INSUFFICIENT DATA"

    return trend


def _is_stale(snapshot: Mapping[str, Any]) -> bool:
    return bool(snapshot.get("quote_stale", False))


def _market_is_open(snapshot: Mapping[str, Any]) -> bool:
    state = snapshot.get(
        "market_state",
        snapshot.get("marketState"),
    )

    if state is None:
        return True

    normalized = str(state).strip().upper()

    return normalized not in {
        "CLOSED",
        "OFFLINE",
        "UNAVAILABLE",
    }


def _risk_reward(snapshot: Mapping[str, Any]) -> float | None:
    value = snapshot.get("risk_reward_ratio")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_is_valid(
    snapshot: Mapping[str, Any],
    minimum_risk_reward: float,
) -> bool:
    ratio = _risk_reward(snapshot)

    if ratio is None:
        return False

    return ratio >= minimum_risk_reward


def _decision_reason(
    decision: str,
    signal: str,
    trend: str,
    snapshot: Mapping[str, Any],
    minimum_risk_reward: float,
) -> str:
    if _is_stale(snapshot):
        return "Quote is stale; no trade is allowed."

    if not _market_is_open(snapshot):
        return "Market is not open; no trade is allowed."

    if trend == "INSUFFICIENT DATA":
        return "Insufficient trend data; no trade is allowed."

    if not _risk_is_valid(
        snapshot,
        minimum_risk_reward,
    ):
        return (
            "Risk/reward is missing or below the "
            "minimum accepted threshold."
        )

    if signal == "NO TRADE":
        return "The strategy does not produce an active trade signal."

    if decision == "LONG":
        return (
            "BUY signal is aligned with the UP trend "
            "and risk conditions are valid."
        )

    if decision == "SHORT":
        return (
            "SELL signal is aligned with the DOWN trend "
            "and risk conditions are valid."
        )

    if signal == "BUY" and trend == "DOWN":
        return (
            "BUY signal conflicts with the DOWN trend; "
            "counter-trend trading is rejected."
        )

    if signal == "SELL" and trend == "UP":
        return (
            "SELL signal conflicts with the UP trend; "
            "counter-trend trading is rejected."
        )

    if trend == "FLAT":
        return (
            "Trend is FLAT; directional confirmation is "
            "not strong enough for a trade."
        )

    return "Signal and trend are not sufficiently aligned."


def _calculate_confidence(
    decision: str,
    signal: str,
    trend: str,
    snapshot: Mapping[str, Any],
) -> float:
    if decision == "NO TRADE":
        return 0.0

    score = 0.0

    if signal in {"BUY", "SELL"}:
        score += 40.0

    if (
        (signal == "BUY" and trend == "UP")
        or (signal == "SELL" and trend == "DOWN")
    ):
        score += 35.0

    if not _is_stale(snapshot):
        score += 10.0

    if _market_is_open(snapshot):
        score += 5.0

    ratio = _risk_reward(snapshot)

    if ratio is not None:
        if ratio >= 2.0:
            score += 10.0
        elif ratio >= 1.0:
            score += 5.0

    return round(min(score, 100.0), 2)


def build_live_trade_decision(
    snapshot: Mapping[str, Any],
    minimum_risk_reward: float = DEFAULT_MIN_RISK_REWARD,
) -> dict[str, Any]:
    """
    Build the final human-facing trade decision.

    The function does not recalculate the underlying strategy signal,
    trend, entry, stop loss, or take profit. It combines the values
    already present in the live snapshot into one deterministic
    LONG / SHORT / NO TRADE decision.

    Counter-trend signals are rejected by default. A BUY signal requires
    an UP trend for LONG, and a SELL signal requires a DOWN trend for
    SHORT.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    try:
        minimum_risk_reward = float(minimum_risk_reward)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "minimum_risk_reward must be numeric"
        ) from exc

    if minimum_risk_reward < 0:
        raise ValueError(
            "minimum_risk_reward must be non-negative"
        )

    signal = _normalize_signal(snapshot)
    trend = _normalize_trend(snapshot)

    if _is_stale(snapshot):
        decision = "NO TRADE"
    elif not _market_is_open(snapshot):
        decision = "NO TRADE"
    elif not _risk_is_valid(
        snapshot,
        minimum_risk_reward,
    ):
        decision = "NO TRADE"
    elif signal == "BUY" and trend == "UP":
        decision = "LONG"
    elif signal == "SELL" and trend == "DOWN":
        decision = "SHORT"
    else:
        decision = "NO TRADE"

    reason = _decision_reason(
        decision,
        signal,
        trend,
        snapshot,
        minimum_risk_reward,
    )

    confidence = _calculate_confidence(
        decision,
        signal,
        trend,
        snapshot,
    )

    return {
        "decision": decision,
        "decision_label": decision,
        "signal": signal,
        "trend": trend,
        "confidence": confidence,
        "confidence_pct": confidence,
        "reason": reason,
        "trade_allowed": decision in {"LONG", "SHORT"},
        "counter_trend": (
            (signal == "BUY" and trend == "DOWN")
            or (signal == "SELL" and trend == "UP")
        ),
        "risk_reward_ratio": _risk_reward(snapshot),
        "minimum_risk_reward": minimum_risk_reward,
        "entry_price": snapshot.get("entry_price"),
        "stop_loss": snapshot.get("stop_loss"),
        "take_profit": snapshot.get("take_profit"),
        "tp1": snapshot.get(
            "tp1",
            snapshot.get("take_profit"),
        ),
        "tp2": snapshot.get("tp2"),
        "tp3": snapshot.get("tp3"),
        "quote_stale": _is_stale(snapshot),
        "market_state": snapshot.get(
            "market_state",
            snapshot.get("marketState"),
        ),
        "strategy": snapshot.get("strategy"),
        "stability_score": snapshot.get(
            "stability_score"
        ),
        "timestamp": snapshot.get("timestamp"),
        "symbol": snapshot.get("symbol"),
        "interval": snapshot.get("interval"),
    }


def validate_live_trade_decision(
    decision: Mapping[str, Any],
) -> bool:
    """
    Validate the structural integrity of a final trade decision.
    """
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping")

    required_fields = (
        "decision",
        "decision_label",
        "signal",
        "trend",
        "confidence",
        "confidence_pct",
        "reason",
        "trade_allowed",
        "counter_trend",
    )

    missing = [
        field
        for field in required_fields
        if field not in decision
    ]

    if missing:
        raise ValueError(
            "decision is missing required fields: "
            + ", ".join(missing)
        )

    if decision["decision"] not in VALID_DECISIONS:
        raise ValueError(
            "decision must be LONG, SHORT, or NO TRADE"
        )

    if decision["decision_label"] != decision["decision"]:
        raise ValueError(
            "decision_label must match decision"
        )

    confidence = decision["confidence"]

    if not isinstance(
        confidence,
        (int, float),
    ):
        raise ValueError(
            "confidence must be numeric"
        )

    if not 0.0 <= float(confidence) <= 100.0:
        raise ValueError(
            "confidence must be between 0 and 100"
        )

    if not isinstance(
        decision["trade_allowed"],
        bool,
    ):
        raise ValueError(
            "trade_allowed must be boolean"
        )

    if not isinstance(
        decision["counter_trend"],
        bool,
    ):
        raise ValueError(
            "counter_trend must be boolean"
        )

    expected_trade_allowed = decision[
        "decision"
    ] in {"LONG", "SHORT"}

    if decision["trade_allowed"] != expected_trade_allowed:
        raise ValueError(
            "trade_allowed does not match decision"
        )

    return True
