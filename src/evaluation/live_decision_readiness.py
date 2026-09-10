from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_HANDOFF_FIELDS = (
    "timestamp",
    "symbol",
    "interval",
    "signal",
    "signal_label",
    "trend",
    "strategy",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward_ratio",
    "stability_score",
    "market_state",
    "quote_age_seconds",
    "quote_stale",
    "candle_count",
    "health_status",
    "healthy",
    "audit_passed",
    "record_count",
    "persisted",
    "store_path",
    "actionable",
)


def evaluate_live_decision_readiness(
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Evaluate whether a recorded live decision is ready for the
    next execution-facing layer.

    This function does not execute orders and does not recalculate
    signals, risk levels, stability, or performance.
    """
    if not isinstance(handoff, Mapping):
        raise TypeError("handoff must be a mapping")

    missing = [
        field
        for field in REQUIRED_HANDOFF_FIELDS
        if field not in handoff
    ]

    if missing:
        raise ValueError(
            "handoff is missing required fields: "
            + ", ".join(missing)
        )

    signal_label = handoff.get("signal_label")
    actionable = handoff.get("actionable")
    healthy = handoff.get("healthy")
    audit_passed = handoff.get("audit_passed")
    quote_stale = handoff.get("quote_stale")
    market_state = handoff.get("market_state")

    checks = {
        "has_symbol": bool(handoff.get("symbol")),
        "has_timestamp": bool(handoff.get("timestamp")),
        "has_signal": bool(signal_label),
        "has_entry_price": handoff.get("entry_price") is not None,
        "has_stop_loss": handoff.get("stop_loss") is not None,
        "has_take_profit": handoff.get("take_profit") is not None,
        "healthy": healthy is True,
        "audit_passed": audit_passed is True,
        "quote_fresh": quote_stale is False,
        "market_open": str(market_state).lower() == "open",
        "actionable": actionable is True,
    }

    ready = all(checks.values())

    if ready:
        status = "READY"
    elif not checks["healthy"] or not checks["audit_passed"]:
        status = "BLOCKED_HEALTH"
    elif checks["actionable"] and not checks["quote_fresh"]:
        status = "BLOCKED_STALE_QUOTE"
    elif checks["actionable"] and not checks["market_open"]:
        status = "BLOCKED_MARKET"
    elif not checks["actionable"]:
        status = "NO_ACTION"
    else:
        status = "INCOMPLETE"

    return {
        "ready": ready,
        "status": status,
        "symbol": handoff.get("symbol"),
        "signal_label": signal_label,
        "trend": handoff.get("trend"),
        "strategy": handoff.get("strategy"),
        "entry_price": handoff.get("entry_price"),
        "stop_loss": handoff.get("stop_loss"),
        "take_profit": handoff.get("take_profit"),
        "risk_reward_ratio": handoff.get(
            "risk_reward_ratio"
        ),
        "stability_score": handoff.get(
            "stability_score"
        ),
        "quote_stale": quote_stale,
        "market_state": market_state,
        "checks": checks,
    }


def validate_live_decision_readiness(
    readiness: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live decision readiness result.
    """
    if not isinstance(readiness, Mapping):
        raise TypeError("readiness must be a mapping")

    required_fields = (
        "ready",
        "status",
        "symbol",
        "signal_label",
        "trend",
        "strategy",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "stability_score",
        "quote_stale",
        "market_state",
        "checks",
    )

    missing = [
        field
        for field in required_fields
        if field not in readiness
    ]

    if missing:
        raise ValueError(
            "readiness is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(readiness["ready"], bool):
        raise ValueError("ready must be boolean")

    if not isinstance(readiness["status"], str):
        raise ValueError("status must be a string")

    if not isinstance(readiness["checks"], Mapping):
        raise ValueError("checks must be a mapping")

    return True
