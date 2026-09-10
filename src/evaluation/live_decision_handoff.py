from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_live_decision_handoff(
    pipeline_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a production-facing handoff from a processed live decision.

    This function only exposes already-recorded decision and evaluation
    values. It does not recalculate signals, risk levels, stability,
    performance, or execute orders.
    """
    if not isinstance(pipeline_result, Mapping):
        raise TypeError("pipeline_result must be a mapping")

    required_fields = (
        "record",
        "history",
        "evaluation",
        "persisted",
        "store_path",
    )

    missing = [
        field
        for field in required_fields
        if field not in pipeline_result
    ]

    if missing:
        raise ValueError(
            "pipeline_result is missing required fields: "
            + ", ".join(missing)
        )

    record = pipeline_result["record"]
    evaluation = pipeline_result["evaluation"]

    if not isinstance(record, Mapping):
        raise ValueError("pipeline_result record must be a mapping")

    if not isinstance(evaluation, Mapping):
        raise ValueError(
            "pipeline_result evaluation must be a mapping"
        )

    signal_label = record.get("signal_label")

    return {
        "timestamp": record.get("timestamp"),
        "symbol": record.get("symbol"),
        "interval": record.get("interval"),
        "signal": record.get("signal"),
        "signal_label": signal_label,
        "trend": record.get("trend"),
        "strategy": record.get("strategy"),
        "entry_price": record.get("entry_price"),
        "stop_loss": record.get("stop_loss"),
        "take_profit": record.get("take_profit"),
        "risk_reward_ratio": record.get("risk_reward_ratio"),
        "stability_score": record.get("stability_score"),
        "market_state": record.get("market_state"),
        "quote_age_seconds": record.get("quote_age_seconds"),
        "quote_stale": record.get("quote_stale"),
        "candle_count": record.get("candle_count"),
        "health_status": evaluation.get("health_status"),
        "healthy": evaluation.get("healthy"),
        "audit_passed": evaluation.get("audit_passed"),
        "record_count": evaluation.get("record_count"),
        "persisted": pipeline_result.get("persisted"),
        "store_path": pipeline_result.get("store_path"),
        "actionable": signal_label in {"BUY", "SELL"},
    }


def validate_live_decision_handoff(
    handoff: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live decision handoff.
    """
    if not isinstance(handoff, Mapping):
        raise TypeError("handoff must be a mapping")

    required_fields = (
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

    missing = [
        field
        for field in required_fields
        if field not in handoff
    ]

    if missing:
        raise ValueError(
            "handoff is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(handoff["actionable"], bool):
        raise ValueError("actionable must be boolean")

    if not isinstance(handoff["healthy"], bool):
        raise ValueError("healthy must be boolean")

    if not isinstance(handoff["audit_passed"], bool):
        raise ValueError("audit_passed must be boolean")

    if not isinstance(handoff["persisted"], bool):
        raise ValueError("persisted must be boolean")

    if handoff["store_path"] is not None and not isinstance(
        handoff["store_path"],
        str,
    ):
        raise ValueError(
            "store_path must be a string or None"
        )

    return True
