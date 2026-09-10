from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.live_decision_health import (
    calculate_live_decision_health,
)


def monitor_live_decision_history(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Monitor the operational state of live decision history.

    This function reports health and basic activity information only.
    It does not recalculate trading signals, risk levels, or performance.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    health = calculate_live_decision_health(records)

    latest = records[-1] if records else None

    buy_count = sum(
        1
        for record in records
        if record.get("signal_label") == "BUY"
    )

    no_trade_count = sum(
        1
        for record in records
        if record.get("signal_label") == "NO TRADE"
    )

    stale_count = sum(
        1
        for record in records
        if record.get("quote_stale") is True
    )

    return {
        "status": health["status"],
        "healthy": health["healthy"],
        "record_count": health["record_count"],
        "buy_count": buy_count,
        "no_trade_count": no_trade_count,
        "stale_quote_count": stale_count,
        "latest_timestamp": (
            latest.get("timestamp")
            if latest is not None
            else None
        ),
        "latest_signal": (
            latest.get("signal_label")
            if latest is not None
            else None
        ),
        "latest_trend": (
            latest.get("trend")
            if latest is not None
            else None
        ),
        "latest_strategy": (
            latest.get("strategy")
            if latest is not None
            else None
        ),
        "health": health,
    }


def validate_live_decision_monitor(
    monitor: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live decision monitor result.
    """
    if not isinstance(monitor, Mapping):
        raise TypeError("monitor must be a mapping")

    required_fields = (
        "status",
        "healthy",
        "record_count",
        "buy_count",
        "no_trade_count",
        "stale_quote_count",
        "latest_timestamp",
        "latest_signal",
        "latest_trend",
        "latest_strategy",
        "health",
    )

    missing = [
        field
        for field in required_fields
        if field not in monitor
    ]

    if missing:
        raise ValueError(
            "monitor is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(monitor["status"], str):
        raise ValueError("status must be a string")

    if not isinstance(monitor["healthy"], bool):
        raise ValueError("healthy must be boolean")

    for field in (
        "record_count",
        "buy_count",
        "no_trade_count",
        "stale_quote_count",
    ):
        if not isinstance(monitor[field], int):
            raise ValueError(
                f"{field} must be an integer"
            )

    if not isinstance(monitor["health"], Mapping):
        raise ValueError("health must be a mapping")

    return True
