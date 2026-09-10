from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


DECISION_RECORD_FIELDS = (
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
)


def build_live_decision_record(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a normalized, immutable-style record of one live decision.

    The function only records values already present in the supplied
    snapshot. It does not recalculate signals, risk levels, or stability.
    Missing optional values remain None.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    timestamp = snapshot.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    signal = snapshot.get("signal")
    signal_label = snapshot.get("signal_label")

    if signal_label is None and isinstance(signal, str):
        signal_label = signal

    return {
        "timestamp": timestamp,
        "symbol": snapshot.get("symbol"),
        "interval": snapshot.get("interval"),
        "signal": signal,
        "signal_label": signal_label,
        "trend": snapshot.get("trend"),
        "strategy": snapshot.get("strategy"),
        "entry_price": snapshot.get("entry_price"),
        "stop_loss": snapshot.get("stop_loss"),
        "take_profit": snapshot.get("take_profit"),
        "risk_reward_ratio": snapshot.get("risk_reward_ratio"),
        "stability_score": snapshot.get("stability_score"),
        "market_state": snapshot.get(
            "market_state",
            snapshot.get("marketState"),
        ),
        "quote_age_seconds": snapshot.get("quote_age_seconds"),
        "quote_stale": snapshot.get("quote_stale"),
        "candle_count": snapshot.get("candle_count"),
    }


def validate_live_decision_record(
    record: Mapping[str, Any],
) -> bool:
    """
    Validate the structural integrity of a live decision record.

    Required fields must exist and core identity fields must not be empty.
    Trading values are not recalculated or inferred.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    missing = [
        field
        for field in DECISION_RECORD_FIELDS
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"record is missing required fields: {', '.join(missing)}"
        )

    for field in ("timestamp", "symbol", "interval"):
        value = record.get(field)
        if value is None or value == "":
            raise ValueError(
                f"record field '{field}' must not be empty"
            )

    return True


def record_live_decision(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build and validate a live decision record in one operation.
    """
    record = build_live_decision_record(snapshot)
    validate_live_decision_record(record)
    return record
