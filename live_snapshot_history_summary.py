"""Summarize persisted live trading snapshot history."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from live_snapshot_history import (
    DEFAULT_HISTORY_PATH,
    load_live_snapshot_history,
)


def summarize_live_snapshot_history(
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> dict[str, Any]:
    """Return a compact statistical summary of live snapshot history."""
    records = load_live_snapshot_history(path)

    if not records:
        return {
            "records": 0,
            "buy_count": 0,
            "no_trade_count": 0,
            "other_signal_count": 0,
            "buy_ratio": 0.0,
            "trend_counts": {},
            "symbol_counts": {},
            "interval_counts": {},
            "first_timestamp": None,
            "last_timestamp": None,
        }

    signal_counts = Counter(record.get("signal") for record in records)
    trend_counts = Counter(
        str(record.get("trend"))
        for record in records
        if record.get("trend") is not None
    )
    symbol_counts = Counter(
        str(record.get("symbol"))
        for record in records
        if record.get("symbol") is not None
    )
    interval_counts = Counter(
        str(record.get("interval"))
        for record in records
        if record.get("interval") is not None
    )

    buy_count = signal_counts.get(1, 0)
    no_trade_count = signal_counts.get(0, 0)
    other_signal_count = len(records) - buy_count - no_trade_count

    timestamps = [
        str(record["timestamp"])
        for record in records
        if record.get("timestamp") is not None
    ]

    return {
        "records": len(records),
        "buy_count": buy_count,
        "no_trade_count": no_trade_count,
        "other_signal_count": other_signal_count,
        "buy_ratio": buy_count / len(records),
        "trend_counts": dict(trend_counts),
        "symbol_counts": dict(symbol_counts),
        "interval_counts": dict(interval_counts),
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
    }
