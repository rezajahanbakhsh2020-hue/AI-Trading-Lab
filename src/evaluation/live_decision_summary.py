from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def summarize_live_decisions(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a compact summary of recorded live decisions.

    This function summarizes recorded values only. It does not
    recalculate trading signals, risk levels, or performance.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    signal_counts: dict[str, int] = {}
    trend_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}

    for record in records:
        signal = str(
            record.get("signal_label")
            or record.get("signal")
            or "UNKNOWN"
        )
        trend = str(record.get("trend") or "UNKNOWN")
        strategy = str(record.get("strategy") or "UNKNOWN")

        signal_counts[signal] = signal_counts.get(signal, 0) + 1
        trend_counts[trend] = trend_counts.get(trend, 0) + 1
        strategy_counts[strategy] = (
            strategy_counts.get(strategy, 0) + 1
        )

    latest = records[-1] if records else None

    return {
        "record_count": len(records),
        "signal_counts": signal_counts,
        "trend_counts": trend_counts,
        "strategy_counts": strategy_counts,
        "latest": dict(latest) if latest is not None else None,
    }


def get_live_decision_counts(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """
    Return counts grouped by signal label.
    """
    summary = summarize_live_decisions(history)
    return dict(summary["signal_counts"])


def get_latest_live_decision_summary(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """
    Return a copy of the latest recorded decision.
    """
    summary = summarize_live_decisions(history)

    latest = summary["latest"]

    if latest is None:
        return None

    return dict(latest)
