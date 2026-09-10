from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.live_runtime_history import LiveRuntimeHistory
from src.evaluation.live_runtime_history_health import (
    build_live_runtime_history_health,
)


@dataclass(frozen=True)
class LiveRuntimeHistorySummary:
    count: int
    latest_timestamp: str | None
    symbol: str | None
    interval: str | None
    latest_strategy: str | None
    latest_decision: str | None
    latest_trend: str | None
    strategy_count: int
    strategy_changes: int
    decision_counts: tuple[tuple[str, int], ...]
    trend_counts: tuple[tuple[str, int], ...]
    controller_ready_count: int
    session_ready_count: int
    ready_ratio: float
    health_status: str
    health_score: float


def _sorted_counts(values: list[str | None]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}

    for value in values:
        if value is None:
            continue

        normalized = str(value).strip()

        if not normalized:
            continue

        counts[normalized] = counts.get(normalized, 0) + 1

    return tuple(sorted(counts.items()))


def build_live_runtime_history_summary(
    history: LiveRuntimeHistory,
) -> LiveRuntimeHistorySummary:
    if not isinstance(history, LiveRuntimeHistory):
        raise TypeError("history must be a LiveRuntimeHistory.")

    health = build_live_runtime_history_health(history)

    if history.count == 0:
        return LiveRuntimeHistorySummary(
            count=0,
            latest_timestamp=None,
            symbol=None,
            interval=None,
            latest_strategy=None,
            latest_decision=None,
            latest_trend=None,
            strategy_count=0,
            strategy_changes=0,
            decision_counts=(),
            trend_counts=(),
            controller_ready_count=0,
            session_ready_count=0,
            ready_ratio=0.0,
            health_status=health.status,
            health_score=health.score,
        )

    snapshots = history.snapshots
    latest = history.latest

    strategies = [
        snapshot.strategy
        for snapshot in snapshots
        if snapshot.strategy is not None
        and str(snapshot.strategy).strip()
    ]

    controller_ready_count = sum(
        1
        for snapshot in snapshots
        if snapshot.controller_ready
    )

    session_ready_count = sum(
        1
        for snapshot in snapshots
        if snapshot.session_ready
    )

    return LiveRuntimeHistorySummary(
        count=history.count,
        latest_timestamp=(
            str(latest.timestamp)
            if latest is not None
            else None
        ),
        symbol=(
            latest.symbol
            if latest is not None
            else None
        ),
        interval=(
            latest.interval
            if latest is not None
            else None
        ),
        latest_strategy=(
            latest.strategy
            if latest is not None
            else None
        ),
        latest_decision=(
            latest.decision
            if latest is not None
            else None
        ),
        latest_trend=(
            latest.trend
            if latest is not None
            else None
        ),
        strategy_count=len(set(strategies)),
        strategy_changes=health.strategy_changes,
        decision_counts=_sorted_counts(
            [snapshot.decision for snapshot in snapshots]
        ),
        trend_counts=_sorted_counts(
            [snapshot.trend for snapshot in snapshots]
        ),
        controller_ready_count=controller_ready_count,
        session_ready_count=session_ready_count,
        ready_ratio=health.ready_ratio,
        health_status=health.status,
        health_score=health.score,
    )


def live_runtime_history_summary_message(
    history: LiveRuntimeHistory,
) -> str:
    summary = build_live_runtime_history_summary(history)

    if summary.count == 0:
        return "LIVE RUNTIME HISTORY SUMMARY: EMPTY"

    return (
        "LIVE RUNTIME HISTORY SUMMARY: "
        f"{summary.symbol} "
        f"{summary.interval}; "
        f"snapshots={summary.count}; "
        f"strategy={summary.latest_strategy}; "
        f"decision={summary.latest_decision}; "
        f"ready_ratio={summary.ready_ratio:.2f}; "
        f"health={summary.health_status}; "
        f"health_score={summary.health_score:.2f}"
    )


def live_runtime_history_summary_dict(
    history: LiveRuntimeHistory,
) -> dict[str, Any]:
    summary = build_live_runtime_history_summary(history)

    return {
        "count": summary.count,
        "latest_timestamp": summary.latest_timestamp,
        "symbol": summary.symbol,
        "interval": summary.interval,
        "latest_strategy": summary.latest_strategy,
        "latest_decision": summary.latest_decision,
        "latest_trend": summary.latest_trend,
        "strategy_count": summary.strategy_count,
        "strategy_changes": summary.strategy_changes,
        "decision_counts": dict(summary.decision_counts),
        "trend_counts": dict(summary.trend_counts),
        "controller_ready_count": summary.controller_ready_count,
        "session_ready_count": summary.session_ready_count,
        "ready_ratio": summary.ready_ratio,
        "health_status": summary.health_status,
        "health_score": summary.health_score,
    }
