from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.live_runtime_history import LiveRuntimeHistory


@dataclass(frozen=True)
class LiveRuntimeHistoryHealth:
    healthy: bool
    status: str
    count: int
    ready_count: int
    blocked_count: int
    ready_ratio: float
    duplicate_timestamps: int
    out_of_order: bool
    missing_timestamps: int
    strategy_changes: int
    score: float
    issues: tuple[str, ...]


def build_live_runtime_history_health(
    history: LiveRuntimeHistory,
) -> LiveRuntimeHistoryHealth:
    if not isinstance(history, LiveRuntimeHistory):
        raise TypeError("history must be a LiveRuntimeHistory.")

    count = history.count

    if count == 0:
        return LiveRuntimeHistoryHealth(
            healthy=False,
            status="EMPTY",
            count=0,
            ready_count=0,
            blocked_count=0,
            ready_ratio=0.0,
            duplicate_timestamps=0,
            out_of_order=False,
            missing_timestamps=0,
            strategy_changes=0,
            score=0.0,
            issues=("history is empty",),
        )

    snapshots = history.snapshots

    ready_count = sum(
        1 for snapshot in snapshots if snapshot.controller_ready
    )
    blocked_count = count - ready_count
    ready_ratio = ready_count / count

    timestamps = [
        str(snapshot.timestamp).strip()
        for snapshot in snapshots
    ]

    missing_timestamps = sum(
        1 for timestamp in timestamps if not timestamp
    )

    parsed_timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    )

    invalid_timestamp_count = sum(
        1
        for timestamp, parsed in zip(
            timestamps,
            parsed_timestamps,
        )
        if timestamp and pd.isna(parsed)
    )

    valid_timestamp_count = count - (
        missing_timestamps + invalid_timestamp_count
    )

    duplicate_timestamps = 0
    out_of_order = False

    if valid_timestamp_count == count:
        duplicate_timestamps = int(
            parsed_timestamps.duplicated(keep=False).sum()
        )
        out_of_order = not parsed_timestamps.is_monotonic_increasing

    strategies = [
        snapshot.strategy
        for snapshot in snapshots
        if snapshot.strategy is not None
    ]

    strategy_changes = sum(
        1
        for previous, current in zip(
            strategies,
            strategies[1:],
        )
        if previous != current
    )

    issues: list[str] = []

    if missing_timestamps > 0:
        issues.append("missing timestamps")

    if invalid_timestamp_count > 0:
        issues.append("invalid timestamps")

    if duplicate_timestamps > 0:
        issues.append("duplicate timestamps")

    if out_of_order:
        issues.append("timestamps out of order")

    score = 1.0

    if missing_timestamps > 0:
        score -= 0.25

    if invalid_timestamp_count > 0:
        score -= 0.25

    if duplicate_timestamps > 0:
        score -= 0.20

    if out_of_order:
        score -= 0.20

    score = max(0.0, min(1.0, score))

    healthy = (
        count > 0
        and missing_timestamps == 0
        and invalid_timestamp_count == 0
        and duplicate_timestamps == 0
        and not out_of_order
    )

    status = "HEALTHY" if healthy else "DEGRADED"

    return LiveRuntimeHistoryHealth(
        healthy=healthy,
        status=status,
        count=count,
        ready_count=ready_count,
        blocked_count=blocked_count,
        ready_ratio=ready_ratio,
        duplicate_timestamps=duplicate_timestamps,
        out_of_order=out_of_order,
        missing_timestamps=(
            missing_timestamps + invalid_timestamp_count
        ),
        strategy_changes=strategy_changes,
        score=score,
        issues=tuple(issues),
    )


def is_live_runtime_history_healthy(
    history: LiveRuntimeHistory,
) -> bool:
    return build_live_runtime_history_health(history).healthy


def live_runtime_history_health_message(
    history: LiveRuntimeHistory,
) -> str:
    health = build_live_runtime_history_health(history)

    if health.count == 0:
        return "LIVE RUNTIME HISTORY HEALTH: EMPTY"

    return (
        "LIVE RUNTIME HISTORY HEALTH: "
        f"{health.status}; "
        f"snapshots={health.count}; "
        f"ready_ratio={health.ready_ratio:.2f}; "
        f"score={health.score:.2f}"
    )


def live_runtime_history_health_dict(
    history: LiveRuntimeHistory,
) -> dict[str, Any]:
    health = build_live_runtime_history_health(history)

    return {
        "healthy": health.healthy,
        "status": health.status,
        "count": health.count,
        "ready_count": health.ready_count,
        "blocked_count": health.blocked_count,
        "ready_ratio": health.ready_ratio,
        "duplicate_timestamps": health.duplicate_timestamps,
        "out_of_order": health.out_of_order,
        "missing_timestamps": health.missing_timestamps,
        "strategy_changes": health.strategy_changes,
        "score": health.score,
        "issues": health.issues,
    }
