from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.live_runtime_history import LiveRuntimeHistory
from src.evaluation.live_runtime_history_health import (
    build_live_runtime_history_health,
)
from src.evaluation.live_runtime_history_summary import (
    build_live_runtime_history_summary,
)


@dataclass(frozen=True)
class LiveRuntimeMonitor:
    healthy: bool
    status: str
    health_score: float
    snapshot_count: int
    ready_ratio: float
    latest_strategy: str | None
    latest_decision: str | None
    latest_trend: str | None
    strategy_changes: int
    issues: tuple[str, ...]


def build_live_runtime_monitor(
    history: LiveRuntimeHistory,
) -> LiveRuntimeMonitor:
    health = build_live_runtime_history_health(history)
    summary = build_live_runtime_history_summary(history)

    return LiveRuntimeMonitor(
        healthy=health.healthy,
        status=health.status,
        health_score=health.score,
        snapshot_count=summary.count,
        ready_ratio=summary.ready_ratio,
        latest_strategy=summary.latest_strategy,
        latest_decision=summary.latest_decision,
        latest_trend=summary.latest_trend,
        strategy_changes=summary.strategy_changes,
        issues=health.issues,
    )


def is_live_runtime_monitor_healthy(
    history: LiveRuntimeHistory,
) -> bool:
    return build_live_runtime_monitor(history).healthy


def live_runtime_monitor_message(
    history: LiveRuntimeHistory,
) -> str:
    monitor = build_live_runtime_monitor(history)

    if monitor.snapshot_count == 0:
        return "LIVE RUNTIME MONITOR: EMPTY"

    return (
        "LIVE RUNTIME MONITOR: "
        f"status={monitor.status}; "
        f"snapshots={monitor.snapshot_count}; "
        f"ready_ratio={monitor.ready_ratio:.2f}; "
        f"health_score={monitor.health_score:.2f}; "
        f"strategy={monitor.latest_strategy}; "
        f"decision={monitor.latest_decision}; "
        f"trend={monitor.latest_trend}; "
        f"strategy_changes={monitor.strategy_changes}"
    )
