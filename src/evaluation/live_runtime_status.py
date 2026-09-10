from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.live_runtime_alerts import build_live_runtime_alerts
from src.evaluation.live_runtime_history import LiveRuntimeHistory
from src.evaluation.live_runtime_monitor import build_live_runtime_monitor


@dataclass(frozen=True)
class LiveRuntimeStatus:
    state: str
    healthy: bool
    health_score: float
    snapshot_count: int
    ready_ratio: float
    strategy: str | None
    decision: str | None
    trend: str | None
    alert_count: int


def build_live_runtime_status(
    history: LiveRuntimeHistory,
) -> LiveRuntimeStatus:
    monitor = build_live_runtime_monitor(history)
    alerts = build_live_runtime_alerts(history)

    if monitor.snapshot_count == 0:
        state = "EMPTY"
    elif not monitor.healthy:
        state = "DEGRADED"
    elif any(alert.level == "WARNING" for alert in alerts):
        state = "WARNING"
    else:
        state = "READY"

    return LiveRuntimeStatus(
        state=state,
        healthy=monitor.healthy,
        health_score=monitor.health_score,
        snapshot_count=monitor.snapshot_count,
        ready_ratio=monitor.ready_ratio,
        strategy=monitor.latest_strategy,
        decision=monitor.latest_decision,
        trend=monitor.latest_trend,
        alert_count=len(alerts),
    )


def is_live_runtime_ready(
    history: LiveRuntimeHistory,
) -> bool:
    return build_live_runtime_status(history).state == "READY"


def live_runtime_status_message(
    history: LiveRuntimeHistory,
) -> str:
    status = build_live_runtime_status(history)

    return (
        "LIVE RUNTIME STATUS: "
        f"state={status.state}; "
        f"healthy={status.healthy}; "
        f"snapshots={status.snapshot_count}; "
        f"ready_ratio={status.ready_ratio:.2f}; "
        f"health_score={status.health_score:.2f}; "
        f"strategy={status.strategy}; "
        f"decision={status.decision}; "
        f"trend={status.trend}; "
        f"alerts={status.alert_count}"
    )
