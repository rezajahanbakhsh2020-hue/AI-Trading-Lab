from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.live_runtime_history import LiveRuntimeHistory
from src.evaluation.live_runtime_monitor import build_live_runtime_monitor


@dataclass(frozen=True)
class LiveRuntimeAlert:
    level: str
    code: str
    message: str


def build_live_runtime_alerts(
    history: LiveRuntimeHistory,
) -> tuple[LiveRuntimeAlert, ...]:
    monitor = build_live_runtime_monitor(history)
    alerts: list[LiveRuntimeAlert] = []

    if monitor.snapshot_count == 0:
        alerts.append(
            LiveRuntimeAlert(
                level="WARNING",
                code="EMPTY_HISTORY",
                message="No live runtime snapshots are available.",
            )
        )
        return tuple(alerts)

    if not monitor.healthy:
        alerts.append(
            LiveRuntimeAlert(
                level="WARNING",
                code="UNHEALTHY_HISTORY",
                message=(
                    "Live runtime history is unhealthy: "
                    f"status={monitor.status}, "
                    f"health_score={monitor.health_score:.2f}."
                ),
            )
        )

    if monitor.ready_ratio < 1.0:
        alerts.append(
            LiveRuntimeAlert(
                level="WARNING",
                code="READINESS_GAP",
                message=(
                    "Live runtime readiness is below 100%: "
                    f"ready_ratio={monitor.ready_ratio:.2f}."
                ),
            )
        )

    if monitor.strategy_changes > 0:
        alerts.append(
            LiveRuntimeAlert(
                level="INFO",
                code="STRATEGY_CHANGED",
                message=(
                    "Strategy changes detected in live runtime history: "
                    f"{monitor.strategy_changes}."
                ),
            )
        )

    return tuple(alerts)


def has_live_runtime_alerts(
    history: LiveRuntimeHistory,
) -> bool:
    return bool(build_live_runtime_alerts(history))


def live_runtime_alert_level(
    history: LiveRuntimeHistory,
) -> str:
    alerts = build_live_runtime_alerts(history)

    if not alerts:
        return "OK"

    levels = {alert.level for alert in alerts}

    if "WARNING" in levels:
        return "WARNING"

    if "ERROR" in levels:
        return "ERROR"

    return "INFO"
