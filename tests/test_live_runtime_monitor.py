from src.evaluation.live_runtime_history import create_live_runtime_history
from src.evaluation.live_runtime_history_health import (
    build_live_runtime_history_health,
)
from src.evaluation.live_runtime_monitor import (
    build_live_runtime_monitor,
    is_live_runtime_monitor_healthy,
    live_runtime_monitor_message,
)
from src.evaluation.live_runtime_snapshot import LiveRuntimeSnapshot


def _snapshot(
    timestamp: str,
    *,
    strategy: str = "momentum",
    decision: str = "BUY",
    trend: str = "UP",
    controller_ready: bool = True,
    session_ready: bool = True,
) -> LiveRuntimeSnapshot:
    return LiveRuntimeSnapshot(
        timestamp=timestamp,
        symbol="XAUUSD",
        interval="5m",
        strategy=strategy,
        stability_score=0.8,
        controller_ready=controller_ready,
        session_ready=session_ready,
        decision=decision,
        trend=trend,
        entry=3000.0,
        stop_loss=2990.0,
        take_profit_1=3010.0,
        take_profit_2=3020.0,
        take_profit_3=3030.0,
        failed_gates=(),
        failed_checks=(),
    )


def test_empty_history_monitor() -> None:
    history = create_live_runtime_history()

    monitor = build_live_runtime_monitor(history)

    assert monitor.healthy is False
    assert monitor.status == "EMPTY"
    assert monitor.health_score == 0.0
    assert monitor.snapshot_count == 0
    assert monitor.ready_ratio == 0.0
    assert monitor.latest_strategy is None
    assert monitor.latest_decision is None
    assert monitor.latest_trend is None
    assert monitor.strategy_changes == 0
    assert monitor.issues


def test_healthy_monitor_reflects_latest_state() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                strategy="momentum",
                decision="BUY",
                trend="UP",
            ),
            _snapshot(
                "2026-01-01T00:05:00+00:00",
                strategy="breakout",
                decision="SELL",
                trend="DOWN",
            ),
        ]
    )

    monitor = build_live_runtime_monitor(history)

    assert monitor.healthy is True
    assert monitor.status == "HEALTHY"
    assert monitor.health_score == 1.0
    assert monitor.snapshot_count == 2
    assert monitor.ready_ratio == 1.0
    assert monitor.latest_strategy == "breakout"
    assert monitor.latest_decision == "SELL"
    assert monitor.latest_trend == "DOWN"
    assert monitor.strategy_changes == 1
    assert monitor.issues == ()


def test_monitor_detects_degraded_history() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "",
                controller_ready=False,
                session_ready=False,
            ),
            _snapshot(
                "invalid",
            ),
        ]
    )

    health = build_live_runtime_history_health(history)
    monitor = build_live_runtime_monitor(history)

    assert health.status == "DEGRADED"
    assert monitor.healthy is False
    assert monitor.status == "DEGRADED"
    assert monitor.health_score == health.score
    assert monitor.snapshot_count == 2
    assert monitor.ready_ratio == 0.5
    assert monitor.issues


def test_monitor_health_helper() -> None:
    healthy_history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    empty_history = create_live_runtime_history()

    assert is_live_runtime_monitor_healthy(healthy_history) is True
    assert is_live_runtime_monitor_healthy(empty_history) is False


def test_monitor_message_contains_runtime_state() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                strategy="momentum",
                decision="BUY",
                trend="UP",
            )
        ]
    )

    message = live_runtime_monitor_message(history)

    assert "LIVE RUNTIME MONITOR" in message
    assert "status=HEALTHY" in message
    assert "snapshots=1" in message
    assert "strategy=momentum" in message
    assert "decision=BUY" in message
    assert "trend=UP" in message
    assert "health_score=1.00" in message


def test_empty_monitor_message() -> None:
    history = create_live_runtime_history()

    assert (
        live_runtime_monitor_message(history)
        == "LIVE RUNTIME MONITOR: EMPTY"
    )
