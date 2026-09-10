from src.evaluation.live_runtime_history import create_live_runtime_history
from src.evaluation.live_runtime_snapshot import LiveRuntimeSnapshot
from src.evaluation.live_runtime_status import (
    build_live_runtime_status,
    is_live_runtime_ready,
    live_runtime_status_message,
)


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


def test_empty_history_status() -> None:
    history = create_live_runtime_history()

    status = build_live_runtime_status(history)

    assert status.state == "EMPTY"
    assert status.healthy is False
    assert status.snapshot_count == 0
    assert status.alert_count == 1
    assert is_live_runtime_ready(history) is False


def test_healthy_history_is_ready() -> None:
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    status = build_live_runtime_status(history)

    assert status.state == "READY"
    assert status.healthy is True
    assert status.health_score == 1.0
    assert status.snapshot_count == 1
    assert status.ready_ratio == 1.0
    assert status.strategy == "momentum"
    assert status.decision == "BUY"
    assert status.trend == "UP"
    assert status.alert_count == 0
    assert is_live_runtime_ready(history) is True


def test_degraded_history_is_not_ready() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "",
                controller_ready=False,
                session_ready=False,
            ),
            _snapshot("invalid"),
        ]
    )

    status = build_live_runtime_status(history)

    assert status.state == "DEGRADED"
    assert status.healthy is False
    assert status.snapshot_count == 2
    assert status.ready_ratio == 0.5
    assert status.alert_count >= 1
    assert is_live_runtime_ready(history) is False


def test_status_message_contains_key_state() -> None:
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    message = live_runtime_status_message(history)

    assert "LIVE RUNTIME STATUS" in message
    assert "state=READY" in message
    assert "healthy=True" in message
    assert "snapshots=1" in message
    assert "strategy=momentum" in message
    assert "decision=BUY" in message
    assert "trend=UP" in message
    assert "alerts=0" in message
