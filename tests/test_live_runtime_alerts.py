from src.evaluation.live_runtime_alerts import (
    build_live_runtime_alerts,
    has_live_runtime_alerts,
    live_runtime_alert_level,
)
from src.evaluation.live_runtime_history import create_live_runtime_history
from src.evaluation.live_runtime_snapshot import LiveRuntimeSnapshot


def _snapshot(
    timestamp: str,
    *,
    strategy: str = "momentum",
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
        decision="BUY",
        trend="UP",
        entry=3000.0,
        stop_loss=2990.0,
        take_profit_1=3010.0,
        take_profit_2=3020.0,
        take_profit_3=3030.0,
        failed_gates=(),
        failed_checks=(),
    )


def test_empty_history_creates_warning() -> None:
    history = create_live_runtime_history()

    alerts = build_live_runtime_alerts(history)

    assert len(alerts) == 1
    assert alerts[0].level == "WARNING"
    assert alerts[0].code == "EMPTY_HISTORY"
    assert has_live_runtime_alerts(history) is True
    assert live_runtime_alert_level(history) == "WARNING"


def test_healthy_history_has_no_alerts() -> None:
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    alerts = build_live_runtime_alerts(history)

    assert alerts == ()
    assert has_live_runtime_alerts(history) is False
    assert live_runtime_alert_level(history) == "OK"


def test_unhealthy_history_creates_warning() -> None:
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

    alerts = build_live_runtime_alerts(history)

    codes = {alert.code for alert in alerts}

    assert "UNHEALTHY_HISTORY" in codes
    assert "READINESS_GAP" in codes
    assert live_runtime_alert_level(history) == "WARNING"


def test_strategy_change_creates_info_alert() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                strategy="momentum",
            ),
            _snapshot(
                "2026-01-01T00:05:00+00:00",
                strategy="breakout",
            ),
        ]
    )

    alerts = build_live_runtime_alerts(history)

    assert len(alerts) == 1
    assert alerts[0].level == "INFO"
    assert alerts[0].code == "STRATEGY_CHANGED"
    assert live_runtime_alert_level(history) == "INFO"


def test_readiness_gap_alert() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                controller_ready=False,
            ),
            _snapshot("2026-01-01T00:05:00+00:00"),
        ]
    )

    alerts = build_live_runtime_alerts(history)

    assert any(
        alert.code == "READINESS_GAP"
        for alert in alerts
    )
    assert live_runtime_alert_level(history) == "WARNING"
