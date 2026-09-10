from src.evaluation.live_runtime_history import (
    create_live_runtime_history,
)
from src.evaluation.live_runtime_history_health import (
    build_live_runtime_history_health,
)
from src.evaluation.live_runtime_history_summary import (
    build_live_runtime_history_summary,
    live_runtime_history_summary_dict,
    live_runtime_history_summary_message,
)
from src.evaluation.live_runtime_snapshot import LiveRuntimeSnapshot


def _snapshot(
    timestamp: str,
    *,
    strategy: str | None = "momentum",
    decision: str | None = "BUY",
    trend: str | None = "UP",
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


def test_empty_history_summary() -> None:
    history = create_live_runtime_history()

    summary = build_live_runtime_history_summary(history)

    assert summary.count == 0
    assert summary.latest_timestamp is None
    assert summary.strategy_count == 0
    assert summary.decision_counts == ()
    assert summary.ready_ratio == 0.0
    assert summary.health_status == "EMPTY"
    assert summary.health_score == 0.0

    assert (
        live_runtime_history_summary_message(history)
        == "LIVE RUNTIME HISTORY SUMMARY: EMPTY"
    )


def test_summary_tracks_latest_runtime_state() -> None:
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
                session_ready=False,
            ),
        ]
    )

    summary = build_live_runtime_history_summary(history)

    assert summary.count == 2
    assert summary.latest_timestamp == "2026-01-01T00:05:00+00:00"
    assert summary.symbol == "XAUUSD"
    assert summary.interval == "5m"
    assert summary.latest_strategy == "breakout"
    assert summary.latest_decision == "SELL"
    assert summary.latest_trend == "DOWN"
    assert summary.strategy_count == 2
    assert summary.strategy_changes == 1
    assert summary.decision_counts == (
        ("BUY", 1),
        ("SELL", 1),
    )
    assert summary.trend_counts == (
        ("DOWN", 1),
        ("UP", 1),
    )
    assert summary.controller_ready_count == 2
    assert summary.session_ready_count == 1
    assert summary.ready_ratio == 1.0
    assert summary.health_status == "HEALTHY"
    assert summary.health_score == 1.0


def test_summary_uses_history_health() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "",
                controller_ready=False,
                session_ready=False,
            ),
            _snapshot(
                "invalid",
                controller_ready=True,
                session_ready=True,
            ),
        ]
    )

    health = build_live_runtime_history_health(history)
    summary = build_live_runtime_history_summary(history)

    assert health.status == "DEGRADED"
    assert summary.health_status == "DEGRADED"
    assert summary.health_score == health.score
    assert summary.ready_ratio == 0.5


def test_summary_dict_is_serializable() -> None:
    history = create_live_runtime_history(
        [
            _snapshot(
                "2026-01-01T00:00:00+00:00",
                decision="BUY",
                trend="UP",
            ),
            _snapshot(
                "2026-01-01T00:05:00+00:00",
                decision="BUY",
                trend="UP",
            ),
        ]
    )

    result = live_runtime_history_summary_dict(history)

    assert result["count"] == 2
    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["latest_decision"] == "BUY"
    assert result["decision_counts"] == {"BUY": 2}
    assert result["trend_counts"] == {"UP": 2}
    assert result["strategy_count"] == 1
    assert result["health_status"] == "HEALTHY"


def test_summary_rejects_invalid_history() -> None:
    try:
        build_live_runtime_history_summary(None)  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "history must be a LiveRuntimeHistory."
    else:
        raise AssertionError("TypeError was not raised.")
