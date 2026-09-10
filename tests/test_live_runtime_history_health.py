from src.evaluation.live_runtime_history import (
    create_live_runtime_history,
)
from src.evaluation.live_runtime_history_health import (
    LiveRuntimeHistoryHealth,
    build_live_runtime_history_health,
    is_live_runtime_history_healthy,
    live_runtime_history_health_dict,
    live_runtime_history_health_message,
)
from src.evaluation.live_runtime_snapshot import (
    LiveRuntimeSnapshot,
)


def _snapshot(
    timestamp: str = "2025-01-01T00:00:00+00:00",
    ready: bool = True,
    strategy: str | None = "momentum",
) -> LiveRuntimeSnapshot:
    return LiveRuntimeSnapshot(
        timestamp=timestamp,
        symbol="XAUUSD",
        interval="5m",
        strategy=strategy,
        stability_score=0.90,
        controller_ready=ready,
        session_ready=ready,
        decision="BUY" if ready else None,
        trend="UP" if ready else None,
        entry=2605.0 if ready else None,
        stop_loss=2590.0 if ready else None,
        take_profit_1=2620.0 if ready else None,
        take_profit_2=2635.0 if ready else None,
        take_profit_3=2650.0 if ready else None,
        failed_gates=(),
        failed_checks=(),
    )


def test_empty_history_is_not_healthy():
    history = create_live_runtime_history()

    health = build_live_runtime_history_health(
        history
    )

    assert isinstance(
        health,
        LiveRuntimeHistoryHealth,
    )
    assert health.healthy is False
    assert health.status == "EMPTY"
    assert health.count == 0
    assert health.score == 0.0


def test_valid_history_is_healthy():
    history = create_live_runtime_history(
        [
            _snapshot(
                "2025-01-01T00:00:00+00:00"
            ),
            _snapshot(
                "2025-01-01T00:05:00+00:00"
            ),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.healthy is True
    assert health.status == "HEALTHY"
    assert health.count == 2
    assert health.ready_count == 2
    assert health.blocked_count == 0
    assert health.ready_ratio == 1.0
    assert health.duplicate_timestamps == 0
    assert health.out_of_order is False
    assert health.score == 1.0


def test_blocked_snapshots_are_counted():
    history = create_live_runtime_history(
        [
            _snapshot(ready=True),
            _snapshot(
                timestamp="2025-01-01T00:05:00+00:00",
                ready=False,
            ),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.count == 2
    assert health.ready_count == 1
    assert health.blocked_count == 1
    assert health.ready_ratio == 0.5
    assert health.healthy is True


def test_duplicate_timestamps_degrade_health():
    timestamp = "2025-01-01T00:00:00+00:00"

    history = create_live_runtime_history(
        [
            _snapshot(timestamp),
            _snapshot(timestamp),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.healthy is False
    assert health.status == "DEGRADED"
    assert health.duplicate_timestamps == 2
    assert "duplicate timestamps" in health.issues
    assert health.score == 0.8


def test_out_of_order_timestamps_degrade_health():
    history = create_live_runtime_history(
        [
            _snapshot(
                "2025-01-01T00:05:00+00:00"
            ),
            _snapshot(
                "2025-01-01T00:00:00+00:00"
            ),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.healthy is False
    assert health.out_of_order is True
    assert "timestamps out of order" in health.issues
    assert health.score == 0.8


def test_invalid_timestamp_degrades_health():
    history = create_live_runtime_history(
        [
            _snapshot("not-a-timestamp"),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.healthy is False
    assert health.status == "DEGRADED"
    assert health.missing_timestamps == 1
    assert "invalid timestamps" in health.issues
    assert health.score == 0.75


def test_missing_timestamp_degrades_health():
    history = create_live_runtime_history(
        [
            _snapshot(""),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.healthy is False
    assert health.missing_timestamps == 1
    assert "missing timestamps" in health.issues


def test_strategy_changes_are_tracked():
    history = create_live_runtime_history(
        [
            _snapshot(
                "2025-01-01T00:00:00+00:00",
                strategy="momentum",
            ),
            _snapshot(
                "2025-01-01T00:05:00+00:00",
                strategy="trend",
            ),
            _snapshot(
                "2025-01-01T00:10:00+00:00",
                strategy="trend",
            ),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.strategy_changes == 1
    assert health.healthy is True


def test_none_strategy_is_ignored_for_changes():
    history = create_live_runtime_history(
        [
            _snapshot(
                "2025-01-01T00:00:00+00:00",
                strategy=None,
            ),
            _snapshot(
                "2025-01-01T00:05:00+00:00",
                strategy="momentum",
            ),
        ]
    )

    health = build_live_runtime_history_health(
        history
    )

    assert health.strategy_changes == 0
    assert health.healthy is True


def test_health_boolean_helper():
    healthy_history = create_live_runtime_history(
        [_snapshot()]
    )

    unhealthy_history = create_live_runtime_history(
        [
            _snapshot(
                "invalid"
            )
        ]
    )

    assert (
        is_live_runtime_history_healthy(
            healthy_history
        )
        is True
    )
    assert (
        is_live_runtime_history_healthy(
            unhealthy_history
        )
        is False
    )


def test_health_message():
    history = create_live_runtime_history(
        [_snapshot()]
    )

    message = live_runtime_history_health_message(
        history
    )

    assert message.startswith(
        "LIVE RUNTIME HISTORY HEALTH:"
    )
    assert "HEALTHY" in message
    assert "snapshots=1" in message
    assert "score=1.00" in message


def test_empty_health_message():
    history = create_live_runtime_history()

    assert (
        live_runtime_history_health_message(
            history
        )
        == "LIVE RUNTIME HISTORY HEALTH: EMPTY"
    )


def test_health_dict():
    history = create_live_runtime_history(
        [_snapshot()]
    )

    result = live_runtime_history_health_dict(
        history
    )

    assert result["healthy"] is True
    assert result["status"] == "HEALTHY"
    assert result["count"] == 1
    assert result["ready_count"] == 1
    assert result["score"] == 1.0
    assert result["issues"] == ()


def test_health_rejects_invalid_history():
    try:
        build_live_runtime_history_health({})
    except TypeError as exc:
        assert "LiveRuntimeHistory" in str(exc)
    else:
        raise AssertionError(
            "TypeError was not raised"
        )


def test_health_dict_rejects_invalid_history():
    try:
        live_runtime_history_health_dict({})
    except TypeError as exc:
        assert "LiveRuntimeHistory" in str(exc)
    else:
        raise AssertionError(
            "TypeError was not raised"
        )


def test_message_rejects_invalid_history():
    try:
        live_runtime_history_health_message({})
    except TypeError as exc:
        assert "LiveRuntimeHistory" in str(exc)
    else:
        raise AssertionError(
            "TypeError was not raised"
        )
