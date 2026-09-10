from __future__ import annotations

import pytest

from src.evaluation.live_decision_monitor import (
    monitor_live_decision_history,
    validate_live_decision_monitor,
)


def _record(
    timestamp: str,
    signal_label: str = "BUY",
    trend: str = "UP",
    strategy: str = "momentum",
    stale: bool = False,
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
        "trend": trend,
        "strategy": strategy,
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": stale,
        "candle_count": 201,
    }


def test_monitor_reports_healthy_history():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
        ),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    result = monitor_live_decision_history(history)

    assert result["status"] == "HEALTHY"
    assert result["healthy"] is True
    assert result["record_count"] == 3


def test_monitor_counts_signals():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
        ),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    result = monitor_live_decision_history(history)

    assert result["buy_count"] == 2
    assert result["no_trade_count"] == 1


def test_monitor_counts_stale_quotes():
    history = [
        _record(
            "2026-09-10T05:45:00+00:00",
            stale=True,
        ),
        _record(
            "2026-09-10T05:50:00+00:00",
            stale=False,
        ),
        _record(
            "2026-09-10T05:55:00+00:00",
            stale=True,
        ),
    ]

    result = monitor_live_decision_history(history)

    assert result["stale_quote_count"] == 2


def test_monitor_reports_latest_decision():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
            "mean_reversion",
        ),
    ]

    result = monitor_live_decision_history(history)

    assert result["latest_timestamp"] == (
        "2026-09-10T05:50:00+00:00"
    )
    assert result["latest_signal"] == "NO TRADE"
    assert result["latest_trend"] == "FLAT"
    assert result["latest_strategy"] == "mean_reversion"


def test_monitor_empty_history():
    result = monitor_live_decision_history([])

    assert result["status"] == "EMPTY"
    assert result["healthy"] is False
    assert result["record_count"] == 0
    assert result["buy_count"] == 0
    assert result["no_trade_count"] == 0
    assert result["stale_quote_count"] == 0
    assert result["latest_timestamp"] is None
    assert result["latest_signal"] is None


def test_monitor_detects_unhealthy_history():
    history = [
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = monitor_live_decision_history(history)

    assert result["status"] == "UNHEALTHY"
    assert result["healthy"] is False


def test_monitor_does_not_recalculate_values():
    record = _record("2026-09-10T05:45:00+00:00")

    record["entry_price"] = 9999.0
    record["stop_loss"] = 9000.0
    record["take_profit"] = 12000.0
    record["stability_score"] = 0.11

    result = monitor_live_decision_history([record])

    assert result["status"] == "INITIALIZING"
    assert record["entry_price"] == 9999.0
    assert record["stop_loss"] == 9000.0
    assert record["take_profit"] == 12000.0
    assert record["stability_score"] == 0.11


def test_invalid_history_input_is_rejected():
    with pytest.raises(TypeError):
        monitor_live_decision_history("invalid")


def test_invalid_history_record_is_rejected():
    with pytest.raises(TypeError):
        monitor_live_decision_history([[]])


def test_validation_accepts_valid_monitor():
    result = monitor_live_decision_history(
        [_record("2026-09-10T05:45:00+00:00")]
    )

    assert validate_live_decision_monitor(result) is True


def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_monitor(
            {
                "status": "HEALTHY",
            }
        )


def test_validation_rejects_wrong_status_type():
    with pytest.raises(ValueError):
        validate_live_decision_monitor(
            {
                "status": 1,
                "healthy": True,
                "record_count": 1,
                "buy_count": 1,
                "no_trade_count": 0,
                "stale_quote_count": 0,
                "latest_timestamp": "2026-09-10T05:45:00+00:00",
                "latest_signal": "BUY",
                "latest_trend": "UP",
                "latest_strategy": "momentum",
                "health": {},
            }
        )


def test_validation_rejects_wrong_count_type():
    with pytest.raises(ValueError):
        validate_live_decision_monitor(
            {
                "status": "HEALTHY",
                "healthy": True,
                "record_count": "1",
                "buy_count": 1,
                "no_trade_count": 0,
                "stale_quote_count": 0,
                "latest_timestamp": "2026-09-10T05:45:00+00:00",
                "latest_signal": "BUY",
                "latest_trend": "UP",
                "latest_strategy": "momentum",
                "health": {},
            }
        )
