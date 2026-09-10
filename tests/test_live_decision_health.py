from __future__ import annotations

import pytest

from src.evaluation.live_decision_health import (
    calculate_live_decision_health,
    validate_live_decision_health,
)


def _record(
    timestamp: str,
    signal_label: str = "BUY",
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_empty_history_is_empty():
    result = calculate_live_decision_health([])

    assert result["status"] == "EMPTY"
    assert result["healthy"] is False
    assert result["record_count"] == 0


def test_single_record_is_initializing():
    result = calculate_live_decision_health(
        [_record("2026-09-10T05:45:00+00:00")]
    )

    assert result["status"] == "INITIALIZING"
    assert result["healthy"] is True
    assert result["record_count"] == 1
    assert result["audit_passed"] is True


def test_valid_history_is_healthy():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    result = calculate_live_decision_health(history)

    assert result["status"] == "HEALTHY"
    assert result["healthy"] is True
    assert result["record_count"] == 3
    assert result["invalid_structure_count"] == 0
    assert result["timeline_issue_count"] == 0
    assert result["audit_passed"] is True


def test_out_of_order_history_is_unhealthy():
    history = [
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = calculate_live_decision_health(history)

    assert result["status"] == "UNHEALTHY"
    assert result["healthy"] is False
    assert result["timeline_issue_count"] == 1
    assert result["audit_passed"] is False


def test_invalid_structure_is_unhealthy():
    record = _record("2026-09-10T05:45:00+00:00")
    del record["strategy"]

    result = calculate_live_decision_health([record])

    assert result["status"] == "UNHEALTHY"
    assert result["healthy"] is False
    assert result["invalid_structure_count"] == 1
    assert result["audit_passed"] is False


def test_changed_transition_is_reported():
    first = _record(
        "2026-09-10T05:45:00+00:00",
        "BUY",
    )
    second = _record(
        "2026-09-10T05:50:00+00:00",
        "NO TRADE",
    )

    result = calculate_live_decision_health(
        [first, second]
    )

    assert result["status"] == "HEALTHY"
    assert result["changed_transition_count"] == 1


def test_health_does_not_recalculate_values():
    record = _record("2026-09-10T05:45:00+00:00")
    record["entry_price"] = 9999.0
    record["stop_loss"] = 9000.0
    record["take_profit"] = 12000.0

    result = calculate_live_decision_health([record])

    assert result["status"] == "INITIALIZING"
    assert record["entry_price"] == 9999.0
    assert record["stop_loss"] == 9000.0
    assert record["take_profit"] == 12000.0


def test_invalid_history_input_is_rejected():
    with pytest.raises(TypeError):
        calculate_live_decision_health("invalid")


def test_invalid_history_record_is_rejected():
    with pytest.raises(TypeError):
        calculate_live_decision_health([[]])


def test_validation_accepts_valid_health_result():
    result = calculate_live_decision_health(
        [_record("2026-09-10T05:45:00+00:00")]
    )

    assert validate_live_decision_health(result) is True


def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_health(
            {
                "status": "HEALTHY",
            }
        )


def test_validation_rejects_wrong_status_type():
    with pytest.raises(ValueError):
        validate_live_decision_health(
            {
                "status": 123,
                "healthy": True,
                "record_count": 1,
                "invalid_structure_count": 0,
                "timeline_issue_count": 0,
                "changed_transition_count": 0,
                "audit_passed": True,
            }
        )


def test_validation_rejects_wrong_count_type():
    with pytest.raises(ValueError):
        validate_live_decision_health(
            {
                "status": "HEALTHY",
                "healthy": True,
                "record_count": "1",
                "invalid_structure_count": 0,
                "timeline_issue_count": 0,
                "changed_transition_count": 0,
                "audit_passed": True,
            }
        )
