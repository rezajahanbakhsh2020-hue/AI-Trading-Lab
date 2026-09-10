from __future__ import annotations

import pytest

from src.evaluation.live_decision_consistency import (
    check_live_decision_consistency,
    validate_live_decision_consistency,
)


def _record(timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
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


def test_consistent_history_is_valid():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    result = check_live_decision_consistency(history)

    assert result["consistent"] is True
    assert result["record_count"] == 3
    assert result["issue_count"] == 0
    assert result["issues"] == []


def test_empty_history_is_consistent():
    result = check_live_decision_consistency([])

    assert result["consistent"] is True
    assert result["record_count"] == 0
    assert result["issue_count"] == 0


def test_out_of_order_timestamp_is_detected():
    history = [
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = check_live_decision_consistency(history)

    assert result["consistent"] is False
    assert result["issue_count"] == 1
    assert "earlier than" in result["issues"][0]


def test_duplicate_consecutive_timestamp_is_detected():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = check_live_decision_consistency(history)

    assert result["consistent"] is False
    assert result["duplicate_consecutive_timestamps"] == 1
    assert result["issue_count"] == 1
    assert "duplicate" in result["issues"][0]


def test_invalid_timestamp_is_detected():
    history = [
        _record("not-a-timestamp"),
    ]

    result = check_live_decision_consistency(history)

    assert result["consistent"] is False
    assert result["issue_count"] == 1
    assert "invalid timestamp" in result["issues"][0]


def test_z_timestamp_is_supported():
    history = [
        _record("2026-09-10T05:45:00Z"),
        _record("2026-09-10T05:50:00Z"),
    ]

    result = check_live_decision_consistency(history)

    assert result["consistent"] is True


def test_non_mapping_history_item_is_rejected():
    with pytest.raises(TypeError):
        check_live_decision_consistency([[]])


def test_invalid_history_input_is_rejected():
    with pytest.raises(TypeError):
        check_live_decision_consistency("invalid")


def test_consistency_does_not_recalculate_values():
    record = _record("2026-09-10T05:45:00+00:00")
    record["entry_price"] = 9999.0
    record["stop_loss"] = 1.0
    record["take_profit"] = 20000.0

    result = check_live_decision_consistency([record])

    assert result["consistent"] is True
    assert record["entry_price"] == 9999.0
    assert record["stop_loss"] == 1.0
    assert record["take_profit"] == 20000.0


def test_validate_consistency_result_accepts_valid_result():
    result = check_live_decision_consistency(
        [_record("2026-09-10T05:45:00+00:00")]
    )

    assert validate_live_decision_consistency(result) is True


def test_validate_consistency_result_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_consistency(
            {
                "consistent": True,
            }
        )


def test_validate_consistency_result_rejects_wrong_types():
    with pytest.raises(ValueError):
        validate_live_decision_consistency(
            {
                "consistent": "yes",
                "record_count": 0,
                "issue_count": 0,
                "issues": [],
                "duplicate_consecutive_timestamps": 0,
            }
        )


def test_validate_consistency_result_rejects_wrong_issue_count():
    with pytest.raises(ValueError):
        validate_live_decision_consistency(
            {
                "consistent": False,
                "record_count": 1,
                "issue_count": 2,
                "issues": ["one"],
                "duplicate_consecutive_timestamps": 0,
            }
        )
