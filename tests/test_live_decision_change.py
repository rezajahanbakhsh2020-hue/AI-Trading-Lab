from __future__ import annotations

import pytest

from src.evaluation.live_decision_change import (
    compare_live_decisions,
    summarize_live_decision_changes,
    validate_live_decision_change,
)


def _record(
    timestamp: str,
    signal_label: str = "BUY",
    trend: str = "UP",
    strategy: str = "momentum",
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
        "quote_stale": False,
        "candle_count": 201,
    }


def test_identical_decisions_have_no_changes():
    record = _record("2026-09-10T05:45:00+00:00")

    result = compare_live_decisions(record, dict(record))

    assert result["changed"] is False
    assert result["changed_field_count"] == 0
    assert result["changed_fields"] == {}


def test_signal_change_is_detected():
    previous = _record(
        "2026-09-10T05:45:00+00:00",
        "BUY",
    )
    current = _record(
        "2026-09-10T05:50:00+00:00",
        "NO TRADE",
        "FLAT",
    )

    result = compare_live_decisions(previous, current)

    assert result["changed"] is True
    assert "signal_label" in result["changed_fields"]
    assert result["changed_fields"]["signal_label"] == {
        "previous": "BUY",
        "current": "NO TRADE",
    }


def test_trend_change_is_detected():
    previous = _record(
        "2026-09-10T05:45:00+00:00",
        "BUY",
        "UP",
    )
    current = _record(
        "2026-09-10T05:50:00+00:00",
        "BUY",
        "DOWN",
    )

    result = compare_live_decisions(previous, current)

    assert result["changed"] is True
    assert "trend" in result["changed_fields"]


def test_strategy_change_is_detected():
    previous = _record(
        "2026-09-10T05:45:00+00:00",
        strategy="momentum",
    )
    current = _record(
        "2026-09-10T05:50:00+00:00",
        strategy="mean_reversion",
    )

    result = compare_live_decisions(previous, current)

    assert result["changed"] is True
    assert result["changed_fields"]["strategy"] == {
        "previous": "momentum",
        "current": "mean_reversion",
    }


def test_risk_change_is_detected():
    previous = _record(
        "2026-09-10T05:45:00+00:00",
    )
    current = _record(
        "2026-09-10T05:50:00+00:00",
    )
    current["stop_loss"] = 4400.0

    result = compare_live_decisions(previous, current)

    assert result["changed"] is True
    assert "stop_loss" in result["changed_fields"]


def test_stability_change_is_detected():
    previous = _record(
        "2026-09-10T05:45:00+00:00",
    )
    current = _record(
        "2026-09-10T05:50:00+00:00",
    )
    current["stability_score"] = 0.72

    result = compare_live_decisions(previous, current)

    assert result["changed"] is True
    assert result["changed_fields"]["stability_score"] == {
        "previous": 0.65,
        "current": 0.72,
    }


def test_summary_classifies_changes():
    previous = _record(
        "2026-09-10T05:45:00+00:00",
        "BUY",
        "UP",
        "momentum",
    )
    current = _record(
        "2026-09-10T05:50:00+00:00",
        "NO TRADE",
        "FLAT",
        "mean_reversion",
    )
    current["take_profit"] = 4600.0
    current["stability_score"] = 0.70
    current["market_state"] = "closed"
    current["quote_stale"] = True

    result = summarize_live_decision_changes(
        previous,
        current,
    )

    assert result["changed"] is True
    assert result["signal_changed"] is True
    assert result["trend_changed"] is True
    assert result["strategy_changed"] is True
    assert result["risk_changed"] is True
    assert result["stability_changed"] is True
    assert result["market_state_changed"] is True
    assert result["quote_status_changed"] is True


def test_summary_reports_no_changes_for_identical_records():
    record = _record("2026-09-10T05:45:00+00:00")

    result = summarize_live_decision_changes(
        record,
        dict(record),
    )

    assert result["changed"] is False
    assert result["signal_changed"] is False
    assert result["trend_changed"] is False
    assert result["strategy_changed"] is False
    assert result["risk_changed"] is False
    assert result["stability_changed"] is False


def test_original_values_are_not_modified():
    previous = _record("2026-09-10T05:45:00+00:00")
    current = _record("2026-09-10T05:50:00+00:00")

    compare_live_decisions(previous, current)

    assert previous["entry_price"] == 4429.802
    assert current["entry_price"] == 4429.802


def test_invalid_previous_record_is_rejected():
    with pytest.raises(TypeError):
        compare_live_decisions([], _record(
            "2026-09-10T05:50:00+00:00"
        ))


def test_invalid_current_record_is_rejected():
    with pytest.raises(TypeError):
        compare_live_decisions(
            _record("2026-09-10T05:45:00+00:00"),
            [],
        )


def test_validation_accepts_valid_comparison():
    result = compare_live_decisions(
        _record("2026-09-10T05:45:00+00:00"),
        _record("2026-09-10T05:50:00+00:00"),
    )

    assert validate_live_decision_change(result) is True


def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_change(
            {
                "changed": False,
            }
        )


def test_validation_rejects_wrong_changed_type():
    with pytest.raises(ValueError):
        validate_live_decision_change(
            {
                "changed": "no",
                "changed_field_count": 0,
                "changed_fields": {},
                "previous_timestamp": None,
                "current_timestamp": None,
            }
        )


def test_validation_rejects_wrong_field_count():
    with pytest.raises(ValueError):
        validate_live_decision_change(
            {
                "changed": True,
                "changed_field_count": 2,
                "changed_fields": {
                    "signal_label": {
                        "previous": "BUY",
                        "current": "NO TRADE",
                    }
                },
                "previous_timestamp": "a",
                "current_timestamp": "b",
            }
        )
