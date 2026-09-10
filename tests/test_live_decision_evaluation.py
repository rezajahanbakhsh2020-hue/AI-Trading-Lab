from __future__ import annotations

import pytest

from src.evaluation.live_decision_evaluation import (
    evaluate_live_decision_history,
    validate_live_decision_evaluation,
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


def test_evaluation_combines_summary_and_health():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
        ),
        _record("2026-09-10T05:55:00+00:00"),
    ]

    result = evaluate_live_decision_history(history)

    assert result["record_count"] == 3
    assert result["signal_counts"] == {
        "BUY": 2,
        "NO TRADE": 1,
    }
    assert result["trend_counts"] == {
        "UP": 2,
        "FLAT": 1,
    }
    assert result["health_status"] == "HEALTHY"
    assert result["healthy"] is True
    assert result["audit_passed"] is True


def test_evaluation_returns_latest_decision():
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
        ),
    ]

    result = evaluate_live_decision_history(history)

    assert result["latest"]["timestamp"] == (
        "2026-09-10T05:50:00+00:00"
    )
    assert result["latest"]["signal_label"] == "NO TRADE"


def test_empty_history_evaluation():
    result = evaluate_live_decision_history([])

    assert result["record_count"] == 0
    assert result["signal_counts"] == {}
    assert result["trend_counts"] == {}
    assert result["strategy_counts"] == {}
    assert result["latest"] is None
    assert result["health_status"] == "EMPTY"
    assert result["healthy"] is False


def test_evaluation_reports_changed_transitions():
    history = [
        _record(
            "2026-09-10T05:45:00+00:00",
            "BUY",
            "UP",
            "momentum",
        ),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
            "mean_reversion",
        ),
    ]

    result = evaluate_live_decision_history(history)

    assert result["changed_transition_count"] == 1


def test_evaluation_reports_timeline_problem():
    history = [
        _record("2026-09-10T05:50:00+00:00"),
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = evaluate_live_decision_history(history)

    assert result["health_status"] == "UNHEALTHY"
    assert result["healthy"] is False
    assert result["audit_passed"] is False
    assert result["timeline_issue_count"] == 1


def test_evaluation_reports_structure_problem():
    record = _record("2026-09-10T05:45:00+00:00")
    del record["strategy"]

    result = evaluate_live_decision_history([record])

    assert result["health_status"] == "UNHEALTHY"
    assert result["healthy"] is False
    assert result["invalid_structure_count"] == 1


def test_evaluation_preserves_recorded_trading_values():
    record = _record("2026-09-10T05:45:00+00:00")

    record["entry_price"] = 9999.0
    record["stop_loss"] = 9000.0
    record["take_profit"] = 12000.0
    record["stability_score"] = 0.11

    result = evaluate_live_decision_history([record])

    assert result["latest"]["entry_price"] == 9999.0
    assert result["latest"]["stop_loss"] == 9000.0
    assert result["latest"]["take_profit"] == 12000.0
    assert result["latest"]["stability_score"] == 0.11


def test_invalid_history_input_is_rejected():
    with pytest.raises(TypeError):
        evaluate_live_decision_history("invalid")


def test_invalid_history_record_is_rejected():
    with pytest.raises(TypeError):
        evaluate_live_decision_history([[]])


def test_validation_accepts_valid_evaluation():
    result = evaluate_live_decision_history(
        [_record("2026-09-10T05:45:00+00:00")]
    )

    assert validate_live_decision_evaluation(result) is True


def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_evaluation(
            {
                "record_count": 1,
            }
        )


def test_validation_rejects_wrong_count_type():
    with pytest.raises(ValueError):
        validate_live_decision_evaluation(
            {
                "record_count": "1",
                "signal_counts": {},
                "trend_counts": {},
                "strategy_counts": {},
                "latest": None,
                "health_status": "EMPTY",
                "healthy": False,
                "audit_passed": True,
                "invalid_structure_count": 0,
                "timeline_issue_count": 0,
                "changed_transition_count": 0,
            }
        )


def test_validation_rejects_wrong_latest_type():
    with pytest.raises(ValueError):
        validate_live_decision_evaluation(
            {
                "record_count": 1,
                "signal_counts": {},
                "trend_counts": {},
                "strategy_counts": {},
                "latest": "invalid",
                "health_status": "HEALTHY",
                "healthy": True,
                "audit_passed": True,
                "invalid_structure_count": 0,
                "timeline_issue_count": 0,
                "changed_transition_count": 0,
            }
        )
