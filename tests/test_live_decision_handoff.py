from __future__ import annotations

import pytest

from src.evaluation.live_decision_handoff import (
    build_live_decision_handoff,
    validate_live_decision_handoff,
)


def _pipeline_result(
    signal_label: str = "BUY",
) -> dict:
    return {
        "record": {
            "timestamp": "2026-09-10T05:45:00+00:00",
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
        },
        "history": [],
        "evaluation": {
            "record_count": 1,
            "health_status": "INITIALIZING",
            "healthy": True,
            "audit_passed": True,
        },
        "persisted": False,
        "store_path": None,
    }


def test_build_live_decision_handoff_preserves_decision_values():
    result = build_live_decision_handoff(
        _pipeline_result()
    )

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["strategy"] == "momentum"

    assert result["entry_price"] == 4429.802
    assert result["stop_loss"] == 4385.50398
    assert result["take_profit"] == 4518.39804
    assert result["risk_reward_ratio"] == 2.0
    assert result["stability_score"] == 0.65

    assert result["quote_stale"] is False
    assert result["candle_count"] == 201


def test_build_live_decision_handoff_exposes_evaluation_state():
    result = build_live_decision_handoff(
        _pipeline_result()
    )

    assert result["health_status"] == "INITIALIZING"
    assert result["healthy"] is True
    assert result["audit_passed"] is True
    assert result["record_count"] == 1


def test_buy_decision_is_actionable():
    result = build_live_decision_handoff(
        _pipeline_result("BUY")
    )

    assert result["actionable"] is True


def test_no_trade_decision_is_not_actionable():
    result = build_live_decision_handoff(
        _pipeline_result("NO TRADE")
    )

    assert result["actionable"] is False
    assert result["signal_label"] == "NO TRADE"


def test_build_live_decision_handoff_preserves_persistence_state():
    pipeline = _pipeline_result()

    pipeline["persisted"] = True
    pipeline["store_path"] = "results/live_decisions.json"

    result = build_live_decision_handoff(pipeline)

    assert result["persisted"] is True
    assert result["store_path"] == "results/live_decisions.json"


def test_build_live_decision_handoff_does_not_recalculate_values():
    pipeline = _pipeline_result()

    pipeline["record"]["entry_price"] = 9999.0
    pipeline["record"]["stop_loss"] = 9000.0
    pipeline["record"]["take_profit"] = 12000.0
    pipeline["record"]["stability_score"] = 0.11

    result = build_live_decision_handoff(pipeline)

    assert result["entry_price"] == 9999.0
    assert result["stop_loss"] == 9000.0
    assert result["take_profit"] == 12000.0
    assert result["stability_score"] == 0.11


def test_validate_live_decision_handoff_accepts_valid_result():
    handoff = build_live_decision_handoff(
        _pipeline_result()
    )

    assert (
        validate_live_decision_handoff(handoff)
        is True
    )


def test_validate_live_decision_handoff_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_handoff(
            {
                "symbol": "XAUUSD",
            }
        )


def test_validate_live_decision_handoff_rejects_wrong_actionable_type():
    handoff = build_live_decision_handoff(
        _pipeline_result()
    )

    handoff["actionable"] = "yes"

    with pytest.raises(ValueError):
        validate_live_decision_handoff(handoff)


def test_validate_live_decision_handoff_rejects_wrong_health_type():
    handoff = build_live_decision_handoff(
        _pipeline_result()
    )

    handoff["healthy"] = "yes"

    with pytest.raises(ValueError):
        validate_live_decision_handoff(handoff)


def test_build_live_decision_handoff_rejects_invalid_pipeline_result():
    with pytest.raises(TypeError):
        build_live_decision_handoff([])


def test_build_live_decision_handoff_rejects_missing_pipeline_fields():
    with pytest.raises(ValueError):
        build_live_decision_handoff(
            {
                "record": {},
                "evaluation": {},
            }
        )
```0
