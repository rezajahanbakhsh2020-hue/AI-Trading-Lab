from __future__ import annotations

import pytest

from src.evaluation.live_decision_final import (
    finalize_live_decision,
    validate_final_live_decision,
)


def _pipeline_result(
    signal_label: str = "BUY",
    quote_stale: bool = False,
    market_state: str = "open",
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
            "market_state": market_state,
            "quote_age_seconds": 0,
            "quote_stale": quote_stale,
            "candle_count": 201,
        },
        "history": [],
        "evaluation": {
            "record_count": 2,
            "health_status": "HEALTHY",
            "healthy": True,
            "audit_passed": True,
        },
        "persisted": True,
        "store_path": "results/live_decisions.json",
    }


def test_finalize_live_decision_builds_complete_package():
    result = finalize_live_decision(
        _pipeline_result()
    )

    assert result["ready"] is True
    assert result["status"] == "READY"
    assert result["actionable"] is True

    assert result["symbol"] == "XAUUSD"
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["strategy"] == "momentum"


def test_finalize_live_decision_preserves_risk_values():
    result = finalize_live_decision(
        _pipeline_result()
    )

    assert result["entry_price"] == 4429.802
    assert result["stop_loss"] == 4385.50398
    assert result["take_profit"] == 4518.39804
    assert result["risk_reward_ratio"] == 2.0
    assert result["stability_score"] == 0.65


def test_finalize_live_decision_contains_handoff_and_readiness():
    result = finalize_live_decision(
        _pipeline_result()
    )

    assert isinstance(result["handoff"], dict)
    assert isinstance(result["readiness"], dict)

    assert result["handoff"]["signal_label"] == "BUY"
    assert result["readiness"]["status"] == "READY"


def test_finalize_live_decision_blocks_stale_quote():
    result = finalize_live_decision(
        _pipeline_result(quote_stale=True)
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED_STALE_QUOTE"
    assert result["actionable"] is True


def test_finalize_live_decision_blocks_closed_market():
    result = finalize_live_decision(
        _pipeline_result(market_state="closed")
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED_MARKET"


def test_finalize_live_decision_handles_no_trade():
    result = finalize_live_decision(
        _pipeline_result(signal_label="NO TRADE")
    )

    assert result["ready"] is False
    assert result["status"] == "NO_ACTION"
    assert result["actionable"] is False


def test_finalize_live_decision_does_not_recalculate_values():
    pipeline = _pipeline_result()

    pipeline["record"]["entry_price"] = 9999.0
    pipeline["record"]["stop_loss"] = 9000.0
    pipeline["record"]["take_profit"] = 12000.0
    pipeline["record"]["stability_score"] = 0.11

    result = finalize_live_decision(pipeline)

    assert result["entry_price"] == 9999.0
    assert result["stop_loss"] == 9000.0
    assert result["take_profit"] == 12000.0
    assert result["stability_score"] == 0.11


def test_validate_final_live_decision_accepts_valid_result():
    result = finalize_live_decision(
        _pipeline_result()
    )

    assert (
        validate_final_live_decision(result)
        is True
    )


def test_validate_final_live_decision_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_final_live_decision(
            {
                "handoff": {},
                "readiness": {},
            }
        )


def test_validate_final_live_decision_rejects_wrong_ready_type():
    result = finalize_live_decision(
        _pipeline_result()
    )

    result["ready"] = "yes"

    with pytest.raises(ValueError):
        validate_final_live_decision(result)


def test_validate_final_live_decision_rejects_invalid_input():
    with pytest.raises(TypeError):
        finalize_live_decision([])
