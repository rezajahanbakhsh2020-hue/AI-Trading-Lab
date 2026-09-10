from __future__ import annotations

import pytest

from src.evaluation.live_decision_readiness import (
    evaluate_live_decision_readiness,
    validate_live_decision_readiness,
)


def _handoff(
    signal_label: str = "BUY",
    quote_stale: bool = False,
    market_state: str = "open",
    healthy: bool = True,
    audit_passed: bool = True,
) -> dict:
    return {
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
        "health_status": "HEALTHY" if healthy else "UNHEALTHY",
        "healthy": healthy,
        "audit_passed": audit_passed,
        "record_count": 2,
        "persisted": True,
        "store_path": "results/live_decisions.json",
        "actionable": signal_label in {"BUY", "SELL"},
    }


def test_buy_live_decision_can_be_ready():
    result = evaluate_live_decision_readiness(
        _handoff()
    )

    assert result["ready"] is True
    assert result["status"] == "READY"
    assert result["symbol"] == "XAUUSD"
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["strategy"] == "momentum"


def test_ready_result_preserves_risk_values():
    result = evaluate_live_decision_readiness(
        _handoff()
    )

    assert result["entry_price"] == 4429.802
    assert result["stop_loss"] == 4385.50398
    assert result["take_profit"] == 4518.39804
    assert result["risk_reward_ratio"] == 2.0
    assert result["stability_score"] == 0.65


def test_stale_quote_blocks_actionable_decision():
    result = evaluate_live_decision_readiness(
        _handoff(quote_stale=True)
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED_STALE_QUOTE"
    assert result["checks"]["quote_fresh"] is False


def test_closed_market_blocks_actionable_decision():
    result = evaluate_live_decision_readiness(
        _handoff(market_state="closed")
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED_MARKET"
    assert result["checks"]["market_open"] is False


def test_no_trade_is_not_actionable():
    result = evaluate_live_decision_readiness(
        _handoff(signal_label="NO TRADE")
    )

    assert result["ready"] is False
    assert result["status"] == "NO_ACTION"
    assert result["checks"]["actionable"] is False


def test_unhealthy_decision_is_blocked():
    result = evaluate_live_decision_readiness(
        _handoff(healthy=False)
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED_HEALTH"


def test_failed_audit_is_blocked():
    result = evaluate_live_decision_readiness(
        _handoff(audit_passed=False)
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED_HEALTH"
    assert result["checks"]["audit_passed"] is False


def test_readiness_validation_accepts_valid_result():
    result = evaluate_live_decision_readiness(
        _handoff()
    )

    assert (
        validate_live_decision_readiness(result)
        is True
    )


def test_readiness_validation_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_decision_readiness(
            {
                "ready": True,
                "status": "READY",
            }
        )


def test_readiness_validation_rejects_wrong_ready_type():
    result = evaluate_live_decision_readiness(
        _handoff()
    )

    result["ready"] = "yes"

    with pytest.raises(ValueError):
        validate_live_decision_readiness(result)


def test_readiness_validation_rejects_wrong_checks_type():
    result = evaluate_live_decision_readiness(
        _handoff()
    )

    result["checks"] = []

    with pytest.raises(ValueError):
        validate_live_decision_readiness(result)


def test_readiness_rejects_invalid_handoff():
    with pytest.raises(TypeError):
        evaluate_live_decision_readiness([])
