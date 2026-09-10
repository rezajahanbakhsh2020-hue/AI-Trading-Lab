from __future__ import annotations

import pytest

from src.evaluation.live_end_to_end import (
    run_live_end_to_end,
    validate_live_end_to_end,
)


def _pipeline_result(
    signal_label: str = "BUY",
    quote_stale: bool = False,
    market_state: str = "open",
    healthy: bool = True,
    audit_passed: bool = True,
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
            "health_status": (
                "HEALTHY"
                if healthy
                else "UNHEALTHY"
            ),
            "healthy": healthy,
            "audit_passed": audit_passed,
        },
        "persisted": True,
        "store_path": "results/live_decisions.json",
    }


def test_live_end_to_end_passes_for_ready_buy():
    result = run_live_end_to_end(
        _pipeline_result()
    )

    assert result["end_to_end_passed"] is True
    assert result["proof"]["symbol"] == "XAUUSD"
    assert result["proof"]["signal_label"] == "BUY"
    assert result["proof"]["trend"] == "UP"
    assert result["proof"]["strategy"] == "momentum"


def test_live_end_to_end_preserves_risk_values():
    result = run_live_end_to_end(
        _pipeline_result()
    )

    proof = result["proof"]

    assert proof["entry_price"] == 4429.802
    assert proof["stop_loss"] == 4385.50398
    assert proof["take_profit"] == 4518.39804
    assert proof["risk_reward_ratio"] == 2.0
    assert proof["stability_score"] == 0.65


def test_live_end_to_end_contains_final_decision():
    result = run_live_end_to_end(
        _pipeline_result()
    )

    assert isinstance(
        result["final_decision"],
        dict,
    )

    assert result["final_decision"]["status"] == "READY"


def test_live_end_to_end_fails_for_stale_quote():
    result = run_live_end_to_end(
        _pipeline_result(quote_stale=True)
    )

    assert result["end_to_end_passed"] is False
    assert result["proof"]["status"] == (
        "BLOCKED_STALE_QUOTE"
    )


def test_live_end_to_end_fails_for_closed_market():
    result = run_live_end_to_end(
        _pipeline_result(
            market_state="closed"
        )
    )

    assert result["end_to_end_passed"] is False
    assert result["proof"]["status"] == (
        "BLOCKED_MARKET"
    )


def test_live_end_to_end_fails_for_no_trade():
    result = run_live_end_to_end(
        _pipeline_result(
            signal_label="NO TRADE"
        )
    )

    assert result["end_to_end_passed"] is False
    assert result["proof"]["actionable"] is False
    assert result["proof"]["status"] == "NO_ACTION"


def test_live_end_to_end_fails_for_unhealthy_decision():
    result = run_live_end_to_end(
        _pipeline_result(
            healthy=False
        )
    )

    assert result["end_to_end_passed"] is False
    assert result["proof"]["healthy"] is False


def test_live_end_to_end_fails_for_failed_audit():
    result = run_live_end_to_end(
        _pipeline_result(
            audit_passed=False
        )
    )

    assert result["end_to_end_passed"] is False
    assert result["proof"]["audit_passed"] is False


def test_live_end_to_end_preserves_recorded_values():
    pipeline = _pipeline_result()

    pipeline["record"]["entry_price"] = 9999.0
    pipeline["record"]["stop_loss"] = 9000.0
    pipeline["record"]["take_profit"] = 12000.0
    pipeline["record"]["stability_score"] = 0.11

    result = run_live_end_to_end(pipeline)

    proof = result["proof"]

    assert proof["entry_price"] == 9999.0
    assert proof["stop_loss"] == 9000.0
    assert proof["take_profit"] == 12000.0
    assert proof["stability_score"] == 0.11


def test_validate_live_end_to_end_accepts_valid_result():
    result = run_live_end_to_end(
        _pipeline_result()
    )

    assert (
        validate_live_end_to_end(result)
        is True
    )


def test_validate_live_end_to_end_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_end_to_end(
            {
                "proof": {},
            }
        )


def test_validate_live_end_to_end_rejects_wrong_pass_type():
    result = run_live_end_to_end(
        _pipeline_result()
    )

    result["end_to_end_passed"] = "yes"

    with pytest.raises(ValueError):
        validate_live_end_to_end(result)


def test_run_live_end_to_end_rejects_invalid_input():
    with pytest.raises(TypeError):
        run_live_end_to_end([])
