from __future__ import annotations

import pytest

from src.evaluation.live_runtime_proof import (
    build_live_runtime_proof,
    validate_live_runtime_proof,
)


def _live_result() -> dict:
    return {
        "signal": 1,
        "signal_label": "BUY",
        "strategy": "momentum",
        "trend": "UP",
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "timestamp": "2026-09-10T05:45:00+00:00",
    }


def test_build_live_runtime_proof_passes_real_runtime_values():
    result = build_live_runtime_proof(
        _live_result(),
        symbol="XAUUSD",
        interval="5m",
        market_state="open",
        quote_age_seconds=0,
        quote_stale=False,
        candle_count=201,
    )

    assert result["passed"] is True
    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["strategy"] == "momentum"


def test_build_live_runtime_proof_preserves_risk_values():
    result = build_live_runtime_proof(
        _live_result(),
    )

    assert result["entry_price"] == 4429.802
    assert result["stop_loss"] == 4385.50398
    assert result["take_profit"] == 4518.39804
    assert result["risk_reward_ratio"] == 2.0
    assert result["stability_score"] == 0.65


def test_build_live_runtime_proof_preserves_market_proof():
    result = build_live_runtime_proof(
        _live_result(),
        symbol="XAUUSD",
        interval="5m",
        market_state="open",
        quote_age_seconds=0,
        quote_stale=False,
        candle_count=201,
    )

    assert result["market_state"] == "open"
    assert result["quote_age_seconds"] == 0
    assert result["quote_stale"] is False
    assert result["candle_count"] == 201


def test_build_live_runtime_proof_handles_no_trade():
    live_result = _live_result()
    live_result["signal"] = 0
    live_result["signal_label"] = "NO TRADE"
    live_result["stop_loss"] = None
    live_result["take_profit"] = None
    live_result["risk_reward_ratio"] = None

    result = build_live_runtime_proof(
        live_result,
    )

    assert result["passed"] is False
    assert (
        result["proof"]["status"]
        == "NO_ACTION"
    )


def test_build_live_runtime_proof_can_persist(tmp_path):
    path = tmp_path / "live_runtime_proof.json"

    result = build_live_runtime_proof(
        _live_result(),
        store_path=str(path),
    )

    assert result["passed"] is True
    assert path.exists()
    assert (
        result["bridge"]["bridge"]["pipeline"]["persisted"]
        is True
    )


def test_build_live_runtime_proof_preserves_history():
    first = _live_result()

    second = _live_result()
    second["timestamp"] = (
        "2026-09-10T05:50:00+00:00"
    )

    result = build_live_runtime_proof(
        second,
        history=[first],
    )

    assert (
        result["bridge"]["bridge"]["pipeline"][
            "evaluation"
        ]["record_count"]
        == 2
    )


def test_build_live_runtime_proof_does_not_recalculate():
    live_result = _live_result()

    live_result["entry_price"] = 9999.0
    live_result["stop_loss"] = 9000.0
    live_result["take_profit"] = 12000.0
    live_result["risk_reward_ratio"] = 3.5
    live_result["stability_score"] = 0.11

    result = build_live_runtime_proof(
        live_result,
    )

    assert result["entry_price"] == 9999.0
    assert result["stop_loss"] == 9000.0
    assert result["take_profit"] == 12000.0
    assert result["risk_reward_ratio"] == 3.5
    assert result["stability_score"] == 0.11


def test_validate_live_runtime_proof_accepts_valid_result():
    result = build_live_runtime_proof(
        _live_result(),
    )

    assert (
        validate_live_runtime_proof(result)
        is True
    )


def test_validate_live_runtime_proof_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_runtime_proof(
            {
                "symbol": "XAUUSD",
            }
        )


def test_validate_live_runtime_proof_rejects_wrong_pass_type():
    result = build_live_runtime_proof(
        _live_result(),
    )

    result["passed"] = "yes"

    with pytest.raises(ValueError):
        validate_live_runtime_proof(result)


def test_validate_live_runtime_proof_rejects_wrong_quote_stale_type():
    result = build_live_runtime_proof(
        _live_result(),
    )

    result["quote_stale"] = "false"

    with pytest.raises(ValueError):
        validate_live_runtime_proof(result)


def test_build_live_runtime_proof_rejects_invalid_input():
    with pytest.raises(TypeError):
        build_live_runtime_proof([])


def test_build_live_runtime_proof_rejects_missing_live_fields():
    with pytest.raises(ValueError):
        build_live_runtime_proof(
            {
                "signal": 1,
                "signal_label": "BUY",
            }
        )
