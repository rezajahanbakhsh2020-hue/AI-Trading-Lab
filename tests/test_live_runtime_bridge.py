from __future__ import annotations

import pytest

from src.evaluation.live_runtime_bridge import (
    build_live_runtime_snapshot,
    run_live_runtime_bridge,
    validate_live_runtime_bridge,
)


def _live_result(
    signal_label: str = "BUY",
) -> dict:
    return {
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
        "timestamp": "2026-09-10T05:45:00+00:00",
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_build_live_runtime_snapshot_preserves_live_values():
    result = build_live_runtime_snapshot(
        _live_result()
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


def test_run_live_runtime_bridge_passes_ready_live_result():
    result = run_live_runtime_bridge(
        _live_result()
    )

    assert result["passed"] is True
    assert result["bridge"]["passed"] is True

    proof = result["bridge"]["end_to_end"]["proof"]

    assert proof["symbol"] == "XAUUSD"
    assert proof["signal_label"] == "BUY"
    assert proof["trend"] == "UP"
    assert proof["strategy"] == "momentum"


def test_run_live_runtime_bridge_preserves_risk_values():
    result = run_live_runtime_bridge(
        _live_result()
    )

    proof = result["bridge"]["end_to_end"]["proof"]

    assert proof["entry_price"] == 4429.802
    assert proof["stop_loss"] == 4385.50398
    assert proof["take_profit"] == 4518.39804
    assert proof["risk_reward_ratio"] == 2.0
    assert proof["stability_score"] == 0.65


def test_run_live_runtime_bridge_handles_no_trade():
    result = run_live_runtime_bridge(
        _live_result("NO TRADE")
    )

    assert result["passed"] is False
    assert (
        result["bridge"]["end_to_end"]["proof"]["status"]
        == "NO_ACTION"
    )


def test_run_live_runtime_bridge_can_persist(
    tmp_path,
):
    path = tmp_path / "live_decisions.json"

    result = run_live_runtime_bridge(
        _live_result(),
        store_path=str(path),
    )

    assert result["passed"] is True
    assert path.exists()
    assert result["bridge"]["pipeline"]["persisted"] is True


def test_run_live_runtime_bridge_preserves_history():
    first = _live_result()
    second = _live_result()

    second["timestamp"] = (
        "2026-09-10T05:50:00+00:00"
    )

    result = run_live_runtime_bridge(
        second,
        history=[first],
    )

    assert result["bridge"]["pipeline"]["evaluation"][
        "record_count"
    ] == 2


def test_runtime_bridge_does_not_recalculate_values():
    live_result = _live_result()

    live_result["entry_price"] = 9999.0
    live_result["stop_loss"] = 9000.0
    live_result["take_profit"] = 12000.0
    live_result["stability_score"] = 0.11

    result = run_live_runtime_bridge(
        live_result
    )

    proof = result["bridge"]["end_to_end"]["proof"]

    assert proof["entry_price"] == 9999.0
    assert proof["stop_loss"] == 9000.0
    assert proof["take_profit"] == 12000.0
    assert proof["stability_score"] == 0.11


def test_validate_live_runtime_bridge_accepts_valid_result():
    result = run_live_runtime_bridge(
        _live_result()
    )

    assert (
        validate_live_runtime_bridge(result)
        is True
    )


def test_validate_live_runtime_bridge_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_runtime_bridge(
            {
                "live_result": {},
            }
        )


def test_validate_live_runtime_bridge_rejects_wrong_pass_type():
    result = run_live_runtime_bridge(
        _live_result()
    )

    result["passed"] = "yes"

    with pytest.raises(ValueError):
        validate_live_runtime_bridge(result)


def test_build_live_runtime_snapshot_rejects_invalid_input():
    with pytest.raises(TypeError):
        build_live_runtime_snapshot([])


def test_build_live_runtime_snapshot_rejects_missing_fields():
    with pytest.raises(ValueError):
        build_live_runtime_snapshot(
            {
                "symbol": "XAUUSD",
            }
        )


def test_run_live_runtime_bridge_rejects_invalid_live_result():
    with pytest.raises(TypeError):
        run_live_runtime_bridge([])
