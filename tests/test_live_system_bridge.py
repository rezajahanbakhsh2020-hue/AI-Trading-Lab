from __future__ import annotations

import pytest

from src.evaluation.live_system_bridge import (
    run_live_system_bridge,
    validate_live_system_bridge,
)


def _snapshot(
    signal_label: str = "BUY",
    quote_stale: bool = False,
    market_state: str = "open",
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
    }


def test_live_system_bridge_passes_ready_buy():
    result = run_live_system_bridge(
        _snapshot()
    )

    assert result["passed"] is True

    assert result["snapshot"]["symbol"] == "XAUUSD"

    assert (
        result["pipeline"]["record"]["signal_label"]
        == "BUY"
    )

    assert (
        result["end_to_end"]["end_to_end_passed"]
        is True
    )


def test_live_system_bridge_preserves_market_decision():
    result = run_live_system_bridge(
        _snapshot()
    )

    final_decision = result["end_to_end"][
        "final_decision"
    ]

    assert final_decision["symbol"] == "XAUUSD"
    assert final_decision["signal_label"] == "BUY"
    assert final_decision["trend"] == "UP"
    assert final_decision["strategy"] == "momentum"


def test_live_system_bridge_preserves_risk_values():
    result = run_live_system_bridge(
        _snapshot()
    )

    proof = result["end_to_end"]["proof"]

    assert proof["entry_price"] == 4429.802
    assert proof["stop_loss"] == 4385.50398
    assert proof["take_profit"] == 4518.39804
    assert proof["risk_reward_ratio"] == 2.0
    assert proof["stability_score"] == 0.65


def test_live_system_bridge_blocks_stale_quote():
    result = run_live_system_bridge(
        _snapshot(quote_stale=True)
    )

    assert result["passed"] is False
    assert (
        result["end_to_end"]["proof"]["status"]
        == "BLOCKED_STALE_QUOTE"
    )


def test_live_system_bridge_blocks_closed_market():
    result = run_live_system_bridge(
        _snapshot(market_state="closed")
    )

    assert result["passed"] is False
    assert (
        result["end_to_end"]["proof"]["status"]
        == "BLOCKED_MARKET"
    )


def test_live_system_bridge_handles_no_trade():
    result = run_live_system_bridge(
        _snapshot(signal_label="NO TRADE")
    )

    assert result["passed"] is False
    assert (
        result["end_to_end"]["proof"]["status"]
        == "NO_ACTION"
    )


def test_live_system_bridge_can_use_history():
    first = _snapshot()
    second = _snapshot()

    second["timestamp"] = (
        "2026-09-10T05:50:00+00:00"
    )

    result = run_live_system_bridge(
        second,
        history=[first],
    )

    assert result["pipeline"]["evaluation"][
        "record_count"
    ] == 2


def test_live_system_bridge_preserves_persistence_state(
    tmp_path,
):
    path = tmp_path / "live_decisions.json"

    result = run_live_system_bridge(
        _snapshot(),
        store_path=str(path),
    )

    assert result["passed"] is True
    assert result["pipeline"]["persisted"] is True
    assert result["pipeline"]["store_path"] == str(path)
    assert path.exists()


def test_live_system_bridge_does_not_recalculate_values():
    snapshot = _snapshot()

    snapshot["entry_price"] = 9999.0
    snapshot["stop_loss"] = 9000.0
    snapshot["take_profit"] = 12000.0
    snapshot["stability_score"] = 0.11

    result = run_live_system_bridge(snapshot)

    proof = result["end_to_end"]["proof"]

    assert proof["entry_price"] == 9999.0
    assert proof["stop_loss"] == 9000.0
    assert proof["take_profit"] == 12000.0
    assert proof["stability_score"] == 0.11


def test_validate_live_system_bridge_accepts_valid_result():
    result = run_live_system_bridge(
        _snapshot()
    )

    assert (
        validate_live_system_bridge(result)
        is True
    )


def test_validate_live_system_bridge_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_live_system_bridge(
            {
                "snapshot": {},
            }
        )


def test_validate_live_system_bridge_rejects_wrong_pass_type():
    result = run_live_system_bridge(
        _snapshot()
    )

    result["passed"] = "yes"

    with pytest.raises(ValueError):
        validate_live_system_bridge(result)


def test_run_live_system_bridge_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        run_live_system_bridge([])
