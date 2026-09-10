from __future__ import annotations

import pytest

from src.evaluation.live_trade_decision import (
    build_live_trade_decision,
    validate_live_trade_decision,
)


def _snapshot(
    signal_label: str = "BUY",
    trend: str = "UP",
    risk_reward_ratio: float = 2.0,
) -> dict:
    return {
        "timestamp": "2026-09-10T10:10:00+00:00",
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": (
            1
            if signal_label == "BUY"
            else -1
            if signal_label == "SELL"
            else 0
        ),
        "signal_label": signal_label,
        "trend": trend,
        "strategy": "momentum",
        "entry_price": 4393.317,
        "stop_loss": 4349.38383,
        "take_profit": 4481.18334,
        "risk_reward_ratio": risk_reward_ratio,
        "stability_score": 0.517268,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_buy_with_up_trend_produces_long():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP")
    )

    assert result["decision"] == "LONG"
    assert result["decision_label"] == "LONG"
    assert result["trade_allowed"] is True
    assert result["counter_trend"] is False
    assert result["confidence"] > 0
    assert "aligned" in result["reason"]


def test_sell_with_down_trend_produces_short():
    result = build_live_trade_decision(
        _snapshot("SELL", "DOWN")
    )

    assert result["decision"] == "SHORT"
    assert result["decision_label"] == "SHORT"
    assert result["trade_allowed"] is True
    assert result["counter_trend"] is False


def test_buy_with_down_trend_is_rejected_as_counter_trend():
    result = build_live_trade_decision(
        _snapshot("BUY", "DOWN")
    )

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert result["counter_trend"] is True
    assert "conflicts" in result["reason"]


def test_sell_with_up_trend_is_rejected_as_counter_trend():
    result = build_live_trade_decision(
        _snapshot("SELL", "UP")
    )

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert result["counter_trend"] is True
    assert "conflicts" in result["reason"]


def test_flat_trend_produces_no_trade():
    result = build_live_trade_decision(
        _snapshot("BUY", "FLAT")
    )

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert result["counter_trend"] is False
    assert result["confidence"] == 0.0


def test_no_trade_signal_produces_no_trade():
    result = build_live_trade_decision(
        _snapshot("NO TRADE", "UP")
    )

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert result["confidence"] == 0.0


def test_stale_quote_blocks_trade():
    snapshot = _snapshot("BUY", "UP")
    snapshot["quote_stale"] = True

    result = build_live_trade_decision(snapshot)

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert "stale" in result["reason"].lower()


def test_closed_market_blocks_trade():
    snapshot = _snapshot("BUY", "UP")
    snapshot["market_state"] = "closed"

    result = build_live_trade_decision(snapshot)

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert "open" in result["reason"].lower()


def test_invalid_risk_reward_blocks_trade():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP", 0.8)
    )

    assert result["decision"] == "NO TRADE"
    assert result["trade_allowed"] is False
    assert "risk/reward" in result["reason"].lower()


def test_custom_minimum_risk_reward_is_supported():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP", 1.5),
        minimum_risk_reward=1.5,
    )

    assert result["decision"] == "LONG"
    assert result["risk_reward_ratio"] == 1.5
    assert result["minimum_risk_reward"] == 1.5


def test_decision_contains_existing_trade_levels():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP")
    )

    assert result["entry_price"] == 4393.317
    assert result["stop_loss"] == 4349.38383
    assert result["take_profit"] == 4481.18334
    assert result["tp1"] == 4481.18334
    assert result["tp2"] is None
    assert result["tp3"] is None


def test_validation_accepts_valid_long_decision():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP")
    )

    assert validate_live_trade_decision(result) is True


def test_validation_accepts_valid_no_trade_decision():
    result = build_live_trade_decision(
        _snapshot("BUY", "DOWN")
    )

    assert validate_live_trade_decision(result) is True


def test_validation_rejects_invalid_decision():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP")
    )
    result["decision"] = "BUY"

    with pytest.raises(ValueError):
        validate_live_trade_decision(result)


def test_validation_rejects_invalid_confidence():
    result = build_live_trade_decision(
        _snapshot("BUY", "UP")
    )
    result["confidence"] = 101.0

    with pytest.raises(ValueError):
        validate_live_trade_decision(result)


def test_non_mapping_snapshot_is_rejected():
    with pytest.raises(TypeError):
        build_live_trade_decision([])


def test_negative_minimum_risk_reward_is_rejected():
    with pytest.raises(ValueError):
        build_live_trade_decision(
            _snapshot(),
            minimum_risk_reward=-1.0,
        )
