import pandas as pd
import pytest

from src.evaluation.live_trade_display import (
    DEFAULT_TP1_MULTIPLIER,
    DEFAULT_TP2_MULTIPLIER,
    DEFAULT_TP3_MULTIPLIER,
    build_live_trade_display,
)


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
    )

    close = pd.Series(
        [2000.0 + index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def test_buy_trade_display_contains_entry_sl_and_three_targets():
    result = build_live_trade_display(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert result["decision"] == "BUY"
    assert result["entry_price"] is not None
    assert result["stop_loss"] is not None
    assert result["tp1"] is not None
    assert result["tp2"] is not None
    assert result["tp3"] is not None

    assert result["stop_loss"] < result["entry_price"]
    assert result["entry_price"] < result["tp1"]
    assert result["tp1"] < result["tp2"]
    assert result["tp2"] < result["tp3"]


def test_default_targets_have_one_two_three_risk_structure():
    result = build_live_trade_display(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert result["tp1_multiplier"] == DEFAULT_TP1_MULTIPLIER
    assert result["tp2_multiplier"] == DEFAULT_TP2_MULTIPLIER
    assert result["tp3_multiplier"] == DEFAULT_TP3_MULTIPLIER

    assert result["risk_reward_tp1"] == pytest.approx(1.0)
    assert result["risk_reward_tp2"] == pytest.approx(2.0)
    assert result["risk_reward_tp3"] == pytest.approx(3.0)


def test_custom_target_multipliers_are_applied():
    result = build_live_trade_display(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
        tp1_multiplier=0.5,
        tp2_multiplier=1.5,
        tp3_multiplier=2.5,
    )

    assert result["risk_reward_tp1"] == pytest.approx(0.5)
    assert result["risk_reward_tp2"] == pytest.approx(1.5)
    assert result["risk_reward_tp3"] == pytest.approx(2.5)


def test_no_trade_has_no_trade_levels():
    result = build_live_trade_display(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.20,
    )

    assert result["decision"] == "NO TRADE"
    assert result["entry_price"] is None
    assert result["stop_loss"] is None
    assert result["tp1"] is None
    assert result["tp2"] is None
    assert result["tp3"] is None
    assert result["risk_distance"] is None


def test_unsupported_strategy_has_no_trade_levels():
    result = build_live_trade_display(
        _rising_data(),
        stable_strategy="moving_average",
        stability_score=0.80,
    )

    assert result["decision"] == "NO TRADE"
    assert result["strategy_supported"] is False
    assert result["tp1"] is None
    assert result["tp2"] is None
    assert result["tp3"] is None


def test_tp_multipliers_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="TP1 < TP2 < TP3"):
        build_live_trade_display(
            _rising_data(),
            stable_strategy="momentum",
            stability_score=0.80,
            tp1_multiplier=2.0,
            tp2_multiplier=1.0,
            tp3_multiplier=3.0,
        )


def test_tp_multipliers_must_be_positive():
    with pytest.raises(
        ValueError,
        match="tp1_multiplier must be greater than 0",
    ):
        build_live_trade_display(
            _rising_data(),
            stable_strategy="momentum",
            stability_score=0.80,
            tp1_multiplier=0.0,
        )


def test_no_sell_decision_is_invented():
    result = build_live_trade_display(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert result["decision"] in {"BUY", "NO TRADE"}
    assert result["decision"] != "SELL"
