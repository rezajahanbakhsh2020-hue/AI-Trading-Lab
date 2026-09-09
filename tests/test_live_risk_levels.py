import pandas as pd
import pytest

import live_risk_levels


def make_market_data(closes):
    closes = [float(value) for value in closes]

    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=len(closes),
                freq="5min",
            ),
            "open": closes,
            "high": [value + 2.0 for value in closes],
            "low": [value - 2.0 for value in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
            "tickVolume": [100.0] * len(closes),
            "isOpen": [False] * len(closes),
        }
    )


def test_build_live_risk_levels_for_buy_signal():
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    result = live_risk_levels.build_live_risk_levels(
        data,
        momentum_window=2,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )

    assert result["signal"] == 1
    assert result["signal_label"] == "BUY"
    assert result["entry_price"] == 6.0
    assert result["stop_loss"] == pytest.approx(6.0 * 0.99)
    assert result["take_profit"] == pytest.approx(6.0 * 1.02)
    assert result["risk_reward_ratio"] == pytest.approx(2.0)
    assert result["stop_loss_pct"] == pytest.approx(0.01)
    assert result["take_profit_pct"] == pytest.approx(0.02)


def test_build_live_risk_levels_returns_no_levels_for_no_trade():
    data = make_market_data(
        [6, 5, 4, 3, 2, 1]
    )

    result = live_risk_levels.build_live_risk_levels(
        data,
        momentum_window=2,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )

    assert result["signal"] == 0
    assert result["signal_label"] == "NO TRADE"
    assert result["entry_price"] == 1.0
    assert result["stop_loss"] is None
    assert result["take_profit"] is None
    assert result["risk_reward_ratio"] is None


def test_build_live_risk_levels_preserves_timestamp():
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    result = live_risk_levels.build_live_risk_levels(
        data,
        momentum_window=2,
    )

    expected_timestamp = (
        data["openTime"].iloc[-1]
        .tz_localize("UTC")
        .isoformat()
    )

    assert result["timestamp"] == expected_timestamp


def test_build_live_risk_levels_rejects_empty_data():
    with pytest.raises(ValueError, match="must not be empty"):
        live_risk_levels.build_live_risk_levels(
            pd.DataFrame()
        )


def test_build_live_risk_levels_rejects_missing_close():
    data = pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="5min",
            ),
            "open": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="close"):
        live_risk_levels.build_live_risk_levels(data)


def test_build_live_risk_levels_rejects_invalid_percentages():
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    with pytest.raises(
        ValueError,
        match="stop_loss_pct must be greater than zero",
    ):
        live_risk_levels.build_live_risk_levels(
            data,
            stop_loss_pct=0,
        )

    with pytest.raises(
        ValueError,
        match="take_profit_pct must be greater than zero",
    ):
        live_risk_levels.build_live_risk_levels(
            data,
            take_profit_pct=-0.01,
        )
