import pandas as pd
import pytest

import app_live_signal_trend_risk


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


def test_build_signal_trend_risk_snapshot_combines_all_engines():
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    snapshot = (
        app_live_signal_trend_risk
        .build_signal_trend_risk_snapshot(
            data,
            momentum_window=2,
            fast_window=2,
            slow_window=4,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
        )
    )

    assert snapshot["signal"] == 1
    assert snapshot["signal_label"] == "BUY"
    assert snapshot["trend"] == "UP"

    assert snapshot["entry_price"] == 6.0
    assert snapshot["stop_loss"] == pytest.approx(5.94)
    assert snapshot["take_profit"] == pytest.approx(6.12)
    assert snapshot["risk_reward_ratio"] == pytest.approx(2.0)

    assert snapshot["momentum"] == pytest.approx(
        (6.0 / 4.0) - 1.0
    )

    assert snapshot["strategy"] == "momentum"
    assert snapshot["momentum_window"] == 2
    assert snapshot["fast_window"] == 2
    assert snapshot["slow_window"] == 4


def test_build_signal_trend_risk_snapshot_has_no_risk_levels_for_no_trade():
    data = make_market_data(
        [6, 5, 4, 3, 2, 1]
    )

    snapshot = (
        app_live_signal_trend_risk
        .build_signal_trend_risk_snapshot(
            data,
            momentum_window=2,
            fast_window=2,
            slow_window=4,
        )
    )

    assert snapshot["signal"] == 0
    assert snapshot["signal_label"] == "NO TRADE"
    assert snapshot["trend"] == "DOWN"

    assert snapshot["entry_price"] == 1.0
    assert snapshot["stop_loss"] is None
    assert snapshot["take_profit"] is None
    assert snapshot["risk_reward_ratio"] is None


def test_build_signal_trend_risk_snapshot_preserves_timestamp():
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    snapshot = (
        app_live_signal_trend_risk
        .build_signal_trend_risk_snapshot(
            data,
            momentum_window=2,
            fast_window=2,
            slow_window=4,
        )
    )

    expected_timestamp = (
        data["openTime"].iloc[-1]
        .tz_localize("UTC")
        .isoformat()
    )

    assert snapshot["timestamp"] == expected_timestamp


def test_build_signal_trend_risk_snapshot_rejects_empty_data():
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        app_live_signal_trend_risk.build_signal_trend_risk_snapshot(
            pd.DataFrame()
        )


def test_build_signal_trend_risk_snapshot_rejects_missing_close():
    data = pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="5min",
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.0, 1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="close"):
        app_live_signal_trend_risk.build_signal_trend_risk_snapshot(
            data
        )


def test_build_candlestick_chart_contains_ohlc_data_and_risk_lines():
    data = make_market_data(
        [100.0, 101.0, 102.0]
    )

    snapshot = {
        "entry_price": 102.0,
        "stop_loss": 100.98,
        "take_profit": 104.04,
    }

    figure = (
        app_live_signal_trend_risk
        .build_candlestick_chart(
            data,
            snapshot=snapshot,
        )
    )

    assert len(figure.data) == 1

    trace = figure.data[0]

    assert trace.type == "candlestick"
    assert list(trace.open) == [100.0, 101.0, 102.0]
    assert list(trace.high) == [102.0, 103.0, 104.0]
    assert list(trace.low) == [98.0, 99.0, 100.0]
    assert list(trace.close) == [100.0, 101.0, 102.0]

    assert len(figure.layout.shapes) == 3


def test_build_candlestick_chart_rejects_missing_columns():
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Missing required chart columns",
    ):
        app_live_signal_trend_risk.build_candlestick_chart(data)


def test_build_candlestick_chart_rejects_empty_data():
    data = pd.DataFrame(
        columns=[
            "openTime",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    with pytest.raises(
        ValueError,
        match="data must not be empty",
    ):
        app_live_signal_trend_risk.build_candlestick_chart(data)
