import pandas as pd
import pytest

import app_live_signal_trend


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


def test_build_signal_trend_snapshot_combines_existing_engines():
    data = make_market_data([1, 2, 3, 4, 5, 6])

    snapshot = app_live_signal_trend.build_signal_trend_snapshot(
        data,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
    )

    assert snapshot["signal"] == 1
    assert snapshot["signal_label"] == "BUY"
    assert snapshot["trend"] == "UP"
    assert snapshot["strategy"] == "momentum"
    assert snapshot["momentum_window"] == 2
    assert snapshot["fast_window"] == 2
    assert snapshot["slow_window"] == 4
    assert snapshot["close"] == 6.0
    assert snapshot["momentum"] == pytest.approx((6.0 / 4.0) - 1.0)


def test_build_signal_trend_snapshot_detects_downtrend():
    data = make_market_data([6, 5, 4, 3, 2, 1])

    snapshot = app_live_signal_trend.build_signal_trend_snapshot(
        data,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
    )

    assert snapshot["signal"] == 0
    assert snapshot["signal_label"] == "NO TRADE"
    assert snapshot["trend"] == "DOWN"
    assert snapshot["close"] == 1.0
    assert snapshot["momentum"] == pytest.approx((1.0 / 3.0) - 1.0)


def test_build_signal_trend_snapshot_preserves_timestamp():
    data = make_market_data([1, 2, 3, 4, 5, 6])

    snapshot = app_live_signal_trend.build_signal_trend_snapshot(
        data,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
    )

    expected_timestamp = (
        data["openTime"].iloc[-1]
        .tz_localize("UTC")
        .isoformat()
    )

    assert snapshot["timestamp"] == expected_timestamp


def test_build_signal_trend_snapshot_rejects_empty_data():
    data = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        app_live_signal_trend.build_signal_trend_snapshot(data)


def test_build_signal_trend_snapshot_rejects_missing_close():
    data = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="close"):
        app_live_signal_trend.build_signal_trend_snapshot(data)


def test_build_candlestick_chart_contains_ohlc_data():
    data = make_market_data(
        [100.0, 101.0, 102.0]
    )

    figure = app_live_signal_trend.build_candlestick_chart(
        data
    )

    assert len(figure.data) == 1

    trace = figure.data[0]

    assert trace.type == "candlestick"
    assert list(trace.open) == [100.0, 101.0, 102.0]
    assert list(trace.high) == [102.0, 103.0, 104.0]
    assert list(trace.low) == [98.0, 99.0, 100.0]
    assert list(trace.close) == [100.0, 101.0, 102.0]


def test_build_candlestick_chart_rejects_empty_data():
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Missing required chart columns",
    ):
        app_live_signal_trend.build_candlestick_chart(data)
