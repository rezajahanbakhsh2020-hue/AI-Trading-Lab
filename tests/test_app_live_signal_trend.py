from __future__ import annotations

import pandas as pd
import pytest

import app_live_signal_trend


def make_market_data(
    closes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-09-09T08:00:00Z",
                periods=len(closes),
                freq="5min",
            ),
            "open": closes,
            "high": [price + 2.0 for price in closes],
            "low": [price - 2.0 for price in closes],
            "close": closes,
        }
    )


def test_build_signal_trend_snapshot_combines_signal_and_uptrend():
    data = make_market_data(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    snapshot = app_live_signal_trend.build_signal_trend_snapshot(
        data,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
    )

    assert snapshot["signal"] == 1
    assert snapshot["signal_label"] == "BUY"
    assert snapshot["trend"] == "UP"
    assert snapshot["close"] == 6.0
    assert snapshot["momentum"] == pytest.approx(
        (6.0 / 5.0) - 1.0
    )
    assert snapshot["fast_ma"] == pytest.approx(5.5)
    assert snapshot["slow_ma"] == pytest.approx(4.5)
    assert snapshot["strategy"] == "momentum"
    assert snapshot["momentum_window"] == 2
    assert snapshot["fast_window"] == 2
    assert snapshot["slow_window"] == 4
    assert "timestamp" in snapshot


def test_build_signal_trend_snapshot_combines_no_trade_and_downtrend():
    data = make_market_data(
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    )

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
    assert snapshot["momentum"] == pytest.approx(
        (1.0 / 2.0) - 1.0
    )


def test_build_signal_trend_snapshot_handles_insufficient_trend_history():
    data = make_market_data(
        [100.0, 101.0, 102.0]
    )

    snapshot = app_live_signal_trend.build_signal_trend_snapshot(
        data,
        momentum_window=2,
        fast_window=2,
        slow_window=10,
    )

    assert snapshot["signal"] == 1
    assert snapshot["signal_label"] == "BUY"
    assert snapshot["trend"] == "INSUFFICIENT DATA"
    assert snapshot["fast_ma"] == pytest.approx(101.5)
    assert snapshot["slow_ma"] is None


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


def test_build_candlestick_chart_rejects_missing_columns():
    data = pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-09-09T08:00:00Z",
                periods=2,
                freq="5min",
            ),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "close": [101.0, 102.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required chart columns",
    ):
        app_live_signal_trend.build_candlestick_chart(data)
