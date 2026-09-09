from __future__ import annotations

import pandas as pd
import pytest

import live_trend


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


def test_generate_live_trend_detects_uptrend():
    data = make_market_data(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    result = live_trend.generate_live_trend(
        data,
        fast_window=2,
        slow_window=4,
    )

    assert "fast_ma" in result.columns
    assert "slow_ma" in result.columns
    assert "trend" in result.columns

    assert result.iloc[-1]["fast_ma"] > result.iloc[-1]["slow_ma"]
    assert result.iloc[-1]["trend"] == "UP"


def test_generate_live_trend_detects_downtrend():
    data = make_market_data(
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    )

    result = live_trend.generate_live_trend(
        data,
        fast_window=2,
        slow_window=4,
    )

    assert result.iloc[-1]["fast_ma"] < result.iloc[-1]["slow_ma"]
    assert result.iloc[-1]["trend"] == "DOWN"


def test_generate_live_trend_reports_insufficient_data():
    data = make_market_data(
        [100.0, 101.0, 102.0]
    )

    result = live_trend.generate_live_trend(
        data,
        fast_window=2,
        slow_window=5,
    )

    assert result.iloc[-1]["fast_ma"] == pytest.approx(101.5)
    assert pd.isna(result.iloc[-1]["slow_ma"])
    assert result.iloc[-1]["trend"] == "INSUFFICIENT DATA"


def test_generate_live_trend_preserves_input_columns():
    data = make_market_data(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    result = live_trend.generate_live_trend(
        data,
        fast_window=2,
        slow_window=4,
    )

    for column in data.columns:
        assert column in result.columns

    assert len(result) == len(data)


def test_generate_live_trend_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        live_trend.generate_live_trend(
            [1.0, 2.0, 3.0]
        )


def test_generate_live_trend_rejects_empty_dataframe():
    data = pd.DataFrame(
        {"close": []}
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        live_trend.generate_live_trend(data)


def test_generate_live_trend_rejects_missing_close():
    data = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Column 'close' not found",
    ):
        live_trend.generate_live_trend(data)


def test_generate_live_trend_rejects_invalid_window_order():
    data = make_market_data(
        [1.0, 2.0, 3.0, 4.0]
    )

    with pytest.raises(
        ValueError,
        match="fast_window must be smaller than slow_window",
    ):
        live_trend.generate_live_trend(
            data,
            fast_window=5,
            slow_window=2,
        )


def test_build_live_trend_snapshot_returns_uptrend():
    data = make_market_data(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )

    snapshot = live_trend.build_live_trend_snapshot(
        data,
        fast_window=2,
        slow_window=4,
    )

    assert snapshot["trend"] == "UP"
    assert snapshot["fast_window"] == 2
    assert snapshot["slow_window"] == 4
    assert snapshot["close"] == 6.0
    assert snapshot["fast_ma"] == pytest.approx(5.5)
    assert snapshot["slow_ma"] == pytest.approx(4.5)
    assert "timestamp" in snapshot


def test_build_live_trend_snapshot_handles_insufficient_history():
    data = make_market_data(
        [100.0, 101.0, 102.0]
    )

    snapshot = live_trend.build_live_trend_snapshot(
        data,
        fast_window=2,
        slow_window=10,
    )

    assert snapshot["trend"] == "INSUFFICIENT DATA"
    assert snapshot["fast_ma"] == pytest.approx(101.5)
    assert snapshot["slow_ma"] is None
    assert snapshot["close"] == 102.0
