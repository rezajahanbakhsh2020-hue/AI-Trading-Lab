import pandas as pd
import pytest

from src.visualization.chart_engine import (
    build_candlestick_series,
    build_signal_markers,
    build_volume_series,
)


def market_data():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
            ],
            "open": [103, 101, 102],
            "high": [105, 103, 104],
            "low": [100, 99, 101],
            "close": [104, 102, 103],
            "volume": [300, 100, 200],
            "signal": [1, 0, -1],
        }
    )


def test_candlestick_series():
    result = build_candlestick_series(market_data())

    assert list(result["timestamp"]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
    ]


def test_candlestick_preserves_columns():
    result = build_candlestick_series(market_data())

    assert {
        "open",
        "high",
        "low",
        "close",
    }.issubset(result.columns)


def test_signal_markers():
    result = build_signal_markers(market_data())

    assert len(result) == 2
    assert set(result["marker"]) == {
        "buy",
        "sell",
    }


def test_signal_marker_prices():
    result = build_signal_markers(market_data())

    assert list(result["price"]) == [
        104,
        103,
    ]


def test_volume_series():
    result = build_volume_series(market_data())

    assert list(result["volume"]) == [
        100,
        200,
        300,
    ]


def test_missing_signal():
    data = market_data().drop(columns=["signal"])

    with pytest.raises(ValueError):
        build_signal_markers(data)


def test_missing_volume():
    data = market_data().drop(columns=["volume"])

    with pytest.raises(ValueError):
        build_volume_series(data)


def test_no_modify():
    data = market_data()
    original = data.copy(deep=True)

    build_candlestick_series(data)

    pd.testing.assert_frame_equal(data, original)
