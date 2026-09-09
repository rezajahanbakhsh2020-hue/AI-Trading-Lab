from __future__ import annotations

import pandas as pd
import pytest

import app_live_signal


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


def test_build_signal_chart_creates_candlestick():
    data = make_market_data(
        [100.0, 101.0, 102.0]
    )

    figure = app_live_signal.build_signal_chart(data)

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"
    assert len(figure.data[0].x) == 3


def test_signal_description_for_buy():
    snapshot = {
        "signal_label": "BUY",
        "momentum": 0.0125,
        "close": 3500.0,
    }

    description = app_live_signal.signal_description(
        snapshot
    )

    assert "positive" in description
    assert "BUY" in description
    assert "3,500.00" in description


def test_signal_description_for_no_trade():
    snapshot = {
        "signal_label": "NO TRADE",
        "momentum": -0.004,
        "close": 3500.0,
    }

    description = app_live_signal.signal_description(
        snapshot
    )

    assert "non-positive" in description
    assert "NO TRADE" in description
    assert "3,500.00" in description


def test_signal_description_for_insufficient_history():
    snapshot = {
        "signal_label": "NO TRADE",
        "momentum": None,
        "close": 3500.0,
    }

    description = app_live_signal.signal_description(
        snapshot
    )

    assert "not enough historical data" in description
    assert "3,500.00" in description


def test_dashboard_signal_uses_existing_strategy_snapshot():
    data = make_market_data(
        [100.0, 101.0, 102.0, 104.0]
    )

    snapshot = app_live_signal.build_live_signal_snapshot(
        data=data,
        window=2,
    )

    assert snapshot["strategy"] == "momentum"
    assert snapshot["signal_label"] == "BUY"
    assert snapshot["close"] == 104.0
    assert snapshot["momentum"] == pytest.approx(
        (104.0 / 102.0) - 1.0
    )


def test_dashboard_signal_handles_no_trade():
    data = make_market_data(
        [100.0, 102.0, 101.0, 99.0]
    )

    snapshot = app_live_signal.build_live_signal_snapshot(
        data=data,
        window=2,
    )

    assert snapshot["signal_label"] == "NO TRADE"
