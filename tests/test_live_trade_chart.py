import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.live_trade_chart import (
    build_live_trade_chart,
    validate_live_trade_chart,
)


def sample_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=4,
                freq="D",
            ),
            "open": [2600.0, 2610.0, 2620.0, 2630.0],
            "high": [2620.0, 2630.0, 2640.0, 2650.0],
            "low": [2590.0, 2600.0, 2610.0, 2620.0],
            "close": [2610.0, 2620.0, 2630.0, 2640.0],
        }
    )


def sample_trade_display() -> dict:
    return {
        "decision": "BUY",
        "stable_strategy": "test_strategy",
        "stability_score": 0.82,
        "signal_label": "BUY",
        "trend": "BULLISH",
        "entry_price": 2640.0,
        "stop_loss": 2620.0,
        "tp1": 2660.0,
        "tp2": 2680.0,
        "tp3": 2700.0,
    }


def test_build_live_trade_chart_returns_figure():
    figure = build_live_trade_chart(
        sample_market_data(),
        sample_trade_display(),
    )

    assert isinstance(
        figure,
        go.Figure,
    )


def test_live_trade_chart_contains_candlestick():
    figure = build_live_trade_chart(
        sample_market_data(),
        sample_trade_display(),
    )

    assert any(
        trace.type == "candlestick"
        for trace in figure.data
    )


def test_live_trade_chart_contains_trade_levels():
    figure = build_live_trade_chart(
        sample_market_data(),
        sample_trade_display(),
    )

    assert len(figure.layout.shapes) == 5
    assert len(figure.layout.annotations) >= 6


def test_live_trade_chart_supports_no_trade():
    display = sample_trade_display()

    display.update(
        {
            "decision": "NO TRADE",
            "entry_price": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
        }
    )

    figure = build_live_trade_chart(
        sample_market_data(),
        display,
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(figure.layout.shapes) == 0


def test_live_trade_chart_validates():
    figure = build_live_trade_chart(
        sample_market_data(),
        sample_trade_display(),
    )

    assert validate_live_trade_chart(figure) is True


def test_invalid_market_data_is_rejected():
    data = sample_market_data().drop(
        columns=["close"]
    )

    with pytest.raises(ValueError):
        build_live_trade_chart(
            data,
            sample_trade_display(),
        )


def test_invalid_trade_display_is_rejected():
    with pytest.raises(TypeError):
        build_live_trade_chart(
            sample_market_data(),
            None,
        )
