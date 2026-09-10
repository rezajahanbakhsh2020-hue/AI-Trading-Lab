import pandas as pd
import plotly.graph_objects as go

from src.visualization.live_trade_dashboard import (
    build_live_trade_dashboard,
)


def market_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=3,
                freq="D",
            ),
            "open": [2600.0, 2610.0, 2620.0],
            "high": [2620.0, 2630.0, 2640.0],
            "low": [2590.0, 2600.0, 2610.0],
            "close": [2610.0, 2620.0, 2630.0],
        }
    )


def live_result():
    return {
        "trade_display": {
            "decision": "BUY",
            "stable_strategy": "test_strategy",
            "stability_score": 0.85,
            "signal_label": "BUY",
            "trend": "BULLISH",
            "entry_price": 2620.0,
            "stop_loss": 2600.0,
            "tp1": 2640.0,
            "tp2": 2660.0,
            "tp3": 2680.0,
        }
    }


def test_build_live_trade_dashboard_returns_figure():
    figure = build_live_trade_dashboard(
        market_data(),
        live_result(),
    )

    assert isinstance(figure, go.Figure)


def test_dashboard_contains_market_chart():
    figure = build_live_trade_dashboard(
        market_data(),
        live_result(),
    )

    assert any(
        trace.type == "candlestick"
        for trace in figure.data
    )


def test_dashboard_contains_trade_levels():
    figure = build_live_trade_dashboard(
        market_data(),
        live_result(),
    )

    assert len(figure.layout.shapes) == 5


def test_dashboard_accepts_direct_trade_display():
    result = live_result()["trade_display"]

    figure = build_live_trade_dashboard(
        market_data(),
        result,
    )

    assert isinstance(figure, go.Figure)
