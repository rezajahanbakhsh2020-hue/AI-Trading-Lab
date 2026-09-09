import pandas as pd
import plotly.graph_objects as go

from src.visualization.live_visual_suite import (
    build_complete_live_visual_output,
    build_live_visual_suite,
)


def sample_candles():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=40,
                freq="5min",
            ),
            "open": [
                4390.0 + i
                for i in range(40)
            ],
            "high": [
                4391.0 + i
                for i in range(40)
            ],
            "low": [
                4389.0 + i
                for i in range(40)
            ],
            "close": [
                4390.5 + i
                for i in range(40)
            ],
        }
    )


def sample_snapshot():
    return {
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 4429.5,
        "stop_loss": 4385.0,
        "take_profit": 4518.0,
        "tp1": 4518.0,
        "tp2": 4560.0,
        "tp3": 4600.0,
        "market_state": "OPEN",
        "quote_stale": False,
        "candle_count": 40,
        "timestamp": "2025-01-01T03:15:00+00:00",
    }


def test_live_visual_suite_returns_plotly_figure():
    figure = build_live_visual_suite(
        sample_candles(),
        sample_snapshot(),
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    assert len(figure.data) > 0


def test_live_visual_suite_contains_complete_decision_text():
    figure = build_live_visual_suite(
        sample_candles(),
        sample_snapshot(),
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "BUY" in html
    assert "UP" in html
    assert "momentum" in html
    assert "OPEN" in html
    assert "4429.50" in html
    assert "4385.00" in html
    assert "4518.00" in html
    assert "4560.00" in html
    assert "4600.00" in html


def test_complete_live_visual_output_creates_html(tmp_path):
    output_path = (
        tmp_path
        / "complete_live_visual.html"
    )

    result = build_complete_live_visual_output(
        sample_candles(),
        sample_snapshot(),
        output_path,
    )

    assert result == str(output_path)
    assert output_path.exists()

    html = output_path.read_text(
        encoding="utf-8"
    )

    assert "AI-Trading-Lab" in html
    assert "XAUUSD" in html
    assert "Live Visual Suite" in html
    assert "Complete Decision Board" in html
    assert "BUY" in html
    assert "TP1" in html
    assert "TP2" in html
    assert "TP3" in html


def test_no_trade_visual_suite_is_supported():
    snapshot = sample_snapshot()

    snapshot.update(
        {
            "signal": 0,
            "signal_label": "NO TRADE",
            "stop_loss": None,
            "take_profit": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
        }
    )

    figure = build_live_visual_suite(
        sample_candles(),
        snapshot,
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "NO TRADE" in html
    assert "WAIT" in html


def test_stale_quote_is_visible():
    snapshot = sample_snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_visual_suite(
        sample_candles(),
        snapshot,
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "STALE" in html
