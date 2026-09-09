import plotly.graph_objects as go
import pytest

from src.visualization.live_decision_board import (
    build_live_decision_board,
)


def buy_snapshot():
    return {
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 4395.50,
        "stop_loss": 4351.55,
        "take_profit": 4483.41,
        "tp1": 4483.41,
        "tp2": 4525.00,
        "tp3": 4570.00,
        "market_state": "OPEN",
        "quote_stale": False,
        "candle_count": 201,
        "timestamp": "2025-01-01T00:45:00+00:00",
    }


def test_buy_decision_board_contains_complete_visual_state():
    figure = build_live_decision_board(
        buy_snapshot()
    )

    assert isinstance(
        figure,
        go.Figure,
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "LIVE DECISION" in html
    assert "BUY" in html
    assert "UP" in html
    assert "OPEN" in html
    assert "momentum" in html
    assert "4395.50" in html
    assert "4351.55" in html
    assert "4483.41" in html
    assert "4525.00" in html
    assert "4570.00" in html


def test_no_trade_decision_board_is_visualized():
    snapshot = buy_snapshot()
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

    figure = build_live_decision_board(
        snapshot
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "NO TRADE" in html
    assert "WAIT" in html
    assert "TP1" in html
    assert "TP2" in html
    assert "TP3" in html


def test_stale_quote_state_is_visible():
    snapshot = buy_snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_decision_board(
        snapshot
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "STALE" in html
    assert "quote freshness" in html


def test_insufficient_data_state_is_supported():
    snapshot = buy_snapshot()
    snapshot.update(
        {
            "signal": 0,
            "signal_label": "NO TRADE",
            "trend": "INSUFFICIENT DATA",
        }
    )

    figure = build_live_decision_board(
        snapshot
    )

    html = figure.to_html(
        include_plotlyjs=False
    )

    assert "INSUFFICIENT DATA" in html


def test_invalid_signal_is_rejected():
    snapshot = buy_snapshot()
    snapshot["signal_label"] = "SELL"

    with pytest.raises(ValueError):
        build_live_decision_board(snapshot)


def test_invalid_trend_is_rejected():
    snapshot = buy_snapshot()
    snapshot["trend"] = "UNKNOWN"

    with pytest.raises(ValueError):
        build_live_decision_board(snapshot)


def test_non_mapping_snapshot_is_rejected():
    with pytest.raises(ValueError):
        build_live_decision_board([])
