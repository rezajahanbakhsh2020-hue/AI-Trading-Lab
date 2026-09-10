from __future__ import annotations

import pandas as pd
import pytest

from src.visualization.live_signal_timeline import (
    build_live_signal_timeline,
    build_live_signal_timeline_html,
)


def _snapshots() -> list[dict]:
    return [
        {
            "timestamp": 1,
            "signal": "BUY",
            "signal_label": "BUY",
            "trend": "UP",
            "strategy": "momentum",
            "entry_price": 3000.0,
            "stop_loss": 2980.0,
            "take_profit": 3040.0,
            "tp2": 3060.0,
            "tp3": 3080.0,
        },
        {
            "timestamp": 2,
            "signal": "NO TRADE",
            "signal_label": "NO TRADE",
            "trend": "FLAT",
            "strategy": "momentum",
        },
        {
            "timestamp": 3,
            "signal": "BUY",
            "signal_label": "BUY",
            "trend": "DOWN",
            "strategy": "momentum",
            "entry": 3020.0,
            "stop_loss": 2990.0,
            "take_profit": 3060.0,
        },
    ]


def test_build_live_signal_timeline_creates_signal_and_trend_traces():
    figure = build_live_signal_timeline(_snapshots())

    assert len(figure.data) == 2
    assert figure.data[0].name == "Signal"
    assert figure.data[1].name == "Trend"
    assert len(figure.data[0].x) == 3
    assert len(figure.data[1].x) == 3


def test_build_live_signal_timeline_maps_buy_and_no_trade():
    figure = build_live_signal_timeline(_snapshots())

    assert list(figure.data[0].y) == [1, 0, 1]


def test_build_live_signal_timeline_contains_decision_details():
    figure = build_live_signal_timeline(_snapshots())

    first_hover = figure.data[0].text[0]

    assert "BUY" in first_hover
    assert "UP" in first_hover
    assert "momentum" in first_hover
    assert "3000.0" in first_hover
    assert "2980.0" in first_hover
    assert "3040.0" in first_hover
    assert "3060.0" in first_hover
    assert "3080.0" in first_hover


def test_build_live_signal_timeline_supports_entry_fallback():
    figure = build_live_signal_timeline(_snapshots())

    third_hover = figure.data[0].text[2]

    assert "3020.0" in third_hover
    assert "3060.0" in third_hover


def test_build_live_signal_timeline_handles_empty_history():
    figure = build_live_signal_timeline([])

    assert len(figure.data) == 0
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Signal Timeline"
    )


def test_build_live_signal_timeline_normalizes_unknown_states():
    snapshots = [
        {
            "timestamp": 1,
            "signal_label": "UNKNOWN",
            "trend": "BULLISH",
        }
    ]

    figure = build_live_signal_timeline(snapshots)

    assert list(figure.data[0].y) == [0]
    assert "NO TRADE" in figure.data[0].text[0]
    assert "INSUFFICIENT DATA" in figure.data[0].text[0]


def test_build_live_signal_timeline_rejects_invalid_container():
    with pytest.raises(TypeError):
        build_live_signal_timeline("invalid")


def test_build_live_signal_timeline_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_signal_timeline([_snapshots()[0], "invalid"])


def test_build_live_signal_timeline_html_is_responsive():
    html = build_live_signal_timeline_html(_snapshots())

    assert "plotly" in html.lower()
    assert "responsive" in html.lower()
    assert "AI-Trading-Lab" in html


def test_imported_pandas_is_available_for_environment_compatibility():
    assert pd.DataFrame is not None
