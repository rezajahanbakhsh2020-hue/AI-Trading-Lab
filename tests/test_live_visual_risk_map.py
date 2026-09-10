from __future__ import annotations

import pytest

from src.visualization.live_visual_risk_map import (
    build_live_visual_risk_map,
    build_live_visual_risk_map_html,
)


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
    }


def test_build_live_visual_risk_map_contains_all_available_levels():
    figure = build_live_visual_risk_map(_snapshot())

    assert len(figure.data) == 5

    names = [trace.name for trace in figure.data]

    assert names == [
        "Entry",
        "Stop Loss",
        "TP1",
        "TP2",
        "TP3",
    ]


def test_build_live_visual_risk_map_contains_expected_prices():
    figure = build_live_visual_risk_map(_snapshot())

    values = [
        value
        for trace in figure.data
        for value in trace.y
    ]

    assert 3000.0 in values
    assert 2980.0 in values
    assert 3040.0 in values
    assert 3060.0 in values
    assert 3080.0 in values


def test_build_live_visual_risk_map_has_expected_title():
    figure = build_live_visual_risk_map(_snapshot())

    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Visual Risk Map"
    )


def test_build_live_visual_risk_map_omits_missing_targets():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")

    figure = build_live_visual_risk_map(snapshot)

    names = [trace.name for trace in figure.data]

    assert names == [
        "Entry",
        "Stop Loss",
        "TP1",
    ]


def test_build_live_visual_risk_map_supports_take_profit_fallback():
    snapshot = _snapshot()
    snapshot.pop("take_profit")

    snapshot["tp1"] = 3040.0

    figure = build_live_visual_risk_map(snapshot)

    names = [trace.name for trace in figure.data]

    assert "TP1" in names

    tp1_trace = next(
        trace for trace in figure.data
        if trace.name == "TP1"
    )

    assert list(tp1_trace.y) == [3040.0, 3040.0]


def test_build_live_visual_risk_map_supports_entry_fallback():
    snapshot = _snapshot()
    snapshot.pop("entry_price")
    snapshot["entry"] = 3000.0

    figure = build_live_visual_risk_map(snapshot)

    entry_trace = next(
        trace for trace in figure.data
        if trace.name == "Entry"
    )

    assert list(entry_trace.y) == [3000.0, 3000.0]


def test_build_live_visual_risk_map_handles_no_available_levels():
    snapshot = {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "NO TRADE",
    }

    figure = build_live_visual_risk_map(snapshot)

    assert len(figure.data) == 0


def test_build_live_visual_risk_map_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_risk_map([])


def test_build_live_visual_risk_map_html_is_generated():
    html = build_live_visual_risk_map_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "Live Visual Risk Map" in html
    assert "Entry" in html
    assert "Stop Loss" in html
    assert "TP1" in html
