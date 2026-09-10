from __future__ import annotations

import pytest

from src.visualization.live_visual_center import (
    build_live_visual_center,
    build_live_visual_center_html,
)


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "market_state": "OPEN",
        "quote_stale": False,
        "quote_age_seconds": 2.0,
        "candle_count": 200,
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
    }


def test_build_live_visual_center_contains_three_tables():
    figure = build_live_visual_center(_snapshot())

    assert len(figure.data) == 3
    assert all(trace.type == "table" for trace in figure.data)


def test_build_live_visual_center_has_expected_title():
    figure = build_live_visual_center(_snapshot())

    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Visual Center"
    )


def test_build_live_visual_center_contains_decision_values():
    figure = build_live_visual_center(_snapshot())

    all_values = [
        value
        for trace in figure.data
        for column in trace.cells.values
        for value in column
    ]

    assert "BUY" in all_values
    assert "UP" in all_values
    assert "momentum" in all_values
    assert "FRESH" in all_values


def test_build_live_visual_center_contains_risk_levels():
    figure = build_live_visual_center(_snapshot())

    all_values = [
        value
        for trace in figure.data
        for column in trace.cells.values
        for value in column
    ]

    assert "3000.0000" in all_values
    assert "2980.0000" in all_values
    assert "3040.0000" in all_values
    assert "3060.0000" in all_values
    assert "3080.0000" in all_values


def test_build_live_visual_center_contains_health_status():
    figure = build_live_visual_center(_snapshot())

    all_values = [
        value
        for trace in figure.data
        for column in trace.cells.values
        for value in column
    ]

    assert "OK" in all_values


def test_build_live_visual_center_accepts_history():
    history = [
        {"signal": "BUY"},
        {"signal": "NO TRADE"},
    ]

    figure = build_live_visual_center(
        _snapshot(),
        history=history,
    )

    assert len(figure.data) == 3


def test_build_live_visual_center_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_center([])


def test_build_live_visual_center_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_visual_center(
            _snapshot(),
            history=123,
        )


def test_build_live_visual_center_html_is_generated():
    html = build_live_visual_center_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "Live Visual Center" in html
