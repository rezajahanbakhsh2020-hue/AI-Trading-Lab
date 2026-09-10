from __future__ import annotations

import pytest

from src.visualization.live_visual_final_view import (
    build_live_visual_final_view,
    build_live_visual_final_view_html,
)


def _candles() -> list[dict]:
    return [
        {
            "timestamp": 1735689600,
            "open": 2990.0,
            "high": 3005.0,
            "low": 2985.0,
            "close": 3000.0,
        },
        {
            "timestamp": 1735689900,
            "open": 3000.0,
            "high": 3010.0,
            "low": 2995.0,
            "close": 3005.0,
        },
    ]


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "marketState": "OPEN",
        "stale": False,
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
        "stability_score": 0.65,
        "candle_count": 2,
    }


def _history() -> list[dict]:
    return [
        {
            "timestamp": 1735689600,
            "signal": "BUY",
            "signal_label": "BUY",
            "trend": "UP",
            "stability_score": 0.50,
        },
        {
            "timestamp": 1735689900,
            "signal": "NO TRADE",
            "signal_label": "NO TRADE",
            "trend": "FLAT",
            "stability_score": 0.60,
        },
        {
            "timestamp": 1735690200,
            "signal": "BUY",
            "signal_label": "BUY",
            "trend": "UP",
            "stability_score": 0.70,
        },
    ]


def test_build_live_visual_final_view_contains_main_visual_components():
    figure = build_live_visual_final_view(
        _candles(),
        _snapshot(),
        history=_history(),
    )

    assert len(figure.data) >= 5

    trace_types = {trace.type for trace in figure.data}

    assert "candlestick" in trace_types
    assert "scatter" in trace_types
    assert "table" in trace_types


def test_build_live_visual_final_view_contains_risk_levels():
    figure = build_live_visual_final_view(
        _candles(),
        _snapshot(),
        history=_history(),
    )

    shapes = list(figure.layout.shapes)

    assert len(shapes) == 5

    y_values = [shape.y0 for shape in shapes]

    assert 3000.0 in y_values
    assert 2980.0 in y_values
    assert 3040.0 in y_values
    assert 3060.0 in y_values
    assert 3080.0 in y_values


def test_build_live_visual_final_view_contains_performance_scores():
    figure = build_live_visual_final_view(
        _candles(),
        _snapshot(),
        history=_history(),
    )

    stability_traces = [
        trace
        for trace in figure.data
        if getattr(trace, "name", None) == "Stability Score"
    ]

    assert len(stability_traces) == 1
    assert list(stability_traces[0].y) == [0.50, 0.60, 0.70]


def test_build_live_visual_final_view_has_expected_title():
    figure = build_live_visual_final_view(
        _candles(),
        _snapshot(),
        history=_history(),
    )

    assert "AI-Trading-Lab — Live Visual Final View" in (
        figure.layout.title.text
    )
    assert "BUY" in figure.layout.title.text
    assert "UP" in figure.layout.title.text
    assert "momentum" in figure.layout.title.text


def test_build_live_visual_final_view_works_without_history():
    figure = build_live_visual_final_view(
        _candles(),
        _snapshot(),
    )

    stability_traces = [
        trace
        for trace in figure.data
        if getattr(trace, "name", None) == "Stability Score"
    ]

    assert len(stability_traces) == 0
    assert len(figure.data) >= 4


def test_build_live_visual_final_view_omits_missing_targets():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")

    figure = build_live_visual_final_view(
        _candles(),
        snapshot,
        history=_history(),
    )

    shapes = list(figure.layout.shapes)

    assert len(shapes) == 3

    y_values = [shape.y0 for shape in shapes]

    assert 3000.0 in y_values
    assert 2980.0 in y_values
    assert 3040.0 in y_values
    assert 3060.0 not in y_values
    assert 3080.0 not in y_values


def test_build_live_visual_final_view_rejects_invalid_candles():
    with pytest.raises(TypeError):
        build_live_visual_final_view(
            {},
            _snapshot(),
            history=_history(),
        )


def test_build_live_visual_final_view_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_final_view(
            _candles(),
            [],
            history=_history(),
        )


def test_build_live_visual_final_view_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_visual_final_view(
            _candles(),
            _snapshot(),
            history={},
        )


def test_build_live_visual_final_view_html_is_generated():
    html = build_live_visual_final_view_html(
        _candles(),
        _snapshot(),
        history=_history(),
    )

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "Live Visual Final View" in html
    assert "BUY" in html
    assert "Stability Score" in html
