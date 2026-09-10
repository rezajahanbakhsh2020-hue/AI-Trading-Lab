from __future__ import annotations

import pytest

from src.visualization.live_visual_performance import (
    build_live_visual_performance,
    build_live_visual_performance_html,
)


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


def test_build_live_visual_performance_contains_stability_trace():
    figure = build_live_visual_performance(_history())

    assert len(figure.data) == 1
    assert figure.data[0].type == "scatter"
    assert figure.data[0].name == "Stability Score"


def test_build_live_visual_performance_contains_scores():
    figure = build_live_visual_performance(_history())

    assert list(figure.data[0].y) == [0.50, 0.60, 0.70]


def test_build_live_visual_performance_contains_signal_and_trend():
    figure = build_live_visual_performance(_history())

    customdata = list(figure.data[0].customdata)

    assert customdata[0][1] == "BUY"
    assert customdata[0][2] == "UP"
    assert customdata[1][1] == "NO TRADE"
    assert customdata[1][2] == "FLAT"


def test_build_live_visual_performance_has_expected_title():
    figure = build_live_visual_performance(_history())

    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Visual Performance"
    )


def test_build_live_visual_performance_skips_missing_scores():
    history = _history()
    history.insert(
        1,
        {
            "timestamp": 1735689750,
            "signal": "BUY",
            "trend": "UP",
        },
    )

    figure = build_live_visual_performance(history)

    assert list(figure.data[0].y) == [0.50, 0.60, 0.70]


def test_build_live_visual_performance_handles_empty_history():
    figure = build_live_visual_performance([])

    assert len(figure.data) == 0


def test_build_live_visual_performance_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_visual_performance({})


def test_build_live_visual_performance_rejects_invalid_item():
    with pytest.raises(TypeError):
        build_live_visual_performance(
            _history() + [[]]
        )


def test_build_live_visual_performance_html_is_generated():
    html = build_live_visual_performance_html(_history())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "Live Visual Performance" in html
    assert "Stability Score" in html
