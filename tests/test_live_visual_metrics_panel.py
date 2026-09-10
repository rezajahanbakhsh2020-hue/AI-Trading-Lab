from __future__ import annotations

import pytest

from src.visualization.live_visual_metrics_panel import (
    build_live_visual_metrics_panel,
    build_live_visual_metrics_panel_html,
)


def _snapshot() -> dict:
    return {
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.517268,
    }


def test_build_live_visual_metrics_panel_contains_metrics_table():
    figure = build_live_visual_metrics_panel(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"


def test_build_live_visual_metrics_panel_has_expected_title():
    figure = build_live_visual_metrics_panel(_snapshot())

    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Metrics Panel"
    )


def test_build_live_visual_metrics_panel_preserves_metric_values():
    figure = build_live_visual_metrics_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "3000.0000" in values
    assert "2980.0000" in values
    assert "3040.0000" in values
    assert "3060.0000" in values
    assert "3080.0000" in values
    assert "2.0000" in values
    assert "0.5173" in values


def test_build_live_visual_metrics_panel_preserves_missing_values():
    figure = build_live_visual_metrics_panel({})

    values = figure.data[0].cells.values[1]

    assert all(value == "N/A" for value in values)


def test_build_live_visual_metrics_panel_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_metrics_panel([])


def test_build_live_visual_metrics_panel_html_is_generated():
    html = build_live_visual_metrics_panel_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "Live Metrics Panel" in html
