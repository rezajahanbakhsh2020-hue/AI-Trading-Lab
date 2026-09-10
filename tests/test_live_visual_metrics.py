from __future__ import annotations

import pytest

from src.visualization.live_visual_metrics import (
    build_live_visual_metrics,
    build_live_visual_metrics_html,
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


def test_build_live_visual_metrics_creates_table():
    figure = build_live_visual_metrics(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Visual Metrics"
    )


def test_build_live_visual_metrics_contains_risk_values():
    figure = build_live_visual_metrics(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "3000.0000" in values
    assert "2980.0000" in values
    assert "3040.0000" in values
    assert "3060.0000" in values
    assert "3080.0000" in values


def test_build_live_visual_metrics_contains_ratio_and_stability():
    figure = build_live_visual_metrics(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "2.0000" in values
    assert "0.5173" in values


def test_build_live_visual_metrics_uses_take_profit_as_tp1():
    snapshot = _snapshot()
    snapshot.pop("tp1", None)

    figure = build_live_visual_metrics(snapshot)

    values = figure.data[0].cells.values[1]

    assert "3040.0000" in values


def test_build_live_visual_metrics_keeps_missing_targets_as_na():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")

    figure = build_live_visual_metrics(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[3] == "N/A"
    assert values[4] == "N/A"


def test_build_live_visual_metrics_handles_missing_values():
    figure = build_live_visual_metrics({})

    values = figure.data[0].cells.values[1]

    assert all(value == "N/A" for value in values)


def test_build_live_visual_metrics_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_metrics([])


def test_build_live_visual_metrics_html_is_generated():
    html = build_live_visual_metrics_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE METRICS" in html
