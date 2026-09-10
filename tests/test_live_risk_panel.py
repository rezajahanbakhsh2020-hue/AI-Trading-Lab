from __future__ import annotations

import pytest

from src.visualization.live_risk_panel import (
    build_live_risk_panel,
    build_live_risk_panel_html,
)


def _snapshot() -> dict:
    return {
        "signal": "BUY",
        "signal_label": "BUY",
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
        "risk_reward_ratio": 2.0,
        "stop_loss_pct": 0.6667,
        "take_profit_pct": 1.3333,
    }


def test_build_live_risk_panel_creates_table():
    figure = build_live_risk_panel(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Risk Levels"
    )


def test_build_live_risk_panel_contains_signal():
    figure = build_live_risk_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "BUY" in values


def test_build_live_risk_panel_contains_entry_and_stop_loss():
    figure = build_live_risk_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "3000.0000" in values
    assert "2980.0000" in values


def test_build_live_risk_panel_contains_all_available_targets():
    figure = build_live_risk_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "3040.0000" in values
    assert "3060.0000" in values
    assert "3080.0000" in values


def test_build_live_risk_panel_contains_risk_metrics():
    figure = build_live_risk_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "2.0000" in values
    assert "0.6667" in values
    assert "1.3333" in values


def test_build_live_risk_panel_uses_take_profit_as_tp1():
    snapshot = _snapshot()
    snapshot.pop("tp1", None)

    figure = build_live_risk_panel(snapshot)

    values = figure.data[0].cells.values[1]

    assert "3040.0000" in values


def test_build_live_risk_panel_does_not_invent_missing_targets():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")

    figure = build_live_risk_panel(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[4] == "N/A"
    assert values[5] == "N/A"


def test_build_live_risk_panel_supports_missing_risk_metrics():
    snapshot = _snapshot()
    snapshot.pop("risk_reward_ratio")
    snapshot.pop("stop_loss_pct")
    snapshot.pop("take_profit_pct")

    figure = build_live_risk_panel(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[6] == "N/A"
    assert values[7] == "N/A"
    assert values[8] == "N/A"


def test_build_live_risk_panel_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_risk_panel([])


def test_build_live_risk_panel_html_is_generated():
    html = build_live_risk_panel_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "RISK LEVEL" in html
