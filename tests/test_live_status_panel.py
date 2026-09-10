from __future__ import annotations

import pytest

from src.visualization.live_status_panel import (
    build_live_status_panel,
    build_live_status_panel_html,
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
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
        "stability_score": 0.517268,
    }


def test_build_live_status_panel_creates_table():
    figure = build_live_status_panel(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == "AI-Trading-Lab — Live Status"


def test_build_live_status_panel_contains_core_status_values():
    figure = build_live_status_panel(_snapshot())

    table = figure.data[0]

    assert "XAU/USD" in table.cells.values[1]
    assert "5m" in table.cells.values[1]
    assert "BUY" in table.cells.values[1]
    assert "UP" in table.cells.values[1]
    assert "momentum" in table.cells.values[1]
    assert "OPEN" in table.cells.values[1]


def test_build_live_status_panel_contains_risk_levels():
    figure = build_live_status_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "3000.0000" in values
    assert "2980.0000" in values
    assert "3040.0000" in values
    assert "3060.0000" in values
    assert "3080.0000" in values


def test_build_live_status_panel_shows_fresh_quote():
    figure = build_live_status_panel(_snapshot())

    assert "FRESH" in figure.data[0].cells.values[1]


def test_build_live_status_panel_shows_stale_quote():
    snapshot = _snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_status_panel(snapshot)

    assert "STALE" in figure.data[0].cells.values[1]


def test_build_live_status_panel_supports_missing_optional_values():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")
    snapshot.pop("stability_score")

    figure = build_live_status_panel(snapshot)

    values = figure.data[0].cells.values[1]

    assert values.count("N/A") >= 3


def test_build_live_status_panel_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_status_panel([])


def test_build_live_status_panel_html_is_generated():
    html = build_live_status_panel_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE STATUS" in html
    assert "XAU/USD" in html
