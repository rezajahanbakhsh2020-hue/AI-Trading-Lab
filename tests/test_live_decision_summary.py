from __future__ import annotations

import pytest

from src.visualization.live_decision_summary import (
    build_live_decision_summary,
    build_live_decision_summary_html,
)


def _snapshot() -> dict:
    return {
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "quote_stale": False,
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
    }


def test_build_live_decision_summary_creates_table():
    figure = build_live_decision_summary(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Decision Summary"
    )


def test_build_live_decision_summary_contains_decision_state():
    figure = build_live_decision_summary(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "BUY" in values
    assert "UP" in values
    assert "momentum" in values
    assert "FRESH" in values


def test_build_live_decision_summary_contains_risk_levels():
    figure = build_live_decision_summary(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "3000.0000" in values
    assert "2980.0000" in values
    assert "3040.0000" in values
    assert "3060.0000" in values
    assert "3080.0000" in values


def test_build_live_decision_summary_uses_take_profit_as_tp1():
    snapshot = _snapshot()
    snapshot.pop("tp1", None)

    figure = build_live_decision_summary(snapshot)

    values = figure.data[0].cells.values[1]

    assert "3040.0000" in values


def test_build_live_decision_summary_keeps_missing_targets_as_na():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")

    figure = build_live_decision_summary(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[7] == "N/A"
    assert values[8] == "N/A"


def test_build_live_decision_summary_shows_stale_quote():
    snapshot = _snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_decision_summary(snapshot)

    values = figure.data[0].cells.values[1]

    assert "STALE" in values


def test_build_live_decision_summary_handles_missing_values():
    figure = build_live_decision_summary(
        {
            "signal": "NO TRADE",
            "trend": "INSUFFICIENT DATA",
        }
    )

    values = figure.data[0].cells.values[1]

    assert "NO TRADE" in values
    assert "INSUFFICIENT DATA" in values
    assert values[2] == "N/A"
    assert values[4] == "N/A"
    assert values[5] == "N/A"


def test_build_live_decision_summary_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_decision_summary([])


def test_build_live_decision_summary_html_is_generated():
    html = build_live_decision_summary_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE DECISION" in html
