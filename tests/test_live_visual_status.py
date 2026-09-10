from __future__ import annotations

import pytest

from src.visualization.live_visual_status import (
    build_live_visual_status,
    build_live_visual_status_html,
)


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "quote_stale": False,
        "candle_count": 200,
    }


def test_build_live_visual_status_creates_table():
    figure = build_live_visual_status(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Visual Status"
    )


def test_build_live_visual_status_marks_ready_fresh_data():
    figure = build_live_visual_status(_snapshot())

    values = figure.data[0].cells.values[1]

    assert values[0] == "READY"
    assert values[1] == "FRESH"
    assert values[2] == "BUY"
    assert values[3] == "UP"
    assert values[4] == "ACTIONABLE"


def test_build_live_visual_status_marks_stale_quote_for_check():
    snapshot = _snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_visual_status(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[0] == "CHECK"
    assert values[1] == "STALE"


def test_build_live_visual_status_marks_empty_candles_for_check():
    snapshot = _snapshot()
    snapshot["candle_count"] = 0

    figure = build_live_visual_status(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[0] == "CHECK"


def test_build_live_visual_status_uses_candle_sequence():
    snapshot = _snapshot()
    snapshot.pop("candle_count")
    snapshot["candles"] = [1, 2, 3]

    figure = build_live_visual_status(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[0] == "READY"


def test_build_live_visual_status_monitors_non_actionable_state():
    snapshot = _snapshot()
    snapshot["signal"] = "NO TRADE"
    snapshot["signal_label"] = "NO TRADE"

    figure = build_live_visual_status(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[4] == "MONITOR"


def test_build_live_visual_status_handles_missing_values():
    figure = build_live_visual_status({})

    values = figure.data[0].cells.values[1]

    assert values[0] == "CHECK"
    assert values[1] == "UNKNOWN"
    assert values[2] == "NO TRADE"
    assert values[3] == "INSUFFICIENT DATA"
    assert values[4] == "MONITOR"


def test_build_live_visual_status_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_status([])


def test_build_live_visual_status_html_is_generated():
    html = build_live_visual_status_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE VISUAL STATUS" in html
