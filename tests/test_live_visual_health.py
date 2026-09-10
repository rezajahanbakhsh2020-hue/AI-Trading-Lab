from __future__ import annotations

import pytest

from src.visualization.live_visual_health import (
    build_live_visual_health,
    build_live_visual_health_html,
)


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "market_state": "OPEN",
        "quote_stale": False,
        "candle_count": 200,
    }


def test_build_live_visual_health_creates_table():
    figure = build_live_visual_health(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Visual Health: HEALTHY"
    )


def test_build_live_visual_health_marks_all_checks_ok():
    figure = build_live_visual_health(_snapshot())

    values = figure.data[0].cells.values[1]

    assert all(value == "OK" for value in values)


def test_build_live_visual_health_detects_stale_quote():
    snapshot = _snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_visual_health(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[0] == "CHECK"
    assert "Visual Health: CHECK" in figure.layout.title.text


def test_build_live_visual_health_detects_missing_candles():
    snapshot = _snapshot()
    snapshot["candle_count"] = 0

    figure = build_live_visual_health(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[2] == "CHECK"


def test_build_live_visual_health_uses_candle_sequence():
    snapshot = _snapshot()
    snapshot.pop("candle_count")
    snapshot["candles"] = [1, 2, 3]

    figure = build_live_visual_health(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[2] == "OK"


def test_build_live_visual_health_detects_missing_trend():
    snapshot = _snapshot()
    snapshot["trend"] = "INSUFFICIENT DATA"

    figure = build_live_visual_health(snapshot)

    values = figure.data[0].cells.values[1]

    assert values[3] == "CHECK"


def test_build_live_visual_health_handles_missing_snapshot_values():
    figure = build_live_visual_health({})

    values = figure.data[0].cells.values[1]

    assert values[0] == "CHECK"
    assert values[1] == "CHECK"
    assert values[2] == "CHECK"
    assert values[3] == "CHECK"
    assert values[4] == "OK"


def test_build_live_visual_health_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_health([])


def test_build_live_visual_health_html_is_generated():
    html = build_live_visual_health_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE VISUAL HEALTH" in html
