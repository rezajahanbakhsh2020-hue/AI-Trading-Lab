from __future__ import annotations

import pytest

from src.visualization.live_visual_snapshot import (
    build_live_visual_snapshot,
    build_live_visual_snapshot_html,
)


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "quote_stale": False,
    }


def test_build_live_visual_snapshot_creates_table():
    figure = build_live_visual_snapshot(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Visual Snapshot"
    )


def test_build_live_visual_snapshot_contains_live_state():
    figure = build_live_visual_snapshot(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "XAU/USD" in values
    assert "5m" in values
    assert "BUY" in values
    assert "UP" in values
    assert "momentum" in values
    assert "FRESH" in values


def test_build_live_visual_snapshot_shows_stale_quote():
    snapshot = _snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_visual_snapshot(snapshot)

    values = figure.data[0].cells.values[1]

    assert "STALE" in values


def test_build_live_visual_snapshot_handles_unknown_quote_state():
    snapshot = _snapshot()
    snapshot.pop("quote_stale")

    figure = build_live_visual_snapshot(snapshot)

    values = figure.data[0].cells.values[1]

    assert "UNKNOWN" in values


def test_build_live_visual_snapshot_uses_signal_fallback():
    snapshot = _snapshot()
    snapshot.pop("signal_label")

    figure = build_live_visual_snapshot(snapshot)

    values = figure.data[0].cells.values[1]

    assert "BUY" in values


def test_build_live_visual_snapshot_handles_missing_values():
    figure = build_live_visual_snapshot({})

    values = figure.data[0].cells.values[1]

    assert values[0] == "N/A"
    assert values[1] == "N/A"
    assert values[2] == "NO TRADE"
    assert values[3] == "INSUFFICIENT DATA"
    assert values[4] == "N/A"
    assert values[5] == "UNKNOWN"


def test_build_live_visual_snapshot_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_snapshot([])


def test_build_live_visual_snapshot_html_is_generated():
    html = build_live_visual_snapshot_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE SNAPSHOT" in html
