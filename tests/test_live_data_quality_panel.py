from __future__ import annotations

import pytest

from src.visualization.live_data_quality_panel import (
    build_live_data_quality_panel,
    build_live_data_quality_panel_html,
)


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "market_state": "OPEN",
        "quote_stale": False,
        "quote_age_seconds": 2.0,
        "candle_count": 200,
    }


def test_build_live_data_quality_panel_creates_table():
    figure = build_live_data_quality_panel(_snapshot())

    assert len(figure.data) == 1
    assert figure.data[0].type == "table"
    assert figure.layout.title.text == (
        "AI-Trading-Lab — Live Data Quality"
    )


def test_build_live_data_quality_panel_contains_market_data():
    figure = build_live_data_quality_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "XAU/USD" in values
    assert "5m" in values
    assert "OPEN" in values
    assert "FRESH" in values


def test_build_live_data_quality_panel_contains_quote_age_and_candles():
    figure = build_live_data_quality_panel(_snapshot())

    values = figure.data[0].cells.values[1]

    assert "2" in values
    assert "200" in values


def test_build_live_data_quality_panel_shows_stale_quote():
    snapshot = _snapshot()
    snapshot["quote_stale"] = True

    figure = build_live_data_quality_panel(snapshot)

    values = figure.data[0].cells.values[1]

    assert "STALE" in values


def test_build_live_data_quality_panel_uses_candle_sequence_when_count_missing():
    snapshot = _snapshot()
    snapshot.pop("candle_count")
    snapshot["candles"] = [1, 2, 3, 4]

    figure = build_live_data_quality_panel(snapshot)

    values = figure.data[0].cells.values[1]

    assert "4" in values


def test_build_live_data_quality_panel_handles_missing_values():
    figure = build_live_data_quality_panel({})

    values = figure.data[0].cells.values[1]

    assert values[0] == "N/A"
    assert values[1] == "N/A"
    assert values[2] == "N/A"
    assert values[3] == "UNKNOWN"
    assert values[4] == "N/A"
    assert values[5] == "N/A"


def test_build_live_data_quality_panel_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_data_quality_panel([])


def test_build_live_data_quality_panel_html_is_generated():
    html = build_live_data_quality_panel_html(_snapshot())

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "LIVE DATA QUALITY" in html
