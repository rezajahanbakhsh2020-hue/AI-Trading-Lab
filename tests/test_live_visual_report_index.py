from __future__ import annotations

import pytest

from src.visualization.live_visual_report_index import (
    build_live_visual_report_index,
    build_live_visual_report_index_html,
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
        "quote_age_seconds": 2.0,
        "candle_count": 200,
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
    }


def test_build_live_visual_report_index_returns_html():
    html = build_live_visual_report_index(_snapshot())

    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html


def test_build_live_visual_report_index_contains_visual_center():
    html = build_live_visual_report_index(_snapshot())

    assert "AI-Trading-Lab — Live Visual Center" in html
    assert "Live Report" in html
    assert 'id="live-report"' in html


def test_build_live_visual_report_index_contains_symbol_and_interval():
    html = build_live_visual_report_index(_snapshot())

    assert "XAU/USD" in html
    assert "5m" in html


def test_build_live_visual_report_index_preserves_live_report():
    html = build_live_visual_report_index(_snapshot())

    assert "AI-Trading-Lab — Live Visual Report" in html
    assert "LIVE DECISION" in html
    assert "LIVE DATA QUALITY" in html


def test_build_live_visual_report_index_preserves_history():
    history = [
        {"signal": "BUY"},
        {"signal": "NO TRADE"},
        {"signal": "BUY"},
    ]

    html = build_live_visual_report_index(
        _snapshot(),
        history=history,
    )

    assert "History snapshots: 3" in html


def test_build_live_visual_report_index_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_report_index([])


def test_build_live_visual_report_index_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_visual_report_index(
            _snapshot(),
            history=123,
        )


def test_build_live_visual_report_index_html_helper():
    html = build_live_visual_report_index_html(_snapshot())

    assert "AI-Trading-Lab — Live Visual Center" in html
    assert "plotly" in html.lower()
