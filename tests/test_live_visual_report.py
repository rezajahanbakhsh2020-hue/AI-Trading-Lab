from __future__ import annotations

import pytest

from src.visualization.live_visual_report import (
    build_live_visual_report,
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


def test_build_live_visual_report_returns_html():
    html = build_live_visual_report(_snapshot())

    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html


def test_build_live_visual_report_contains_report_header():
    html = build_live_visual_report(_snapshot())

    assert "AI-Trading-Lab — Live Visual Report" in html
    assert "XAU/USD" in html
    assert "5m" in html
    assert "BUY" in html
    assert "UP" in html


def test_build_live_visual_report_contains_dashboard_sections():
    html = build_live_visual_report(_snapshot())

    assert "Live decision summary" in html
    assert "Live data quality" in html
    assert "LIVE DECISION" in html
    assert "LIVE DATA QUALITY" in html


def test_build_live_visual_report_preserves_history_count():
    history = [
        {"signal": "BUY"},
        {"signal": "NO TRADE"},
    ]

    html = build_live_visual_report(
        _snapshot(),
        history=history,
    )

    assert "History snapshots: 2" in html


def test_build_live_visual_report_contains_risk_levels():
    html = build_live_visual_report(_snapshot())

    assert "3000.0000" in html
    assert "2980.0000" in html
    assert "3040.0000" in html
    assert "3060.0000" in html
    assert "3080.0000" in html


def test_build_live_visual_report_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_report([])


def test_build_live_visual_report_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_visual_report(
            _snapshot(),
            history=123,
        )
