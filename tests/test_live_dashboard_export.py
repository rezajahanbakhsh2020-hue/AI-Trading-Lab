from __future__ import annotations

import pytest

from src.visualization.live_dashboard_export import (
    build_live_dashboard_export,
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


def test_build_live_dashboard_export_returns_html():
    html = build_live_dashboard_export(_snapshot())

    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html


def test_build_live_dashboard_export_contains_core_sections():
    html = build_live_dashboard_export(_snapshot())

    assert "Live decision summary" in html
    assert "Live data quality" in html
    assert "AI-Trading-Lab" in html


def test_build_live_dashboard_export_contains_history_count():
    history = [
        {"signal": "BUY"},
        {"signal": "NO TRADE"},
        {"signal": "BUY"},
    ]

    html = build_live_dashboard_export(
        _snapshot(),
        history=history,
    )

    assert "History snapshots: 3" in html


def test_build_live_dashboard_export_handles_no_history():
    html = build_live_dashboard_export(
        _snapshot(),
        history=None,
    )

    assert "History snapshots: 0" in html


def test_build_live_dashboard_export_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_dashboard_export([])


def test_build_live_dashboard_export_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_dashboard_export(
            _snapshot(),
            history=123,
        )


def test_build_live_dashboard_export_contains_decision_values():
    html = build_live_dashboard_export(_snapshot())

    assert "BUY" in html
    assert "UP" in html
    assert "momentum" in html


def test_build_live_dashboard_export_contains_risk_values():
    html = build_live_dashboard_export(_snapshot())

    assert "3000.0000" in html
    assert "2980.0000" in html
    assert "3040.0000" in html
    assert "3060.0000" in html
    assert "3080.0000" in html
