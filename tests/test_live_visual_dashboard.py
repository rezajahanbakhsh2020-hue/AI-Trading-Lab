from __future__ import annotations

import pandas as pd
import pytest

from src.visualization.live_visual_dashboard import (
    build_live_visual_dashboard,
    save_live_visual_dashboard,
)


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": [1, 2, 3],
            "open": [3000.0, 3005.0, 3010.0],
            "high": [3010.0, 3015.0, 3020.0],
            "low": [2995.0, 3000.0, 3005.0],
            "close": [3005.0, 3010.0, 3015.0],
        }
    )


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "strategy": "momentum",
        "trend": "UP",
        "market_state": "OPEN",
        "quote_stale": False,
        "entry_price": 3015.0,
        "stop_loss": 2995.0,
        "take_profit": 3055.0,
        "timestamp": 3,
    }


def test_build_live_visual_dashboard_contains_current_and_history_sections():
    snapshots = [
        _snapshot(),
        {
            **_snapshot(),
            "timestamp": 4,
            "entry_price": 3020.0,
            "take_profit": 3060.0,
        },
    ]

    html = build_live_visual_dashboard(
        _candles(),
        _snapshot(),
        snapshots,
    )

    assert "<!DOCTYPE html>" in html
    assert "AI-Trading-Lab — Live Visual Dashboard" in html
    assert "Current Live Decision" in html
    assert "Live Decision History" in html
    assert "XAU/USD" in html
    assert "5m" in html
    assert "plotly" in html.lower()


def test_build_live_visual_dashboard_supports_empty_history():
    html = build_live_visual_dashboard(
        _candles(),
        _snapshot(),
        [],
    )

    assert "<!DOCTYPE html>" in html
    assert "Live Decision History" in html
    assert "Current Live Decision" in html


def test_build_live_visual_dashboard_rejects_invalid_candles():
    with pytest.raises(TypeError):
        build_live_visual_dashboard(
            [],
            _snapshot(),
            [],
        )


def test_build_live_visual_dashboard_rejects_empty_candles():
    with pytest.raises(ValueError):
        build_live_visual_dashboard(
            pd.DataFrame(),
            _snapshot(),
            [],
        )


def test_build_live_visual_dashboard_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_dashboard(
            _candles(),
            [],
            [],
        )


def test_build_live_visual_dashboard_rejects_invalid_history():
    with pytest.raises(TypeError):
        build_live_visual_dashboard(
            _candles(),
            _snapshot(),
            "invalid",
        )


def test_save_live_visual_dashboard_writes_html(tmp_path):
    output_path = tmp_path / "live_dashboard.html"

    result = save_live_visual_dashboard(
        _candles(),
        _snapshot(),
        [_snapshot()],
        str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    assert "Current Live Decision" in content
    assert "Live Decision History" in content
