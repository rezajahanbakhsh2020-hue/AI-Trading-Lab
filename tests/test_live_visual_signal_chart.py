from __future__ import annotations

import pytest

from src.visualization.live_visual_signal_chart import (
    build_live_visual_signal_chart,
    build_live_visual_signal_chart_html,
)


def _candles() -> list[dict]:
    return [
        {
            "timestamp": 1735689600,
            "open": 2990.0,
            "high": 3005.0,
            "low": 2985.0,
            "close": 3000.0,
        },
        {
            "timestamp": 1735689900,
            "open": 3000.0,
            "high": 3010.0,
            "low": 2995.0,
            "close": 3005.0,
        },
    ]


def _snapshot() -> dict:
    return {
        "symbol": "XAU/USD",
        "interval": "5m",
        "signal": "BUY",
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 3000.0,
        "stop_loss": 2980.0,
        "take_profit": 3040.0,
        "tp2": 3060.0,
        "tp3": 3080.0,
    }


def test_build_live_visual_signal_chart_contains_candles():
    figure = build_live_visual_signal_chart(
        _candles(),
        _snapshot(),
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"


def test_build_live_visual_signal_chart_contains_risk_levels():
    figure = build_live_visual_signal_chart(
        _candles(),
        _snapshot(),
    )

    shapes = list(figure.layout.shapes)

    assert len(shapes) == 5

    y_values = [shape.y0 for shape in shapes]

    assert 3000.0 in y_values
    assert 2980.0 in y_values
    assert 3040.0 in y_values
    assert 3060.0 in y_values
    assert 3080.0 in y_values


def test_build_live_visual_signal_chart_has_expected_title():
    figure = build_live_visual_signal_chart(
        _candles(),
        _snapshot(),
    )

    assert "AI-Trading-Lab — Live Signal Chart" in figure.layout.title.text
    assert "BUY" in figure.layout.title.text
    assert "UP" in figure.layout.title.text


def test_build_live_visual_signal_chart_omits_missing_targets():
    snapshot = _snapshot()
    snapshot.pop("tp2")
    snapshot.pop("tp3")

    figure = build_live_visual_signal_chart(
        _candles(),
        snapshot,
    )

    shapes = list(figure.layout.shapes)

    assert len(shapes) == 3

    y_values = [shape.y0 for shape in shapes]

    assert 3000.0 in y_values
    assert 2980.0 in y_values
    assert 3040.0 in y_values
    assert 3060.0 not in y_values
    assert 3080.0 not in y_values


def test_build_live_visual_signal_chart_rejects_invalid_candle_values():
    candles = _candles()
    candles.append(
        {
            "timestamp": 1735690200,
            "open": None,
            "high": 3020.0,
            "low": 3000.0,
            "close": 3015.0,
        }
    )

    with pytest.raises(TypeError):
        build_live_visual_signal_chart(
            candles,
            _snapshot(),
        )


def test_build_live_visual_signal_chart_accepts_open_time():
    candles = [
        {
            "openTime": 1735689600,
            "open": 2990.0,
            "high": 3005.0,
            "low": 2985.0,
            "close": 3000.0,
        }
    ]

    figure = build_live_visual_signal_chart(
        candles,
        _snapshot(),
    )

    assert len(figure.data) == 1
    assert figure.data[0].x[0] == 1735689600


def test_build_live_visual_signal_chart_rejects_invalid_candles():
    with pytest.raises(TypeError):
        build_live_visual_signal_chart(
            {},
            _snapshot(),
        )


def test_build_live_visual_signal_chart_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        build_live_visual_signal_chart(
            _candles(),
            [],
        )


def test_build_live_visual_signal_chart_rejects_invalid_candle():
    with pytest.raises(TypeError):
        build_live_visual_signal_chart(
            [{}],
            _snapshot(),
        )


def test_build_live_visual_signal_chart_html_is_generated():
    html = build_live_visual_signal_chart_html(
        _candles(),
        _snapshot(),
    )

    assert "plotly" in html.lower()
    assert "AI-Trading-Lab" in html
    assert "Live Signal Chart" in html
    assert "BUY" in html
