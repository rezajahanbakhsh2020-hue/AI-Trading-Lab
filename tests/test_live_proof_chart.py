from __future__ import annotations

import pandas as pd

from src.visualization.live_proof_chart import (
    build_live_proof_chart,
)


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [
                4400.0,
                4402.0,
                4404.0,
                4406.0,
            ],
            "high": [
                4403.0,
                4405.0,
                4407.0,
                4409.0,
            ],
            "low": [
                4399.0,
                4401.0,
                4403.0,
                4405.0,
            ],
            "close": [
                4402.0,
                4404.0,
                4406.0,
                4408.0,
            ],
        }
    )


def test_live_proof_chart_contains_market_and_signal_context():
    figure = build_live_proof_chart(
        _sample_data(),
        {
            "symbol": "XAUUSD",
            "interval": "5m",
            "signal": 1,
            "signal_label": "BUY",
            "trend": "UP",
            "entry_price": 4408.0,
            "stop_loss": 4363.92,
            "take_profit": 4496.16,
        },
    )

    assert "XAUUSD" in figure.layout.title.text
    assert "BUY" in figure.layout.title.text
    assert "UP" in figure.layout.title.text

    names = {
        trace.name
        for trace in figure.data
    }

    assert "XAUUSD" in names
    assert "Fast MA (20)" in names
    assert "Slow MA (50)" in names
    assert "BUY Signal" in names


def test_live_proof_chart_contains_tp1_tp2_tp3():
    figure = build_live_proof_chart(
        _sample_data(),
        {
            "symbol": "XAUUSD",
            "interval": "5m",
            "signal": 1,
            "signal_label": "BUY",
            "trend": "UP",
            "entry_price": 4408.0,
            "stop_loss": 4363.92,
            "tp1": 4452.08,
            "tp2": 4496.16,
            "tp3": 4540.24,
        },
    )

    annotations = [
        annotation.text
        for annotation in figure.layout.annotations
    ]

    assert "Entry" in annotations
    assert "SL" in annotations
    assert "TP1" in annotations
    assert "TP2" in annotations
    assert "TP3" in annotations


def test_live_proof_chart_supports_legacy_single_take_profit():
    figure = build_live_proof_chart(
        _sample_data(),
        {
            "symbol": "XAUUSD",
            "interval": "5m",
            "signal": 1,
            "signal_label": "BUY",
            "trend": "UP",
            "entry_price": 4408.0,
            "stop_loss": 4363.92,
            "take_profit": 4496.16,
        },
    )

    annotations = [
        annotation.text
        for annotation in figure.layout.annotations
    ]

    assert "Entry" in annotations
    assert "SL" in annotations
    assert "TP" in annotations


def test_live_proof_chart_rejects_missing_ohlc():
    data = pd.DataFrame(
        {
            "open": [4400.0],
            "high": [4402.0],
            "low": [4399.0],
        }
    )

    try:
        build_live_proof_chart(
            data,
            {},
        )
    except ValueError as exc:
        assert "Missing OHLC columns" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing OHLC columns."
        )
