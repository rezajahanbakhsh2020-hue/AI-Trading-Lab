import pandas as pd
import pytest

from app_live_proof_dashboard import (
    build_human_explanation,
    build_live_proof_chart,
    build_live_proof_snapshot,
)


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=60,
                freq="5min",
                tz="UTC",
            ),
            "open": [2500.0 + i for i in range(60)],
            "high": [2501.0 + i for i in range(60)],
            "low": [2499.0 + i for i in range(60)],
            "close": [2500.5 + i for i in range(60)],
        }
    )


def sample_quote() -> dict:
    return {
        "symbol": "XAUUSD",
        "mid": 2559.5,
        "bid": 2559.4,
        "ask": 2559.6,
        "marketState": "open",
        "stale": False,
        "quoteAgeSeconds": 0,
    }


def test_build_live_proof_snapshot_contains_main_outputs():
    snapshot = build_live_proof_snapshot(
        sample_data(),
        sample_quote(),
    )

    assert snapshot["symbol"] == "XAUUSD"
    assert snapshot["candle_count"] == 60
    assert snapshot["signal_label"] in {
        "BUY",
        "NO TRADE",
    }
    assert snapshot["trend"] in {
        "UP",
        "DOWN",
        "INSUFFICIENT DATA",
    }
    assert snapshot["entry_price"] == 2559.5
    assert snapshot["market_state"] == "OPEN"
    assert snapshot["quote_stale"] is False


def test_build_live_proof_snapshot_preserves_quote_state():
    snapshot = build_live_proof_snapshot(
        sample_data(),
        sample_quote(),
    )

    assert snapshot["bid"] == 2559.4
    assert snapshot["ask"] == 2559.6
    assert snapshot["mid"] == 2559.5
    assert snapshot["quote_age_seconds"] == 0


def test_build_live_proof_snapshot_has_candle_values():
    snapshot = build_live_proof_snapshot(
        sample_data(),
        sample_quote(),
    )

    candle = snapshot["latest_candle"]

    assert candle["open"] == 2559.0
    assert candle["high"] == 2560.0
    assert candle["low"] == 2558.0
    assert candle["close"] == 2559.5


def test_build_human_explanation_for_no_trade():
    snapshot = {
        "signal_label": "NO TRADE",
        "trend": "UP",
        "entry_price": 2500.0,
        "stop_loss": None,
        "take_profit": None,
    }

    text = build_human_explanation(snapshot)

    assert "NO TRADE" in text
    assert "UP" in text
    assert "SL/TP" in text


def test_build_human_explanation_for_buy():
    snapshot = {
        "signal_label": "BUY",
        "trend": "UP",
        "entry_price": 2500.0,
        "stop_loss": 2475.0,
        "take_profit": 2550.0,
    }

    text = build_human_explanation(snapshot)

    assert "BUY" in text
    assert "UP" in text
    assert "2475.00" in text
    assert "2550.00" in text


def test_build_live_proof_chart_contains_candles():
    data = sample_data()

    snapshot = {
        "entry_price": 2559.5,
        "stop_loss": None,
        "take_profit": None,
        "mid": 2559.5,
    }

    figure = build_live_proof_chart(
        data,
        snapshot,
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"


def test_build_live_proof_chart_adds_risk_levels():
    data = sample_data()

    snapshot = {
        "entry_price": 2559.5,
        "stop_loss": 2533.905,
        "take_profit": 2610.69,
        "mid": 2559.5,
    }

    figure = build_live_proof_chart(
        data,
        snapshot,
    )

    assert len(figure.data) == 1
    assert len(figure.layout.shapes) == 4


def test_build_live_proof_snapshot_rejects_empty_data():
    with pytest.raises(ValueError):
        build_live_proof_snapshot(
            pd.DataFrame(),
            sample_quote(),
        )


def test_build_live_proof_snapshot_rejects_invalid_quote():
    with pytest.raises(ValueError):
        build_live_proof_snapshot(
            sample_data(),
            [],
        )
