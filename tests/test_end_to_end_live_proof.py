from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.live_production_decision import (
    build_live_production_decision,
)
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.live_trade_display import (
    build_live_trade_display,
)
from src.visualization.live_trade_overlay import (
    build_live_trade_overlay,
)


def _realistic_rising_market(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="1D",
    )

    close = pd.Series(
        [2000.0 + index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def test_end_to_end_live_proof():
    data = _realistic_rising_market()

    decision = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="1d",
    )

    assert decision["decision"] == "BUY"
    assert decision["trend"] == "UP"
    assert decision["signal_label"] == "BUY"
    assert decision["entry_price"] is not None
    assert decision["stop_loss"] is not None
    assert decision["take_profit"] is not None

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="1d",
    )

    assert display["decision"] == "BUY"
    assert display["entry_price"] == pytest.approx(
        decision["entry_price"]
    )
    assert display["stop_loss"] == pytest.approx(
        decision["stop_loss"]
    )

    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="1d",
    )

    assert runtime.decision["decision"] == "BUY"
    assert runtime.display["decision"] == "BUY"

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["symbol"] == "XAUUSD"
    assert overlay["interval"] == "1d"
    assert overlay["decision"] == "BUY"
    assert overlay["trend"] == "UP"
    assert overlay["stable_strategy"] == "momentum"

    levels = overlay["levels"]

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]

    assert len(overlay["lines"]) == 5
    assert all(
        line["visible"]
        for line in overlay["lines"]
    )


def test_end_to_end_no_trade_proof():
    data = _realistic_rising_market()

    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=0.20,
        symbol="XAUUSD",
        interval="1d",
    )

    assert runtime.decision["decision"] == "NO TRADE"
    assert runtime.display["decision"] == "NO TRADE"

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["decision"] == "NO TRADE"
    assert all(
        value is None
        for value in overlay["levels"].values()
    )
    assert all(
        not line["visible"]
        for line in overlay["lines"]
    )


def test_end_to_end_preserves_latest_market_timestamp():
    data = _realistic_rising_market()

    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="1d",
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["timestamp"] == data["timestamp"].iloc[-1]
