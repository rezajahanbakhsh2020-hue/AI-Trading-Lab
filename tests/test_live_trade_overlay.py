import pandas as pd
import pytest

from src.evaluation.live_trade_display import build_live_trade_display
from src.visualization.live_trade_overlay import (
    build_live_trade_overlay,
)


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
    )

    close = pd.Series(
        [2000.0 + index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def test_buy_overlay_contains_all_trade_lines():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(data, display)

    assert overlay["decision"] == "BUY"

    levels = overlay["levels"]

    assert levels["entry"] is not None
    assert levels["stop_loss"] is not None
    assert levels["tp1"] is not None
    assert levels["tp2"] is not None
    assert levels["tp3"] is not None

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]


def test_buy_overlay_has_five_visible_lines():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(data, display)

    assert len(overlay["lines"]) == 5
    assert all(line["visible"] for line in overlay["lines"])

    assert [line["name"] for line in overlay["lines"]] == [
        "Entry",
        "SL",
        "TP1",
        "TP2",
        "TP3",
    ]


def test_no_trade_overlay_hides_all_trade_lines():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.20,
    )

    overlay = build_live_trade_overlay(data, display)

    assert overlay["decision"] == "NO TRADE"

    assert overlay["levels"] == {
        "entry": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "tp3": None,
    }

    assert len(overlay["lines"]) == 5
    assert all(
        not line["visible"]
        for line in overlay["lines"]
    )


def test_overlay_preserves_strategy_and_stability_metadata():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(data, display)

    assert overlay["stable_strategy"] == "momentum"
    assert overlay["stability_score"] == pytest.approx(0.80)
    assert overlay["trend"] == "UP"
    assert overlay["signal_label"] == "BUY"


def test_overlay_uses_latest_data_timestamp():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(data, display)

    assert overlay["timestamp"] == data["timestamp"].iloc[-1]


def test_invalid_decision_is_rejected():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    display["decision"] = "SELL"

    with pytest.raises(
        ValueError,
        match="decision must be BUY or NO TRADE",
    ):
        build_live_trade_overlay(data, display)


def test_missing_timestamp_is_rejected():
    data = _rising_data().drop(columns=["timestamp"])

    display = build_live_trade_display(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    with pytest.raises(
        ValueError,
        match="data must contain timestamp",
    ):
        build_live_trade_overlay(data, display)
