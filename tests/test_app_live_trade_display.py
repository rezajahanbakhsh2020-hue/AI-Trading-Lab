import pandas as pd
import pytest

from app_live_trade_display import (
    _build_chart,
    _load_data,
)
from src.evaluation.live_trade_display import (
    build_live_trade_display,
)
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


def test_chart_contains_candlestick_trace():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(
        data,
        display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"


def test_buy_chart_contains_five_trade_lines():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(
        data,
        display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    assert len(figure.layout.shapes) == 5


def test_no_trade_chart_has_no_trade_lines():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.20,
    )

    overlay = build_live_trade_overlay(
        data,
        display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    assert len(figure.layout.shapes) == 0


def test_chart_uses_latest_market_timestamp():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(
        data,
        display,
    )

    assert overlay["timestamp"] == data["timestamp"].iloc[-1]


def test_load_data_returns_required_market_columns():
    data = _load_data()

    assert {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }.issubset(data.columns)

    assert not data.empty


def test_load_data_has_numeric_prices():
    data = _load_data()

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):
        assert pd.api.types.is_numeric_dtype(
            data[column]
        )


def test_load_data_has_valid_timestamps():
    data = _load_data()

    assert data["timestamp"].notna().all()


def test_buy_chart_level_order_is_preserved():
    data = _rising_data()

    display = build_live_trade_display(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    overlay = build_live_trade_overlay(
        data,
        display,
    )

    levels = overlay["levels"]

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]
