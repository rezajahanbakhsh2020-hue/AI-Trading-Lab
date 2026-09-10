import pandas as pd
import pytest

from app_live_trade_runtime import (
    INTERVAL,
    STABILITY_SCORE,
    STABLE_STRATEGY,
    SYMBOL,
    _load_runtime_data,
)
from src.evaluation.live_runtime import build_live_runtime
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


def test_runtime_configuration_is_production_consistent():
    assert STABLE_STRATEGY == "momentum"
    assert STABILITY_SCORE == pytest.approx(0.517268)
    assert SYMBOL == "XAUUSD"
    assert INTERVAL == "1d"


def test_runtime_builds_buy_display():
    data = _rising_data()

    runtime = build_live_runtime(
        data,
        stable_strategy=STABLE_STRATEGY,
        stability_score=0.80,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert runtime.decision["decision"] == "BUY"
    assert overlay["decision"] == "BUY"
    assert overlay["levels"]["entry"] is not None
    assert overlay["levels"]["stop_loss"] is not None
    assert overlay["levels"]["tp1"] is not None
    assert overlay["levels"]["tp2"] is not None
    assert overlay["levels"]["tp3"] is not None


def test_runtime_builds_no_trade_display():
    data = _rising_data()

    runtime = build_live_runtime(
        data,
        stable_strategy=STABLE_STRATEGY,
        stability_score=0.20,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert runtime.decision["decision"] == "NO TRADE"
    assert overlay["decision"] == "NO TRADE"
    assert all(
        value is None
        for value in overlay["levels"].values()
    )


def test_runtime_uses_daily_interval_for_daily_dataset():
    data = _rising_data()

    runtime = build_live_runtime(
        data,
        stable_strategy=STABLE_STRATEGY,
        stability_score=0.80,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    assert runtime.decision["interval"] == "1d"
    assert runtime.display["interval"] == "1d"


def test_load_runtime_data_contains_valid_market_data():
    data = _load_runtime_data()

    assert {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }.issubset(data.columns)

    assert not data.empty
    assert data["timestamp"].notna().all()


def test_load_runtime_data_has_numeric_prices():
    data = _load_runtime_data()

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):
        assert pd.api.types.is_numeric_dtype(
            data[column]
        )


def test_runtime_metadata_reaches_display():
    data = _rising_data()

    runtime = build_live_runtime(
        data,
        stable_strategy=STABLE_STRATEGY,
        stability_score=0.80,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    assert overlay["symbol"] == SYMBOL
    assert overlay["interval"] == INTERVAL
    assert overlay["stable_strategy"] == STABLE_STRATEGY
    assert overlay["stability_score"] == pytest.approx(0.80)
