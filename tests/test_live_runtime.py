import pandas as pd
import pytest

from src.evaluation.live_runtime import (
    LiveRuntimeResult,
    build_live_runtime,
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


def test_live_runtime_returns_complete_result():
    result = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert isinstance(result, LiveRuntimeResult)
    assert result.decision["decision"] == "BUY"
    assert result.display["decision"] == "BUY"


def test_live_runtime_keeps_decision_and_display_consistent():
    result = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert (
        result.decision["decision"]
        == result.display["decision"]
    )


def test_live_runtime_preserves_production_metadata():
    result = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="5m",
    )

    assert result.decision["symbol"] == "XAUUSD"
    assert result.decision["interval"] == "5m"
    assert result.display["symbol"] == "XAUUSD"
    assert result.display["interval"] == "5m"


def test_live_runtime_no_trade_when_stability_is_below_threshold():
    result = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.20,
    )

    assert result.decision["decision"] == "NO TRADE"
    assert result.display["decision"] == "NO TRADE"


def test_live_runtime_no_trade_has_no_display_levels():
    result = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.20,
    )

    assert result.display["entry_price"] is None
    assert result.display["stop_loss"] is None
    assert result.display["tp1"] is None
    assert result.display["tp2"] is None
    assert result.display["tp3"] is None


def test_live_runtime_rejects_empty_data():
    with pytest.raises(ValueError, match="must not be empty"):
        build_live_runtime(
            pd.DataFrame(),
            stable_strategy="momentum",
            stability_score=0.80,
        )


def test_live_runtime_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        build_live_runtime(
            [],
            stable_strategy="momentum",
            stability_score=0.80,
        )


def test_live_runtime_supports_custom_stability_threshold():
    result = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.60,
        min_stability_score=0.70,
    )

    assert result.decision["decision"] == "NO TRADE"
    assert result.display["decision"] == "NO TRADE"
