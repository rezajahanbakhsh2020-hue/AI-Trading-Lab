import pandas as pd
import pytest

from src.evaluation.live_runtime import (
    build_live_runtime,
)


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=80,
                freq="5min",
            ),
            "open": [4400.0 + i for i in range(80)],
            "high": [4401.0 + i for i in range(80)],
            "low": [4399.0 + i for i in range(80)],
            "close": [4400.5 + i for i in range(80)],
        }
    )


def test_live_runtime_contains_tp_levels():
    result = build_live_runtime(
        data=_data(),
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="5m",
    )

    assert result.decision["symbol"] == "XAUUSD"
    assert result.decision["interval"] == "5m"

    assert "tp1" in result.decision
    assert "tp2" in result.decision
    assert "tp3" in result.decision

    assert result.display["tp1"] == result.decision["tp1"]
    assert result.display["tp2"] == result.decision["tp2"]
    assert result.display["tp3"] == result.decision["tp3"]


def test_live_runtime_keeps_entry_and_risk_levels_consistent():
    result = build_live_runtime(
        data=_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    for field in (
        "entry_price",
        "stop_loss",
        "take_profit",
    ):
        assert result.display[field] == result.decision[field]


def test_live_runtime_rejects_non_dataframe():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        build_live_runtime(
            data=[],
            stable_strategy="momentum",
            stability_score=0.80,
        )


def test_live_runtime_rejects_empty_dataframe():
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_live_runtime(
            data=pd.DataFrame(),
            stable_strategy="momentum",
            stability_score=0.80,
        )
