from __future__ import annotations

import pandas as pd

from src.evaluation.end_to_end_runner import run_end_to_end


def _rising_data(rows: int = 80) -> pd.DataFrame:
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


def test_end_to_end_runner_produces_complete_buy_result():
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert result["end_to_end_ready"] is True
    assert result["decision"]["decision"] == "BUY"
    assert result["display"]["decision"] == "BUY"
    assert result["overlay"]["decision"] == "BUY"
    assert result["release_gate"]["release_ready"] is True


def test_end_to_end_runner_contains_trade_levels():
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    levels = result["overlay"]["levels"]

    assert levels["entry"] is not None
    assert levels["stop_loss"] is not None
    assert levels["tp1"] is not None
    assert levels["tp2"] is not None
    assert levels["tp3"] is not None

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]


def test_end_to_end_runner_rejects_low_stability_for_release():
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.20,
    )

    assert result["end_to_end_ready"] is False
    assert result["release_gate"]["release_ready"] is False


def test_end_to_end_runner_preserves_symbol_and_interval():
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="1d",
    )

    assert result["decision"]["symbol"] == "XAUUSD"
    assert result["decision"]["interval"] == "1d"
    assert result["display"]["symbol"] == "XAUUSD"
    assert result["display"]["interval"] == "1d"
    assert result["overlay"]["symbol"] == "XAUUSD"
    assert result["overlay"]["interval"] == "1d"


def test_end_to_end_runner_exposes_all_pipeline_layers():
    result = run_end_to_end(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert set(result) == {
        "decision",
        "display",
        "overlay",
        "release_gate",
        "end_to_end_ready",
    }

    assert result["overlay"]["lines"]
    assert len(result["overlay"]["lines"]) == 5
