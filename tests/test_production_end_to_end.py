from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation.production_end_to_end import (
    load_production_market_data,
    run_production_end_to_end,
)


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )

    close = pd.Series(
        [
            2000.0 + index * 2.0
            for index in range(rows)
        ],
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


def test_production_end_to_end_uses_candidate_lineage(
    tmp_path,
):
    # Conflicting production.json file MUST NOT affect the decision or stability authority
    result_file = tmp_path / "production.json"
    result_file.write_text(
        json.dumps(
            {
                "stable_strategy": "bogus_strategy",
                "stability_score": 0.10,
            }
        ),
        encoding="utf-8",
    )

    data = _rising_data()

    result = run_production_end_to_end(
        data,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        reference_now=data["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        min_stability_score=0.40,
        store_path=tmp_path / "store.json",
    )

    assert result["end_to_end_ready"] is True
    assert result["production_selection"]["stable_strategy"] == "momentum"
    assert result["production_selection"]["stability_score"] == pytest.approx(0.7458282289664787)
    assert result["production_selection"]["source"] == "promoted_candidate_binding"

    assert result["decision"]["decision"] == "BUY"
    assert result["overlay"]["decision"] == "BUY"
    assert result["release_gate"]["release_ready"] is True


def test_production_end_to_end_rejects_low_min_stability(
    tmp_path,
):
    result = run_production_end_to_end(
        _rising_data(),
        symbol="XAUUSD",
        interval="1d",
        reference_now=_rising_data()["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        min_stability_score=0.90,
        results_dir=tmp_path,
        store_path=tmp_path / "store.json",
    )

    assert result["end_to_end_ready"] is False
    assert result["release_gate"]["release_ready"] is False


def test_production_end_to_end_rejects_missing_candidate(
    tmp_path,
):
    with pytest.raises(ValueError, match="No promoted candidate found"):
        run_production_end_to_end(
            _rising_data(),
            symbol="EURUSD",
            interval="1d",
            research_dir=tmp_path,
        )


def test_load_production_market_data(tmp_path):
    path = tmp_path / "market.csv"

    data = _rising_data(10)
    data.to_csv(path, index=False)

    loaded = load_production_market_data(path)

    assert len(loaded) == 10
    assert {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }.issubset(loaded.columns)

    assert loaded["timestamp"].notna().all()


def test_load_production_market_data_rejects_missing_file(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="Market data file not found",
    ):
        load_production_market_data(
            tmp_path / "missing.csv"
        )


def test_load_production_market_data_rejects_missing_columns(
    tmp_path,
):
    path = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "timestamp": [1],
            "open": [2000],
            "close": [2001],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        load_production_market_data(path)


def test_load_production_market_data_rejects_empty_data(
    tmp_path,
):
    path = tmp_path / "empty.csv"

    pd.DataFrame(
        {
            "timestamp": [None],
            "open": [None],
            "high": [None],
            "low": [None],
            "close": [None],
        }
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="No valid market data",
    ):
        load_production_market_data(path)
