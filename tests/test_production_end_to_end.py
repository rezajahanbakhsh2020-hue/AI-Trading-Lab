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


def test_production_end_to_end_uses_saved_selection(
    tmp_path,
):
    result_file = tmp_path / "production.json"

    result_file.write_text(
        json.dumps(
            {
                "stable_strategy": "momentum",
                "stability_score": 0.80,
            }
        ),
        encoding="utf-8",
    )

    result = run_production_end_to_end(
        _rising_data(),
        results_dir=tmp_path,
    )

    assert result["end_to_end_ready"] is True
    assert (
        result["production_selection"][
            "stable_strategy"
        ]
        == "momentum"
    )
    assert result["production_selection"][
        "stability_score"
    ] == pytest.approx(0.80)

    assert result["decision"]["decision"] == "BUY"
    assert result["overlay"]["decision"] == "BUY"
    assert result["release_gate"][
        "release_ready"
    ] is True


def test_production_end_to_end_rejects_low_saved_stability(
    tmp_path,
):
    result_file = tmp_path / "production.json"

    result_file.write_text(
        json.dumps(
            {
                "stable_strategy": "momentum",
                "stability_score": 0.20,
            }
        ),
        encoding="utf-8",
    )

    result = run_production_end_to_end(
        _rising_data(),
        results_dir=tmp_path,
    )

    assert result["end_to_end_ready"] is False
    assert result["release_gate"][
        "release_ready"
    ] is False


def test_production_end_to_end_preserves_production_source(
    tmp_path,
):
    result_file = tmp_path / "production.json"

    result_file.write_text(
        json.dumps(
            {
                "stable_strategy": "momentum",
                "stability_score": 0.80,
            }
        ),
        encoding="utf-8",
    )

    result = run_production_end_to_end(
        _rising_data(),
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
    )

    source = result["production_selection"][
        "source_path"
    ]

    assert source is not None
    assert source.endswith("production.json")
    assert result["decision"]["symbol"] == "XAUUSD"
    assert result["decision"]["interval"] == "1d"


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
