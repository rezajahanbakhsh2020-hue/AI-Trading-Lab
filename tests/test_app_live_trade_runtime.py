from __future__ import annotations

import pandas as pd
import pytest

import app_live_trade_runtime as module


def test_load_stable_selection_uses_production_result(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "load_production_selection",
        lambda: {
            "stable_strategy": "momentum",
            "stability_score": 0.517268,
            "source_path": (
                "results/production/latest.json"
            ),
        },
    )

    selection = module._load_stable_selection()

    assert selection["stable_strategy"] == "momentum"
    assert selection["stability_score"] == pytest.approx(
        0.517268
    )
    assert (
        selection["source_path"]
        == "results/production/latest.json"
    )


def test_load_stable_selection_rejects_missing_strategy(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "load_production_selection",
        lambda: {
            "stable_strategy": None,
            "stability_score": 0.5,
        },
    )

    with pytest.raises(ValueError):
        module._load_stable_selection()


def test_load_stable_selection_rejects_missing_score(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "load_production_selection",
        lambda: {
            "stable_strategy": "momentum",
            "stability_score": None,
        },
    )

    with pytest.raises(ValueError):
        module._load_stable_selection()


def test_load_runtime_data():
    data = pd.DataFrame(
        {
            "timestamp": [
                1735689600,
                1735776000,
            ],
            "open": [2625.1, 2623.66],
            "high": [2626.0, 2660.38],
            "low": [2621.48, 2622.45],
            "close": [2623.56, 2658.30],
        }
    )

    original_read_csv = module.pd.read_csv

    module.pd.read_csv = lambda _: data

    try:
        result = module._load_runtime_data()
    finally:
        module.pd.read_csv = original_read_csv

    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(
        result["timestamp"]
    )
    assert list(result["close"]) == [
        2623.56,
        2658.30,
    ]
