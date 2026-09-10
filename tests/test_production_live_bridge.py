from __future__ import annotations

import json

import pytest

from src.evaluation.production_live_bridge import (
    extract_production_selection,
    load_latest_production_result,
    load_production_selection,
)


def test_extracts_stable_strategy_and_score():
    result = extract_production_selection(
        {
            "result": {
                "stable_strategy": "momentum",
                "stability_score": 0.517268,
            },
            "path": "results/production/result.json",
        }
    )

    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == pytest.approx(
        0.517268
    )
    assert (
        result["source_path"]
        == "results/production/result.json"
    )


def test_extracts_strategy_fallback():
    result = extract_production_selection(
        {
            "strategy": "momentum",
            "stability_score": 0.517268,
        }
    )

    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == pytest.approx(
        0.517268
    )


def test_extracts_nested_stability_score():
    result = extract_production_selection(
        {
            "stable_strategy": "momentum",
            "stability": {
                "stability_score": 0.517268,
            },
        }
    )

    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == pytest.approx(
        0.517268
    )


def test_rejects_missing_strategy():
    with pytest.raises(
        ValueError,
        match="stable strategy",
    ):
        extract_production_selection(
            {
                "stability_score": 0.517268,
            }
        )


def test_rejects_missing_stability_score():
    with pytest.raises(
        ValueError,
        match="stability score",
    ):
        extract_production_selection(
            {
                "stable_strategy": "momentum",
            }
        )


def test_rejects_non_dictionary_input():
    with pytest.raises(
        TypeError,
        match="must be a dictionary",
    ):
        extract_production_selection([])


def test_loads_latest_json(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first.write_text(
        json.dumps(
            {
                "stable_strategy": "moving_average",
                "stability_score": 0.46,
            }
        ),
        encoding="utf-8",
    )

    second.write_text(
        json.dumps(
            {
                "stable_strategy": "momentum",
                "stability_score": 0.517268,
            }
        ),
        encoding="utf-8",
    )

    latest = load_latest_production_result(tmp_path)

    assert "result" in latest
    assert latest["result"]["stability_score"] in {
        0.46,
        0.517268,
    }


def test_load_production_selection(tmp_path):
    path = tmp_path / "production.json"

    path.write_text(
        json.dumps(
            {
                "stable_strategy": "momentum",
                "stability_score": 0.517268,
            }
        ),
        encoding="utf-8",
    )

    selection = load_production_selection(tmp_path)

    assert selection["stable_strategy"] == "momentum"
    assert selection["stability_score"] == pytest.approx(
        0.517268
    )
    assert selection["source_path"].endswith(
        "production.json"
    )


def test_missing_production_directory_fails(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Production results directory",
    ):
        load_latest_production_result(missing)


def test_empty_production_directory_fails(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="No production JSON result",
    ):
        load_latest_production_result(tmp_path)
