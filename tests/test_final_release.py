from __future__ import annotations

import pytest

from src.evaluation.final_release import (
    validate_final_release,
)


def _valid_result() -> dict:
    return {
        "end_to_end_ready": True,
        "decision": {
            "decision": "BUY",
        },
        "display": {
            "decision": "BUY",
        },
        "overlay": {
            "decision": "BUY",
            "levels": {
                "stop_loss": 1980.0,
                "entry": 2000.0,
                "tp1": 2020.0,
                "tp2": 2040.0,
                "tp3": 2060.0,
            },
        },
        "release_gate": {
            "release_ready": True,
        },
        "production_selection": {
            "stable_strategy": "momentum",
            "stability_score": 0.517268,
        },
    }


def test_final_release_is_ready_for_valid_result():
    result = validate_final_release(
        _valid_result()
    )

    assert result["final_release_ready"] is True
    assert result["status"] == "RELEASE READY"
    assert all(result["checks"].values())


def test_final_release_requires_end_to_end_ready():
    payload = _valid_result()
    payload["end_to_end_ready"] = False

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["end_to_end_ready"] is False


def test_final_release_requires_release_gate():
    payload = _valid_result()
    payload["release_gate"]["release_ready"] = False

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["release_gate_ready"] is False


def test_final_release_requires_momentum_strategy():
    payload = _valid_result()
    payload["production_selection"][
        "stable_strategy"
    ] = "moving_average"

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["strategy_stable"] is False


def test_final_release_requires_minimum_stability():
    payload = _valid_result()
    payload["production_selection"][
        "stability_score"
    ] = 0.49

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"][
        "stability_score_valid"
    ] is False


def test_final_release_requires_matching_decisions():
    payload = _valid_result()
    payload["display"]["decision"] = "NO TRADE"

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"][
        "display_matches_decision"
    ] is False


def test_final_release_requires_complete_levels():
    payload = _valid_result()
    payload["overlay"]["levels"]["tp3"] = None

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"][
        "trade_levels_complete"
    ] is False


def test_final_release_requires_correct_level_order():
    payload = _valid_result()
    payload["overlay"]["levels"]["tp2"] = 2010.0

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"][
        "trade_levels_ordered"
    ] is False


def test_final_release_requires_all_sections():
    payload = _valid_result()
    payload.pop("overlay")

    with pytest.raises(
        ValueError,
        match="Missing final release sections",
    ):
        validate_final_release(payload)


def test_final_release_rejects_non_dictionary():
    with pytest.raises(
        TypeError,
        match="result must be a dictionary",
    ):
        validate_final_release([])
