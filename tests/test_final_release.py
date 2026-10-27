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
            "stable_strategy": "momentum",
            "quote_stale": False,
        },
        "display": {
            "decision": "BUY",
            "stable_strategy": "momentum",
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


def test_final_release_requires_strategy_consistency():
    # Fixture is already consistent across selection/decision/display.
    payload = _valid_result()

    result = validate_final_release(payload)

    assert result["final_release_ready"] is True

    # Any inconsistency with the authorized binding blocks the release.
    payload["display"]["stable_strategy"] = "moving_average"
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


def _sell_result() -> dict:
    return {
        "end_to_end_ready": True,
        "decision": {
            "decision": "SELL",
            "stable_strategy": "moving_average",
            "quote_stale": False,
        },
        "display": {
            "decision": "SELL",
            "stable_strategy": "moving_average",
        },
        "overlay": {
            "decision": "SELL",
            "levels": {
                "tp3": 1844.94,
                "tp2": 1863.96,
                "tp1": 1882.98,
                "entry": 1902.0,
                "stop_loss": 1921.02,
            },
        },
        "release_gate": {
            "release_ready": True,
        },
        "production_selection": {
            "stable_strategy": "moving_average",
            "stability_score": 0.541826,
        },
    }


def _no_trade_result() -> dict:
    return {
        "end_to_end_ready": True,
        "decision": {
            "decision": "NO TRADE",
            "stable_strategy": "moving_average",
            "quote_stale": False,
        },
        "display": {
            "decision": "NO TRADE",
            "stable_strategy": "moving_average",
        },
        "overlay": {
            "decision": "NO TRADE",
            "levels": {
                "entry": None,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
            },
        },
        "release_gate": {
            "release_ready": True,
        },
        "production_selection": {
            "stable_strategy": "moving_average",
            "stability_score": 0.541826,
        },
    }


def test_final_release_is_ready_for_valid_sell():
    result = validate_final_release(
        _sell_result()
    )

    assert result["final_release_ready"] is True
    assert result["status"] == "RELEASE READY"
    assert result["decision"] == "SELL"
    assert all(result["checks"].values())


def test_final_release_requires_sell_level_ordering():
    payload = _sell_result()
    # Break SELL geometry: stop_loss below entry (BUY geometry on SELL).
    payload["overlay"]["levels"]["stop_loss"] = 1880.0

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["trade_levels_ordered"] is False


def test_final_release_requires_sell_levels_complete():
    payload = _sell_result()
    payload["overlay"]["levels"]["tp3"] = None

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["trade_levels_complete"] is False


def test_final_release_is_ready_for_legitimate_no_trade():
    result = validate_final_release(
        _no_trade_result()
    )

    assert result["final_release_ready"] is True
    assert result["status"] == "RELEASE READY"
    assert result["decision"] == "NO TRADE"
    assert result["checks"]["trade_levels_complete"] is True
    assert result["checks"]["trade_levels_ordered"] is True


def test_final_release_rejects_no_trade_with_active_levels():
    payload = _no_trade_result()

    # NO TRADE must not carry fabricated trade levels.
    payload["overlay"]["levels"]["entry"] = 1902.0

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["trade_levels_complete"] is False


def test_final_release_blocks_stale_no_trade():
    payload = _no_trade_result()

    # Stale inputs must fail closed: NO TRADE must never be a vehicle for
    # laundering stale market data into a releasable state.
    payload["decision"]["quote_stale"] = True
    payload["end_to_end_ready"] = False
    payload["release_gate"]["release_ready"] = False

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["market_data_fresh"] is False
    assert result["status"] == "RELEASE BLOCKED"


def test_final_release_blocks_stale_buy():
    payload = _valid_result()
    payload["decision"]["quote_stale"] = True
    payload["end_to_end_ready"] = False
    payload["release_gate"]["release_ready"] = False

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["market_data_fresh"] is False


def test_final_release_rejects_unsupported_decision():
    payload = _valid_result()
    payload["decision"]["decision"] = "HOLD"
    payload["display"]["decision"] = "HOLD"
    payload["overlay"]["decision"] = "HOLD"
    payload["overlay"]["levels"] = {k: None for k in payload["overlay"]["levels"]}

    result = validate_final_release(payload)

    assert result["final_release_ready"] is False
    assert result["checks"]["decision_supported"] is False
