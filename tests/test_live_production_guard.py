import pytest

from src.evaluation.live_production_guard import (
    build_production_guard,
    is_production_ready,
    production_guard_message,
    validate_production_result,
)


def sample_result() -> dict:
    return {
        "strategy": "momentum",
        "stability_score": 0.517268,
        "total_return": 0.497303,
        "max_drawdown": -0.069050,
        "sharpe_ratio": 0.142056,
    }


def test_validate_production_result():
    result = validate_production_result(
        sample_result()
    )

    assert result["valid"] is True
    assert result["strategy"] == "momentum"
    assert result["stability_score"] == 0.517268
    assert result["total_return"] == 0.497303
    assert result["max_drawdown"] == -0.069050
    assert result["sharpe_ratio"] == 0.142056


def test_build_production_guard_ready():
    guard = build_production_guard(
        sample_result(),
        minimum_stability_score=0.50,
    )

    assert guard["valid"] is True
    assert guard["stability_gate"] is True
    assert guard["status"] == "READY"
    assert guard["strategy"] == "momentum"


def test_build_production_guard_blocked():
    guard = build_production_guard(
        sample_result(),
        minimum_stability_score=0.60,
    )

    assert guard["stability_gate"] is False
    assert guard["status"] == "BLOCKED"


def test_is_production_ready():
    assert (
        is_production_ready(
            sample_result(),
            minimum_stability_score=0.50,
        )
        is True
    )

    assert (
        is_production_ready(
            sample_result(),
            minimum_stability_score=0.60,
        )
        is False
    )


def test_production_guard_message_ready():
    message = production_guard_message(
        sample_result(),
        minimum_stability_score=0.50,
    )

    assert "PRODUCTION READY" in message
    assert "momentum" in message


def test_production_guard_message_blocked():
    message = production_guard_message(
        sample_result(),
        minimum_stability_score=0.60,
    )

    assert "PRODUCTION BLOCKED" in message
    assert "momentum" in message


def test_missing_required_field_is_rejected():
    result = sample_result()
    del result["stability_score"]

    with pytest.raises(ValueError):
        validate_production_result(result)


def test_invalid_numeric_metric_is_rejected():
    result = sample_result()
    result["sharpe_ratio"] = "invalid"

    with pytest.raises(ValueError):
        validate_production_result(result)


def test_empty_strategy_is_rejected():
    result = sample_result()
    result["strategy"] = ""

    with pytest.raises(ValueError):
        validate_production_result(result)


def test_non_mapping_is_rejected():
    with pytest.raises(TypeError):
        validate_production_result([])
