import pytest

from src.evaluation.production_runtime_gate import (
    build_production_runtime_gate,
    is_production_runtime_ready,
    production_runtime_gate_message,
)


def ready_readiness() -> dict:
    return {
        "strategy": "momentum",
        "strategy_stability_score": 0.90,
        "portfolio_stability_score": 0.90,
        "readiness_score": 0.90,
        "status": "READY",
        "failed_gates": [],
    }


def test_runtime_gate_is_ready():
    result = build_production_runtime_gate(
        ready_readiness()
    )

    assert result["runtime_ready"] is True
    assert result["status"] == "READY"
    assert result["strategy"] == "momentum"
    assert result["readiness_score"] == pytest.approx(
        0.90
    )
    assert result["failed_gates"] == []


def test_runtime_gate_blocks_production_readiness_failure():
    readiness = ready_readiness()
    readiness["status"] = "BLOCKED"
    readiness["failed_gates"] = [
        "portfolio_stability"
    ]

    result = build_production_runtime_gate(
        readiness
    )

    assert result["runtime_ready"] is False
    assert result["status"] == "BLOCKED"
    assert "portfolio_stability" in result[
        "failed_gates"
    ]


def test_runtime_gate_blocks_low_combined_score():
    readiness = ready_readiness()
    readiness["readiness_score"] = 0.65

    result = build_production_runtime_gate(
        readiness,
        minimum_readiness_score=0.70,
    )

    assert result["runtime_ready"] is False
    assert result["status"] == "BLOCKED"
    assert "combined_readiness" in result[
        "failed_gates"
    ]


def test_runtime_gate_blocks_missing_strategy():
    readiness = ready_readiness()
    readiness["strategy"] = None

    result = build_production_runtime_gate(
        readiness
    )

    assert result["runtime_ready"] is False
    assert "strategy_selection" in result[
        "failed_gates"
    ]


def test_runtime_gate_blocks_invalid_strategy_score():
    readiness = ready_readiness()
    readiness["strategy_stability_score"] = 1.5

    result = build_production_runtime_gate(
        readiness
    )

    assert result["runtime_ready"] is False
    assert "strategy_stability" in result[
        "failed_gates"
    ]


def test_runtime_gate_blocks_invalid_portfolio_score():
    readiness = ready_readiness()
    readiness["portfolio_stability_score"] = -0.1

    result = build_production_runtime_gate(
        readiness
    )

    assert result["runtime_ready"] is False
    assert "portfolio_stability" in result[
        "failed_gates"
    ]


def test_is_production_runtime_ready_returns_boolean():
    assert (
        is_production_runtime_ready(
            ready_readiness()
        )
        is True
    )

    readiness = ready_readiness()
    readiness["readiness_score"] = 0.60

    assert (
        is_production_runtime_ready(
            readiness
        )
        is False
    )


def test_runtime_gate_message_ready():
    message = production_runtime_gate_message(
        ready_readiness()
    )

    assert message.startswith(
        "RUNTIME READY:"
    )
    assert "momentum" in message


def test_runtime_gate_message_blocked():
    readiness = ready_readiness()
    readiness["status"] = "BLOCKED"
    readiness["failed_gates"] = [
        "portfolio_stability"
    ]

    message = production_runtime_gate_message(
        readiness
    )

    assert message.startswith(
        "RUNTIME BLOCKED:"
    )
    assert "momentum" in message
    assert "portfolio_stability" in message


def test_runtime_gate_rejects_non_mapping():
    with pytest.raises(TypeError):
        build_production_runtime_gate([])
