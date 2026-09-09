import pytest

from src.evaluation.release_gate import (
    assert_release_ready,
    build_release_gate,
)


def valid_walk_forward():
    return {
        "strategy": "momentum",
        "stability_score": 0.517268,
        "total_return": 0.351845,
        "max_drawdown": -0.106211,
        "sharpe_ratio": 0.039249,
    }


def valid_production():
    return {
        "strategy": "momentum",
        "stability_score": 0.517268,
        "observations": 312,
        "total_return": 0.497303,
        "max_drawdown": -0.069050,
        "sharpe_ratio": 0.142056,
    }


def test_release_gate_passes_for_complete_consistent_results():
    gate = build_release_gate(
        walk_forward=valid_walk_forward(),
        production=valid_production(),
        robustness={"robust": True},
    )

    assert gate["ready"] is True
    assert gate["status"] == "RELEASE_READY"
    assert gate["strategy"] == "momentum"
    assert all(gate["checks"].values())


def test_release_gate_blocks_when_robustness_fails():
    gate = build_release_gate(
        walk_forward=valid_walk_forward(),
        production=valid_production(),
        robustness={"robust": False},
    )

    assert gate["ready"] is False
    assert gate["status"] == "RELEASE_BLOCKED"
    assert gate["checks"]["robustness_passed"] is False


def test_release_gate_blocks_when_strategies_are_inconsistent():
    production = valid_production()
    production["strategy"] = "moving_average"

    gate = build_release_gate(
        walk_forward=valid_walk_forward(),
        production=production,
        robustness={"robust": True},
    )

    assert gate["ready"] is False
    assert gate["checks"]["strategy_consistent"] is False


def test_release_gate_blocks_missing_walk_forward_fields():
    walk_forward = valid_walk_forward()
    del walk_forward["sharpe_ratio"]

    gate = build_release_gate(
        walk_forward=walk_forward,
        production=valid_production(),
        robustness={"robust": True},
    )

    assert gate["ready"] is False
    assert gate["checks"]["walk_forward_complete"] is False


def test_release_gate_blocks_missing_production_fields():
    production = valid_production()
    del production["total_return"]

    gate = build_release_gate(
        walk_forward=valid_walk_forward(),
        production=production,
        robustness={"robust": True},
    )

    assert gate["ready"] is False
    assert gate["checks"]["production_complete"] is False


def test_release_gate_blocks_invalid_observations():
    production = valid_production()
    production["observations"] = 0

    gate = build_release_gate(
        walk_forward=valid_walk_forward(),
        production=production,
        robustness={"robust": True},
    )

    assert gate["ready"] is False
    assert gate["checks"]["production_observations_valid"] is False


def test_release_gate_blocks_non_finite_metrics():
    production = valid_production()
    production["sharpe_ratio"] = float("nan")

    gate = build_release_gate(
        walk_forward=valid_walk_forward(),
        production=production,
        robustness={"robust": True},
    )

    assert gate["ready"] is False
    assert gate["checks"]["production_numeric"] is False


def test_mapping_arguments_are_required():
    with pytest.raises(TypeError, match="walk_forward must be a mapping"):
        build_release_gate(
            walk_forward=None,
            production=valid_production(),
            robustness={"robust": True},
        )


def test_assert_release_ready_accepts_ready_gate():
    gate = {
        "ready": True,
        "status": "RELEASE_READY",
    }

    assert_release_ready(gate)


def test_assert_release_ready_rejects_failed_gate():
    gate = {
        "ready": False,
        "status": "RELEASE_BLOCKED",
    }

    with pytest.raises(
        RuntimeError,
        match="release gate failed",
    ):
        assert_release_ready(gate)


def test_assert_release_ready_requires_mapping():
    with pytest.raises(TypeError, match="gate must be a mapping"):
        assert_release_ready(None)
