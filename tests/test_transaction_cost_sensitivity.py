import pandas as pd
import pytest

from src.evaluation.transaction_cost_sensitivity import (
    assess_transaction_cost_robustness,
    build_transaction_cost_sensitivity_report,
    run_transaction_cost_sensitivity,
)


def test_run_transaction_cost_sensitivity_collects_all_scenarios():
    calls = []

    def runner(transaction_cost):
        calls.append(transaction_cost)
        return {
            "total_return": 0.20 - transaction_cost,
            "sharpe_ratio": 1.0 - transaction_cost,
            "max_drawdown": -0.10,
        }

    result = run_transaction_cost_sensitivity(
        runner,
        [0.0, 0.001, 0.002],
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result["transaction_cost"]) == [0.0, 0.001, 0.002]
    assert calls == [0.0, 0.001, 0.002]
    assert list(result["total_return"]) == [0.20, 0.199, 0.198]


def test_assess_transaction_cost_robustness_passes_when_all_returns_positive():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001, 0.002],
            "total_return": [0.20, 0.15, 0.10],
            "sharpe_ratio": [1.0, 0.8, 0.5],
        }
    )

    assessment = assess_transaction_cost_robustness(results)

    assert assessment["scenarios"] == 3
    assert assessment["positive_return_rate"] == 1.0
    assert assessment["worst_total_return"] == 0.10
    assert assessment["best_total_return"] == 0.20
    assert assessment["average_total_return"] == pytest.approx(0.15)
    assert assessment["worst_sharpe_ratio"] == 0.5
    assert assessment["best_sharpe_ratio"] == 1.0
    assert assessment["robust"] is True


def test_assess_transaction_cost_robustness_fails_when_one_scenario_is_negative():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001, 0.002],
            "total_return": [0.20, 0.05, -0.02],
            "sharpe_ratio": [1.0, 0.4, -0.1],
        }
    )

    assessment = assess_transaction_cost_robustness(results)

    assert assessment["positive_return_rate"] == pytest.approx(2 / 3)
    assert assessment["worst_total_return"] == -0.02
    assert assessment["robust"] is False


def test_custom_positive_return_threshold_is_supported():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001, 0.002],
            "total_return": [0.20, 0.05, -0.02],
        }
    )

    assessment = assess_transaction_cost_robustness(
        results,
        minimum_positive_return_rate=2 / 3,
    )

    assert assessment["robust"] is True


def test_empty_cost_list_is_rejected():
    def runner(transaction_cost):
        return {"total_return": 0.1}

    with pytest.raises(ValueError, match="must not be empty"):
        run_transaction_cost_sensitivity(runner, [])


def test_negative_transaction_cost_is_rejected():
    def runner(transaction_cost):
        return {"total_return": 0.1}

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        run_transaction_cost_sensitivity(runner, [-0.001])


def test_non_callable_runner_is_rejected():
    with pytest.raises(TypeError, match="must be callable"):
        run_transaction_cost_sensitivity(
            None,
            [0.0, 0.001],
        )


def test_runner_must_return_mapping():
    def runner(transaction_cost):
        return 0.5

    with pytest.raises(
        TypeError,
        match="must return a mapping",
    ):
        run_transaction_cost_sensitivity(
            runner,
            [0.0],
        )


def test_missing_return_column_is_rejected():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001],
            "sharpe_ratio": [1.0, 0.8],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required return column",
    ):
        assess_transaction_cost_robustness(results)


def test_non_finite_returns_are_rejected():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001],
            "total_return": [0.10, float("nan")],
        }
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        assess_transaction_cost_robustness(results)


def test_build_report_returns_results_and_assessment():
    def runner(transaction_cost):
        return {
            "total_return": 0.20 - transaction_cost,
            "sharpe_ratio": 1.0 - transaction_cost,
        }

    report = build_transaction_cost_sensitivity_report(
        runner,
        [0.0, 0.001, 0.002],
    )

    assert set(report.keys()) == {"results", "assessment"}
    assert isinstance(report["results"], pd.DataFrame)
    assert report["assessment"]["scenarios"] == 3
    assert report["assessment"]["robust"] is True
