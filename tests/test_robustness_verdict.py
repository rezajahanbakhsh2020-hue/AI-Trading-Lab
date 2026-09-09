import pandas as pd
import pytest

from src.evaluation.robustness_verdict import (
    build_robustness_verdict,
    summarize_robustness_verdict,
)


def test_build_robustness_verdict_passes_when_all_checks_are_met():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001, 0.002],
            "total_return": [0.20, 0.15, 0.10],
            "sharpe_ratio": [1.0, 0.8, 0.5],
        }
    )

    verdict = build_robustness_verdict(
        results,
        minimum_positive_return_rate=1.0,
        minimum_worst_return=0.0,
        minimum_worst_sharpe=0.0,
    )

    assert verdict["scenarios"] == 3
    assert verdict["positive_return_rate"] == 1.0
    assert verdict["worst_total_return"] == 0.10
    assert verdict["worst_sharpe_ratio"] == 0.5
    assert verdict["checks"]["positive_return_rate"] is True
    assert verdict["checks"]["worst_total_return"] is True
    assert verdict["checks"]["worst_sharpe_ratio"] is True
    assert verdict["robust"] is True
    assert verdict["verdict"] == "ROBUST"


def test_build_robustness_verdict_fails_when_worst_return_is_below_limit():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001, 0.002],
            "total_return": [0.20, 0.05, -0.02],
        }
    )

    verdict = build_robustness_verdict(
        results,
        minimum_positive_return_rate=2 / 3,
        minimum_worst_return=0.0,
    )

    assert verdict["positive_return_rate"] == pytest.approx(2 / 3)
    assert verdict["checks"]["positive_return_rate"] is True
    assert verdict["checks"]["worst_total_return"] is False
    assert verdict["robust"] is False
    assert verdict["verdict"] == "NOT_ROBUST"


def test_positive_return_threshold_can_be_relaxed():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001, 0.002],
            "total_return": [0.20, 0.05, -0.02],
        }
    )

    verdict = build_robustness_verdict(
        results,
        minimum_positive_return_rate=0.5,
        minimum_worst_return=-0.05,
    )

    assert verdict["checks"]["positive_return_rate"] is True
    assert verdict["checks"]["worst_total_return"] is True
    assert verdict["robust"] is True


def test_sharpe_threshold_is_optional():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001],
            "total_return": [0.20, 0.10],
        }
    )

    verdict = build_robustness_verdict(
        results,
        minimum_worst_return=0.0,
    )

    assert "worst_sharpe_ratio" not in verdict
    assert "worst_sharpe_ratio" not in verdict["checks"]
    assert verdict["robust"] is True


def test_sharpe_threshold_requires_sharpe_column():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0, 0.001],
            "total_return": [0.20, 0.10],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required Sharpe column",
    ):
        build_robustness_verdict(
            results,
            minimum_worst_sharpe=0.0,
        )


def test_empty_results_are_rejected():
    results = pd.DataFrame(
        columns=["transaction_cost", "total_return"]
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_robustness_verdict(results)


def test_missing_return_column_is_rejected():
    results = pd.DataFrame(
        {
            "transaction_cost": [0.0],
            "sharpe_ratio": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required return column",
    ):
        build_robustness_verdict(results)


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
        build_robustness_verdict(results)


def test_invalid_positive_return_threshold_is_rejected():
    results = pd.DataFrame(
        {
            "total_return": [0.10],
        }
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        build_robustness_verdict(
            results,
            minimum_positive_return_rate=1.1,
        )


def test_summary_for_robust_verdict():
    verdict = {
        "robust": True,
        "verdict": "ROBUST",
        "scenarios": 3,
        "positive_return_rate": 1.0,
        "worst_total_return": 0.10,
    }

    summary = summarize_robustness_verdict(verdict)

    assert summary == (
        "PASS: ROBUST; scenarios=3; "
        "positive_return_rate=1.000000; "
        "worst_total_return=0.100000"
    )


def test_summary_for_failed_verdict():
    verdict = {
        "robust": False,
        "verdict": "NOT_ROBUST",
        "scenarios": 3,
        "positive_return_rate": 2 / 3,
        "worst_total_return": -0.02,
    }

    summary = summarize_robustness_verdict(verdict)

    assert summary == (
        "FAIL: NOT_ROBUST; scenarios=3; "
        "positive_return_rate=0.666667; "
        "worst_total_return=-0.020000"
    )


def test_summary_requires_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        summarize_robustness_verdict(None)


def test_summary_requires_verdict_fields():
    with pytest.raises(
        ValueError,
        match="must contain robust and verdict fields",
    ):
        summarize_robustness_verdict({"robust": True})
