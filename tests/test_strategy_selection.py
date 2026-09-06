import pandas as pd
import pytest

from src.evaluation.strategy_selection import (
    select_eligible_strategies,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "strong",
                "weak_return",
                "high_drawdown",
                "weak_sharpe",
                "unstable",
            ],
            "total_return": [
                0.30,
                -0.05,
                0.25,
                0.25,
                0.25,
            ],
            "max_drawdown": [
                -0.10,
                -0.05,
                -0.40,
                -0.10,
                -0.10,
            ],
            "sharpe_ratio": [
                1.50,
                1.50,
                1.50,
                -0.20,
                1.50,
            ],
            "positive_window_rate": [
                0.80,
                0.80,
                0.80,
                0.80,
                0.30,
            ],
        }
    )


def test_select_eligible_strategies_returns_only_passing_strategies():
    report = make_report()

    result = select_eligible_strategies(report)

    assert list(result["strategy"]) == ["strong"]


def test_selection_respects_custom_thresholds():
    report = make_report()

    result = select_eligible_strategies(
        report,
        min_total_return=0.20,
        max_drawdown=0.15,
        min_sharpe_ratio=1.0,
        min_positive_window_rate=0.75,
    )

    assert list(result["strategy"]) == ["strong"]


def test_selection_allows_multiple_strategies():
    report = make_report()

    result = select_eligible_strategies(
        report,
        min_total_return=0.20,
        max_drawdown=0.15,
        min_sharpe_ratio=1.0,
        min_positive_window_rate=0.30,
    )

    assert set(result["strategy"]) == {
        "strong",
        "unstable",
    }


def test_selection_does_not_modify_input():
    report = make_report()
    original = report.copy()

    select_eligible_strategies(report)

    pd.testing.assert_frame_equal(report, original)


def test_selection_returns_empty_dataframe_when_nothing_passes():
    report = make_report()

    result = select_eligible_strategies(
        report,
        min_total_return=1.0,
    )

    assert result.empty
    assert list(result.columns) == list(report.columns)


def test_selection_rejects_invalid_input_type():
    with pytest.raises(
        TypeError,
        match="report must be a pandas DataFrame.",
    ):
        select_eligible_strategies("invalid")


def test_selection_rejects_missing_columns():
    report = pd.DataFrame(
        {
            "total_return": [0.20],
            "max_drawdown": [-0.10],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns:",
    ):
        select_eligible_strategies(report)


def test_selection_rejects_invalid_positive_window_rate():
    report = make_report()

    with pytest.raises(
        ValueError,
        match="min_positive_window_rate must be between 0 and 1.",
    ):
        select_eligible_strategies(
            report,
            min_positive_window_rate=1.5,
        )
