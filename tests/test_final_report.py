import pandas as pd
import pytest

from src.evaluation.final_report import (
    build_final_comparison_report,
    build_final_strategy_report,
    get_best_comparison_strategy,
    get_best_strategy,
)


def create_walk_forward_comparison() -> dict[str, dict]:
    return {
        "strategy_a": {
            "windows": 4,
            "observations": 40,
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.50,
            "calmar_ratio": 2.00,
            "sortino_ratio": 1.80,
            "exposure": 0.75,
            "win_rate": 0.60,
            "profit_factor": 1.80,
            "window_returns": [0.05, 0.03, 0.07, 0.05],
            "profitable_windows": 4,
            "losing_windows": 0,
            "positive_window_rate": 1.00,
        },
        "strategy_b": {
            "windows": 4,
            "observations": 40,
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.20,
            "calmar_ratio": 1.80,
            "sortino_ratio": 1.40,
            "exposure": 0.70,
            "win_rate": 0.55,
            "profit_factor": 1.50,
            "window_returns": [0.02, 0.04, -0.01, 0.05],
            "profitable_windows": 3,
            "losing_windows": 1,
            "positive_window_rate": 0.75,
        },
        "strategy_c": {
            "windows": 4,
            "observations": 40,
            "total_return": 0.15,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.30,
            "calmar_ratio": 1.90,
            "sortino_ratio": 1.50,
            "exposure": 0.72,
            "win_rate": 0.58,
            "profit_factor": 1.60,
            "window_returns": [0.04, 0.03, 0.02, 0.06],
            "profitable_windows": 4,
            "losing_windows": 0,
            "positive_window_rate": 1.00,
        },
    }


def create_regular_comparison() -> dict[str, dict]:
    return {
        "strategy_a": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.50,
        },
        "strategy_b": {
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.20,
        },
        "strategy_c": {
            "total_return": 0.15,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.30,
        },
    }


def test_build_final_strategy_report_returns_ranked_dataframe() -> None:
    comparison = create_walk_forward_comparison()

    report = build_final_strategy_report(comparison)

    assert isinstance(report, pd.DataFrame)
    assert report["strategy"].tolist() == [
        "strategy_a",
        "strategy_c",
        "strategy_b",
    ]
    assert report["rank"].tolist() == [1, 2, 3]


def test_build_final_strategy_report_contains_expected_columns() -> None:
    comparison = create_walk_forward_comparison()

    report = build_final_strategy_report(comparison)

    expected_columns = [
        "rank",
        "strategy",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
        "exposure",
        "win_rate",
        "profit_factor",
        "windows",
        "observations",
        "profitable_windows",
        "losing_windows",
        "positive_window_rate",
    ]

    assert report.columns.tolist() == expected_columns


def test_build_final_strategy_report_supports_custom_metric() -> None:
    comparison = create_walk_forward_comparison()

    report = build_final_strategy_report(
        comparison,
        metric="sharpe_ratio",
    )

    assert report.iloc[0]["strategy"] == "strategy_a"


def test_build_final_strategy_report_supports_ascending() -> None:
    comparison = create_walk_forward_comparison()

    report = build_final_strategy_report(
        comparison,
        metric="total_return",
        ascending=True,
    )

    assert report["strategy"].tolist() == [
        "strategy_b",
        "strategy_c",
        "strategy_a",
    ]


def test_build_final_strategy_report_empty_comparison() -> None:
    report = build_final_strategy_report({})

    assert isinstance(report, pd.DataFrame)
    assert report.empty


def test_get_best_strategy_returns_top_strategy() -> None:
    comparison = create_walk_forward_comparison()

    best = get_best_strategy(comparison)

    assert best == "strategy_a"


def test_get_best_strategy_supports_custom_metric() -> None:
    comparison = create_walk_forward_comparison()

    comparison["strategy_b"]["sharpe_ratio"] = 2.00

    best = get_best_strategy(
        comparison,
        metric="sharpe_ratio",
    )

    assert best == "strategy_b"


def test_get_best_strategy_returns_none_for_empty_comparison() -> None:
    assert get_best_strategy({}) is None


def test_build_final_strategy_report_rejects_unknown_metric() -> None:
    comparison = create_walk_forward_comparison()

    with pytest.raises(
        ValueError,
        match="Unknown ranking metric",
    ):
        build_final_strategy_report(
            comparison,
            metric="unknown_metric",
        )


def test_build_final_comparison_report_returns_ranked_dataframe() -> None:
    comparison = create_regular_comparison()

    report = build_final_comparison_report(comparison)

    assert isinstance(report, pd.DataFrame)
    assert report["strategy"].tolist() == [
        "strategy_a",
        "strategy_c",
        "strategy_b",
    ]
    assert report["rank"].tolist() == [1, 2, 3]


def test_build_final_comparison_report_supports_custom_metric() -> None:
    comparison = create_regular_comparison()

    comparison["strategy_b"]["sharpe_ratio"] = 2.00

    report = build_final_comparison_report(
        comparison,
        metric="sharpe_ratio",
    )

    assert report.iloc[0]["strategy"] == "strategy_b"


def test_build_final_comparison_report_handles_drawdown_metric() -> None:
    comparison = create_regular_comparison()

    report = build_final_comparison_report(
        comparison,
        metric="max_drawdown",
    )

    assert report["strategy"].tolist() == [
        "strategy_b",
        "strategy_c",
        "strategy_a",
    ]


def test_build_final_comparison_report_empty_comparison() -> None:
    report = build_final_comparison_report({})

    assert isinstance(report, pd.DataFrame)
    assert report.empty


def test_build_final_comparison_report_rejects_invalid_input() -> None:
    with pytest.raises(TypeError):
        build_final_comparison_report([])  # type: ignore[arg-type]


def test_build_final_comparison_report_rejects_unknown_metric() -> None:
    comparison = create_regular_comparison()

    with pytest.raises(
        ValueError,
        match="Unknown ranking metric",
    ):
        build_final_comparison_report(
            comparison,
            metric="unknown_metric",
        )


def test_build_final_comparison_report_requires_drawdown() -> None:
    comparison = create_regular_comparison()

    for report in comparison.values():
        report.pop("max_drawdown")

    with pytest.raises(
        ValueError,
        match="Missing required ranking column: max_drawdown",
    ):
        build_final_comparison_report(comparison)


def test_get_best_comparison_strategy_returns_top_strategy() -> None:
    comparison = create_regular_comparison()

    best = get_best_comparison_strategy(comparison)

    assert best == "strategy_a"


def test_get_best_comparison_strategy_returns_none_for_empty_comparison() -> None:
    assert get_best_comparison_strategy({}) is None
