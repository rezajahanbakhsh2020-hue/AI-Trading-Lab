import pandas as pd
import pytest

from src.evaluation.compare import (
    rank_walk_forward_strategies,
)


def create_reports():
    return {
        "strategy_a": {
            "total_return": 0.30,
            "max_drawdown": -0.10,
            "sharpe_ratio": 2.0,
            "positive_window_rate": 0.80,
        },
        "strategy_b": {
            "total_return": 0.20,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.5,
            "positive_window_rate": 0.70,
        },
        "strategy_c": {
            "total_return": 0.10,
            "max_drawdown": -0.20,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.60,
        },
    }


def test_rank_by_total_return():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert list(result["strategy"]) == [
        "strategy_a",
        "strategy_b",
        "strategy_c",
    ]


def test_rank_by_sharpe_ratio():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="sharpe_ratio",
    )

    assert list(result["strategy"]) == [
        "strategy_a",
        "strategy_b",
        "strategy_c",
    ]


def test_rank_by_positive_window_rate():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="positive_window_rate",
    )

    assert list(result["strategy"]) == [
        "strategy_a",
        "strategy_b",
        "strategy_c",
    ]


def test_rank_by_max_drawdown_prefers_lower_absolute_drawdown():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="max_drawdown",
    )

    assert list(result["strategy"]) == [
        "strategy_b",
        "strategy_a",
        "strategy_c",
    ]


def test_ranking_preserves_all_metrics():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert "total_return" in result.columns
    assert "max_drawdown" in result.columns
    assert "sharpe_ratio" in result.columns
    assert "positive_window_rate" in result.columns

    assert result.loc[
        result["strategy"] == "strategy_a",
        "total_return",
    ].iloc[0] == pytest.approx(0.30)


def test_ranking_returns_dataframe():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert isinstance(result, pd.DataFrame)


def test_ranking_has_rank_column():
    reports = create_reports()

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert "rank" in result.columns
    assert list(result["rank"]) == [1, 2, 3]


def test_ranking_is_deterministic_for_ties():
    reports = {
        "zeta": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.60,
        },
        "alpha": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.60,
        },
    }

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert list(result["strategy"]) == [
        "alpha",
        "zeta",
    ]


def test_ranking_empty_reports():
    result = rank_walk_forward_strategies(
        {},
        primary_metric="total_return",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_ranking_rejects_invalid_reports_type():
    with pytest.raises(TypeError):
        rank_walk_forward_strategies(
            [],
            primary_metric="total_return",
        )


def test_ranking_rejects_unknown_metric():
    reports = create_reports()

    with pytest.raises(ValueError):
        rank_walk_forward_strategies(
            reports,
            primary_metric="unknown_metric",
        )


def test_ranking_rejects_non_string_metric():
    reports = create_reports()

    with pytest.raises(TypeError):
        rank_walk_forward_strategies(
            reports,
            primary_metric=None,
        )


def test_ranking_rejects_missing_primary_metric():
    reports = {
        "strategy_a": {
            "max_drawdown": -0.10,
            "sharpe_ratio": 2.0,
            "positive_window_rate": 0.80,
        }
    }

    with pytest.raises(ValueError):
        rank_walk_forward_strategies(
            reports,
            primary_metric="total_return",
        )


def test_ranking_handles_negative_returns():
    reports = {
        "strategy_a": {
            "total_return": -0.05,
            "max_drawdown": -0.10,
            "sharpe_ratio": -0.5,
            "positive_window_rate": 0.40,
        },
        "strategy_b": {
            "total_return": -0.10,
            "max_drawdown": -0.20,
            "sharpe_ratio": -1.0,
            "positive_window_rate": 0.30,
        },
    }

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert list(result["strategy"]) == [
        "strategy_a",
        "strategy_b",
    ]


def test_ranking_prefers_positive_window_rate_after_primary_metric_tie():
    reports = {
        "strategy_a": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.60,
        },
        "strategy_b": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.80,
        },
    }

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert list(result["strategy"]) == [
        "strategy_b",
        "strategy_a",
    ]


def test_ranking_prefers_lower_absolute_drawdown_after_other_ties():
    reports = {
        "strategy_a": {
            "total_return": 0.20,
            "max_drawdown": -0.20,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.80,
        },
        "strategy_b": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.0,
            "positive_window_rate": 0.80,
        },
    }

    result = rank_walk_forward_strategies(
        reports,
        primary_metric="total_return",
    )

    assert list(result["strategy"]) == [
        "strategy_b",
        "strategy_a",
    ]
