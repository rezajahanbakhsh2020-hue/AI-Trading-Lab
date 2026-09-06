import pandas as pd
import pytest

from src.evaluation.compare import (
    rank_walk_forward_strategies,
)


def create_comparison() -> dict[str, dict]:
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
            "window_returns": [
                0.05,
                0.03,
                0.07,
                0.05,
            ],
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
            "window_returns": [
                0.02,
                0.04,
                -0.01,
                0.05,
            ],
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
            "window_returns": [
                0.04,
                0.03,
                0.02,
                0.06,
            ],
            "profitable_windows": 4,
            "losing_windows": 0,
            "positive_window_rate": 1.00,
        },
    }


def test_rank_walk_forward_strategies_by_total_return() -> None:
    comparison = create_comparison()

    ranked = rank_walk_forward_strategies(
        comparison
    )

    assert isinstance(ranked, pd.DataFrame)
    assert ranked["strategy"].tolist() == [
        "strategy_a",
        "strategy_c",
        "strategy_b",
    ]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_rank_walk_forward_strategies_preserves_metrics() -> None:
    comparison = create_comparison()

    ranked = rank_walk_forward_strategies(
        comparison
    )

    assert "total_return" in ranked.columns
    assert "max_drawdown" in ranked.columns
    assert "positive_window_rate" in ranked.columns
    assert "sharpe_ratio" in ranked.columns
    assert ranked.loc[
        ranked["strategy"] == "strategy_a",
        "total_return",
    ].iloc[0] == pytest.approx(0.20)


def test_rank_walk_forward_strategies_supports_ascending_order() -> None:
    comparison = create_comparison()

    ranked = rank_walk_forward_strategies(
        comparison,
        metric="total_return",
        ascending=True,
    )

    assert ranked["strategy"].tolist() == [
        "strategy_b",
        "strategy_c",
        "strategy_a",
    ]


def test_rank_walk_forward_strategies_supports_sharpe_ratio() -> None:
    comparison = create_comparison()

    comparison["strategy_b"]["sharpe_ratio"] = 2.00

    ranked = rank_walk_forward_strategies(
        comparison,
        metric="sharpe_ratio",
    )

    assert ranked.iloc[0]["strategy"] == "strategy_b"
    assert ranked.iloc[0]["rank"] == 1


def test_rank_walk_forward_strategies_prefers_lower_drawdown() -> None:
    comparison = create_comparison()

    comparison["strategy_b"]["total_return"] = 0.20
    comparison["strategy_b"]["positive_window_rate"] = 1.00

    ranked = rank_walk_forward_strategies(
        comparison
    )

    assert ranked["strategy"].tolist() == [
        "strategy_b",
        "strategy_a",
        "strategy_c",
    ]


def test_rank_walk_forward_strategies_is_deterministic_on_ties() -> None:
    comparison = create_comparison()

    comparison["strategy_b"]["total_return"] = 0.20
    comparison["strategy_b"]["positive_window_rate"] = 1.00
    comparison["strategy_b"]["max_drawdown"] = -0.10

    ranked = rank_walk_forward_strategies(
        comparison
    )

    assert ranked["strategy"].tolist() == [
        "strategy_a",
        "strategy_b",
        "strategy_c",
    ]


def test_rank_walk_forward_strategies_empty_comparison() -> None:
    ranked = rank_walk_forward_strategies({})

    assert isinstance(ranked, pd.DataFrame)
    assert ranked.empty


def test_rank_walk_forward_strategies_requires_dictionary() -> None:
    with pytest.raises(TypeError):
        rank_walk_forward_strategies([])  # type: ignore[arg-type]


def test_rank_walk_forward_strategies_requires_known_metric() -> None:
    comparison = create_comparison()

    with pytest.raises(
        ValueError,
        match="Unknown ranking metric",
    ):
        rank_walk_forward_strategies(
            comparison,
            metric="unknown_metric",
        )


def test_rank_walk_forward_strategies_requires_string_metric() -> None:
    comparison = create_comparison()

    with pytest.raises(TypeError):
        rank_walk_forward_strategies(
            comparison,
            metric=123,  # type: ignore[arg-type]
        )
