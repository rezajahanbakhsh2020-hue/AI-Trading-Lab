import pandas as pd
import pytest

from src.evaluation.strategy_evaluator import (
    evaluate_all_strategies,
    evaluate_strategy,
    select_top_strategies,
)
from src.strategies.registry import StrategyRegistry, StrategySpec


def sample_market_data(rows: int = 80) -> pd.DataFrame:
    close = [
        100.0,
        101.0,
        100.5,
        102.0,
        103.0,
        102.5,
        104.0,
        105.0,
    ]

    values = [
        close[index % len(close)] + (index // len(close))
        for index in range(rows)
    ]

    return pd.DataFrame(
        {
            "timestamp": range(rows),
            "open": values,
            "high": [value + 1.0 for value in values],
            "low": [value - 1.0 for value in values],
            "close": values,
        }
    )


def test_evaluate_one_registered_strategy():
    df = sample_market_data()

    result = evaluate_strategy(
        df,
        "momentum",
        strategy_kwargs={"window": 5},
    )

    assert result.name == "momentum"
    assert result.category == "momentum"
    assert result.observations == len(df)
    assert isinstance(result.total_return, float)
    assert isinstance(result.max_drawdown, float)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.ranking_score, float)


def test_evaluate_all_returns_all_registered_strategies():
    df = sample_market_data()

    result = evaluate_all_strategies(
        df,
        strategy_kwargs={
            "momentum": {"window": 5},
            "baseline": {
                "fast_window": 5,
                "slow_window": 20,
            },
            "breakout": {"window": 10},
        },
    )

    assert set(result["name"]) == {
        "baseline",
        "breakout",
        "momentum",
    }

    assert len(result) == 3


def test_evaluation_is_ranked_by_score():
    df = sample_market_data()

    result = evaluate_all_strategies(
        df,
        strategy_kwargs={
            "momentum": {"window": 5},
            "baseline": {
                "fast_window": 5,
                "slow_window": 20,
            },
            "breakout": {"window": 10},
        },
    )

    scores = result["ranking_score"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_select_top_three_returns_at_most_three():
    df = sample_market_data()

    evaluations = evaluate_all_strategies(
        df,
        strategy_kwargs={
            "momentum": {"window": 5},
            "baseline": {
                "fast_window": 5,
                "slow_window": 20,
            },
            "breakout": {"window": 10},
        },
    )

    selected = select_top_strategies(
        evaluations,
        top_n=3,
    )

    assert len(selected) == 3
    assert list(selected["ranking_score"]) == list(
        evaluations["ranking_score"]
    )


def test_select_top_strategies_can_select_one():
    df = sample_market_data()

    evaluations = evaluate_all_strategies(
        df,
        strategy_kwargs={
            "momentum": {"window": 5},
            "baseline": {
                "fast_window": 5,
                "slow_window": 20,
            },
            "breakout": {"window": 10},
        },
    )

    selected = select_top_strategies(
        evaluations,
        top_n=1,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["name"] == evaluations.iloc[0]["name"]


def test_unknown_strategy_is_rejected():
    df = sample_market_data()

    with pytest.raises(KeyError):
        evaluate_strategy(
            df,
            "unknown",
        )


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError):
        evaluate_all_strategies(
            pd.DataFrame(),
        )


def test_invalid_strategy_kwargs_are_rejected():
    df = sample_market_data()

    with pytest.raises(TypeError):
        evaluate_strategy(
            df,
            "momentum",
            strategy_kwargs=["invalid"],
        )


def test_invalid_top_n_is_rejected():
    df = sample_market_data()

    evaluations = evaluate_all_strategies(df)

    with pytest.raises(ValueError):
        select_top_strategies(
            evaluations,
            top_n=0,
        )


def test_custom_registry_can_be_evaluated():
    def always_long(df):
        result = df.copy()
        result["signal"] = 1
        return result

    registry = StrategyRegistry(
        (
            StrategySpec(
                name="always_long",
                category="test",
                function=always_long,
            ),
        )
    )

    result = evaluate_all_strategies(
        sample_market_data(),
        registry=registry,
    )

    assert len(result) == 1
    assert result.iloc[0]["name"] == "always_long"
    assert result.iloc[0]["observations"] == 80


def test_transaction_cost_and_slippage_are_passed_to_backtest():
    df = sample_market_data()

    free = evaluate_strategy(
        df,
        "momentum",
        strategy_kwargs={"window": 5},
    )

    costly = evaluate_strategy(
        df,
        "momentum",
        strategy_kwargs={"window": 5},
        transaction_cost=0.001,
        slippage=0.001,
    )

    assert costly.total_return <= free.total_return


def test_evaluation_does_not_require_precomputed_returns():
    df = sample_market_data()

    assert "return" not in df.columns

    result = evaluate_strategy(
        df,
        "breakout",
        strategy_kwargs={"window": 10},
    )

    assert result.observations == len(df)
