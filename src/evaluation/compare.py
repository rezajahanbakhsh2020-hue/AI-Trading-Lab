from collections.abc import Callable

import pandas as pd

from src.backtest.runner import run_strategy
from src.evaluation.walk_forward_report import evaluate_walk_forward
from src.evaluation.walk_forward_runner import (
    run_walk_forward_strategy,
)


StrategyFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compare_strategies(
    df: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, dict]:
    """
    Run and compare multiple strategies on the same input data.

    The same transaction cost and slippage assumptions are
    applied to every strategy.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(strategies, dict):
        raise TypeError("strategies must be a dictionary.")

    if not strategies:
        raise ValueError("strategies must not be empty.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    if slippage < 0:
        raise ValueError("slippage must be non-negative.")

    results = {}

    for name, strategy in strategies.items():
        if not isinstance(name, str):
            raise TypeError(
                "Every strategy name must be a string."
            )

        if not callable(strategy):
            raise TypeError(
                "Every strategy must be callable."
            )

        _, report = run_strategy(
            df=df,
            strategy=strategy,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        results[name] = report

    return results


def compare_walk_forward_strategies(
    df: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
    train_size: int,
    test_size: int,
    step: int | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, dict]:
    """
    Run and compare multiple strategies using the same
    chronological walk-forward configuration.

    Every strategy receives the same input data and the same
    train/test/step configuration.

    Returns
    -------
    dict[str, dict]
        Walk-forward evaluation report for every strategy.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(strategies, dict):
        raise TypeError("strategies must be a dictionary.")

    if not strategies:
        raise ValueError("strategies must not be empty.")

    if train_size <= 0:
        raise ValueError(
            "train_size must be positive."
        )

    if test_size <= 0:
        raise ValueError(
            "test_size must be positive."
        )

    if step is not None and step <= 0:
        raise ValueError(
            "step must be positive when provided."
        )

    if transaction_cost < 0:
        raise ValueError(
            "transaction_cost must be non-negative."
        )

    if slippage < 0:
        raise ValueError(
            "slippage must be non-negative."
        )

    for name, strategy in strategies.items():
        if not isinstance(name, str):
            raise TypeError(
                "Every strategy name must be a string."
            )

        if not callable(strategy):
            raise TypeError(
                "Every strategy must be callable."
            )

    results = {}

    for name, strategy in strategies.items():
        oos_results = run_walk_forward_strategy(
            df=df,
            strategy=strategy,
            train_size=train_size,
            test_size=test_size,
            step=step,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        results[name] = evaluate_walk_forward(
            oos_results
        )

    return results


def comparison_dataframe(
    comparison: dict[str, dict],
) -> pd.DataFrame:
    """
    Convert strategy comparison results into a DataFrame.

    Each row represents one strategy.
    Each column represents one evaluation metric.
    """

    if not isinstance(comparison, dict):
        raise TypeError(
            "comparison must be a dictionary."
        )

    if not comparison:
        return pd.DataFrame()

    return pd.DataFrame.from_dict(
        comparison,
        orient="index",
    )


def rank_walk_forward_strategies(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank strategies from a walk-forward comparison.

    Parameters
    ----------
    comparison:
        Output of compare_walk_forward_strategies().

    metric:
        Primary metric used for ranking.

    ascending:
        Ranking direction for the primary metric.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the original metrics plus a
        deterministic rank column.

    Ranking
    -------
    Strategies are primarily ranked by the selected metric.

    When the primary metric is tied:
    1. positive_window_rate is used.
    2. max_drawdown magnitude is used.
    3. strategy name is used as a deterministic final tie-breaker.

    For max_drawdown, lower absolute drawdown is considered better.
    """

    if not isinstance(comparison, dict):
        raise TypeError(
            "comparison must be a dictionary."
        )

    if not comparison:
        return pd.DataFrame()

    if not isinstance(metric, str):
        raise TypeError(
            "metric must be a string."
        )

    dataframe = comparison_dataframe(comparison)

    if metric not in dataframe.columns:
        raise ValueError(
            f"Unknown ranking metric: {metric}"
        )

    required_tie_breakers = {
        "positive_window_rate",
        "max_drawdown",
    }

    missing_tie_breakers = [
        column
        for column in required_tie_breakers
        if column not in dataframe.columns
    ]

    if missing_tie_breakers:
        raise ValueError(
            "Missing required ranking columns: "
            f"{missing_tie_breakers}"
        )

    dataframe = dataframe.copy()
    dataframe.index.name = "strategy"
    dataframe = dataframe.reset_index()

    if metric == "max_drawdown":
        dataframe["_ranking_max_drawdown"] = (
            dataframe["max_drawdown"].abs()
        )
        primary_column = "_ranking_max_drawdown"
        primary_ascending = True
    else:
        primary_column = metric
        primary_ascending = ascending

    dataframe = dataframe.sort_values(
        by=[
            primary_column,
            "positive_window_rate",
            "_ranking_max_drawdown"
            if primary_column != "_ranking_max_drawdown"
            else primary_column,
            "strategy",
        ],
        ascending=[
            primary_ascending,
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if "_ranking_max_drawdown" in dataframe.columns:
        dataframe = dataframe.drop(
            columns=["_ranking_max_drawdown"]
        )

    dataframe["rank"] = dataframe.index + 1

    return dataframe
