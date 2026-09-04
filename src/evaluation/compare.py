from collections.abc import Callable

import pandas as pd

from src.backtest.runner import run_strategy


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

    Parameters
    ----------
    df:
        Common market/feature DataFrame.

    strategies:
        Dictionary mapping strategy names to strategy functions.

    transaction_cost:
        Proportional transaction cost applied to all strategies.

    slippage:
        Proportional slippage applied to all strategies.

    Returns
    -------
    dict
        Evaluation report for every strategy.
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

        _, report = run_strategy(
            df=df,
            strategy=strategy,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        results[name] = report

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
