from collections.abc import Callable

import pandas as pd

from src.backtest.runner import run_strategy


StrategyFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compare_strategies(
    df: pd.DataFrame,
    strategies: dict[str, StrategyFunction],
) -> dict[str, dict]:
    """
    Run and compare multiple strategies on the same input data.

    Parameters
    ----------
    df:
        Market/feature DataFrame used as the common input.

    strategies:
        Dictionary mapping strategy names to strategy functions.

    Returns
    -------
    dict
        A dictionary containing the evaluation report for
        every strategy.

    Example
    -------
    {
        "baseline": {
            "total_return": ...,
            "max_drawdown": ...,
            ...
        }
    }
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(strategies, dict):
        raise TypeError("strategies must be a dictionary.")

    if not strategies:
        raise ValueError("strategies must not be empty.")

    results = {}

    for name, strategy in strategies.items():
        if not isinstance(name, str):
            raise TypeError(
                "Every strategy name must be a string."
            )

        _, report = run_strategy(
            df=df,
            strategy=strategy,
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
