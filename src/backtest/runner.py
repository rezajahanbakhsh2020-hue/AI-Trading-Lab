from collections.abc import Callable

import pandas as pd

from src.backtest.engine import run_backtest
from src.evaluation.report import evaluate_backtest


StrategyFunction = Callable[[pd.DataFrame], pd.DataFrame]


def run_strategy(
    df: pd.DataFrame,
    strategy: StrategyFunction,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Run one strategy through the standard backtest pipeline.

    Pipeline:
        Input Data
        -> Strategy
        -> Backtest
        -> Evaluation

    The strategy must return a DataFrame containing
    a 'signal' column.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not callable(strategy):
        raise TypeError("strategy must be callable.")

    strategy_result = strategy(df.copy())

    if not isinstance(strategy_result, pd.DataFrame):
        raise TypeError(
            "strategy must return a pandas DataFrame."
        )

    if "signal" not in strategy_result.columns:
        raise ValueError(
            "Strategy output must contain a 'signal' column."
        )

    backtest_result = run_backtest(
        strategy_result,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )

    report = evaluate_backtest(backtest_result)

    return backtest_result, report
