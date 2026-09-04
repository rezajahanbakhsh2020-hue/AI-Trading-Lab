import pandas as pd

from src.evaluation.metrics import (
    total_return,
    max_drawdown,
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    exposure,
    win_rate,
    profit_factor,
)


def evaluate_backtest(
    df: pd.DataFrame,
    equity_column: str = "equity",
    signal_column: str = "signal",
    return_column: str = "strategy_return",
) -> dict:
    """
    Calculate a complete evaluation report for a backtest result.

    Returns all supported performance metrics in a dictionary.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return {
        "total_return": total_return(df, equity_column),
        "max_drawdown": max_drawdown(df, equity_column),
        "sharpe_ratio": sharpe_ratio(df, equity_column),
        "calmar_ratio": calmar_ratio(df, equity_column),
        "sortino_ratio": sortino_ratio(df, equity_column),
        "exposure": exposure(df, signal_column),
        "win_rate": win_rate(df, return_column),
        "profit_factor": profit_factor(df, return_column),
    }
