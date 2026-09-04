import pandas as pd

from configs.strategies import (
    MOVING_AVERAGE_CONFIG,
    MOMENTUM_CONFIG,
    BACKTEST_CONFIG,
)
from src.evaluation.compare import compare_strategies
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal


def run_default_strategy_suite(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Run the default configured strategy suite.

    The suite compares:
        - Moving Average
        - Momentum

    Strategy and backtest parameters are loaded from
    configs/strategies.py.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    strategies = {
        "moving_average": lambda data: baseline_signal(
            data,
            fast_window=MOVING_AVERAGE_CONFIG["fast_window"],
            slow_window=MOVING_AVERAGE_CONFIG["slow_window"],
        ),
        "momentum": lambda data: momentum_signal(
            data,
            window=MOMENTUM_CONFIG["window"],
        ),
    }

    return compare_strategies(
        df=df,
        strategies=strategies,
        transaction_cost=BACKTEST_CONFIG["transaction_cost"],
        slippage=BACKTEST_CONFIG["slippage"],
    )
