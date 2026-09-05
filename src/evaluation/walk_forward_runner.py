from collections.abc import Callable

import pandas as pd

from src.backtest.engine import run_backtest
from src.evaluation.walk_forward import (
    WalkForwardWindow,
    generate_walk_forward_windows,
)


StrategyFunction = Callable[[pd.DataFrame], pd.DataFrame]


def run_walk_forward_strategy(
    df: pd.DataFrame,
    strategy: StrategyFunction,
    train_size: int,
    test_size: int,
    step: int | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> list[pd.DataFrame]:
    """
    Run a strategy across chronological walk-forward OOS windows.

    The strategy is generated using data available up to the end of
    each test window. The returned results contain only the OOS portion
    of each window.

    The training period is included when generating the strategy signal
    so that rolling indicators and other causal calculations have the
    historical context they need.

    Parameters
    ----------
    df:
        Chronological market/feature DataFrame.

    strategy:
        Function that receives a DataFrame and returns a DataFrame
        containing a 'signal' column.

    train_size:
        Number of observations in each training window.

    test_size:
        Number of observations in each OOS test window.

    step:
        Number of observations to move forward between windows.
        Defaults to test_size.

    transaction_cost:
        Proportional transaction cost.

    slippage:
        Proportional slippage.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame for every OOS window.

        Each result contains:
        - the original OOS market data
        - strategy signal
        - position
        - returns
        - trading costs
        - equity
        - walk-forward window metadata
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not callable(strategy):
        raise TypeError("strategy must be callable.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    if slippage < 0:
        raise ValueError("slippage must be non-negative.")

    windows = generate_walk_forward_windows(
        df=df,
        train_size=train_size,
        test_size=test_size,
        step=step,
    )

    results: list[pd.DataFrame] = []

    for window in windows:
        combined = df.iloc[
            window.train_start : window.test_end
        ].copy()

        strategy_result = strategy(combined)

        if not isinstance(strategy_result, pd.DataFrame):
            raise TypeError(
                "strategy must return a pandas DataFrame."
            )

        if len(strategy_result) != len(combined):
            raise ValueError(
                "strategy must return the same number of rows "
                "as its input."
            )

        if "signal" not in strategy_result.columns:
            raise ValueError(
                "strategy result must contain a 'signal' column."
            )

        backtest_result = run_backtest(
            strategy_result,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        oos_result = backtest_result.iloc[
            window.test_start - window.train_start :
        ].copy()

        if oos_result.empty:
            continue

        oos_result["walk_forward_train_start"] = (
            window.train_start
        )
        oos_result["walk_forward_train_end"] = (
            window.train_end
        )
        oos_result["walk_forward_test_start"] = (
            window.test_start
        )
        oos_result["walk_forward_test_end"] = (
            window.test_end
        )

        oos_result["oos_equity"] = (
            1.0
            + oos_result["strategy_return"].fillna(0.0)
        ).cumprod()

        results.append(oos_result)

    return results
