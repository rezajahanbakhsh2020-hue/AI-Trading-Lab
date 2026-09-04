 import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_column: str = "signal",
    return_column: str = "return",
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """
    Run a simple long-only backtest.

    The signal generated at period t is executed during period t+1
    to avoid look-ahead bias.

    Transaction costs and slippage are charged when the executed
    position changes.

    Parameters
    ----------
    df:
        Input DataFrame containing signal and return columns.

    signal_column:
        Name of the trading signal column.

    return_column:
        Name of the market return column.

    transaction_cost:
        Proportional transaction cost applied to position changes.

    slippage:
        Proportional slippage applied to position changes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing position, turnover, gross strategy
        return, trading cost, net strategy return, and equity.
    """

    result = df.copy()

    if signal_column not in result.columns:
        raise ValueError(f"Column '{signal_column}' not found.")

    if return_column not in result.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    if slippage < 0:
        raise ValueError("slippage must be non-negative.")

    # Execute the signal on the following period.
    result["position"] = (
        result[signal_column]
        .shift(1)
        .fillna(0.0)
    )

    # Turnover is the absolute change in the executed position.
    result["turnover"] = (
        result["position"]
        .diff()
        .abs()
        .fillna(0.0)
    )

    # Gross strategy return before transaction costs and slippage.
    result["gross_strategy_return"] = (
        result["position"] * result[return_column]
    )

    # There is no strategy return for the first period because
    # there is no previous signal available.
    if not result.empty:
        result.loc[
            result.index[0],
            "gross_strategy_return",
        ] = float("nan")

    total_cost_rate = transaction_cost + slippage

    result["trading_cost"] = (
        result["turnover"] * total_cost_rate
    )

    # Net strategy return after trading costs.
    result["strategy_return"] = (
        result["gross_strategy_return"]
        - result["trading_cost"]
    )

    # Equity starts from 1.0. The first NaN strategy return
    # therefore has no effect on initial equity.
    result["equity"] = (
        1.0
        + result["strategy_return"].fillna(0.0)
    ).cumprod()

    return result
