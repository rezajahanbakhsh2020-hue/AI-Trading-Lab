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

    The strategy holds the asset when signal == 1
    and stays out of the market when signal == 0.

    The signal is executed on the following period to avoid
    look-ahead bias.

    Transaction costs and slippage are applied when the
    executed position changes.

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
        DataFrame containing strategy returns, position,
        turnover, costs, and equity.
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
    position = result[signal_column].shift(1).fillna(0.0)

    result["position"] = position

    # Position change represents trading activity.
    result["turnover"] = (
        result["position"]
        .diff()
        .abs()
        .fillna(0.0)
    )

    # Gross return before trading costs.
    result["gross_strategy_return"] = (
        result["position"] * result[return_column]
    )

    total_cost_rate = transaction_cost + slippage

    result["trading_cost"] = (
        result["turnover"] * total_cost_rate
    )

    # Net strategy return after transaction costs and slippage.
    result["strategy_return"] = (
        result["gross_strategy_return"]
        - result["trading_cost"]
    )

    result["equity"] = (
        1 + result["strategy_return"]
    ).cumprod()

    return result
