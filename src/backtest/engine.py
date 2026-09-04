import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_column: str = "signal",
    return_column: str = "return",
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """
    Run a simple backtest without lookahead bias.
    """

    if signal_column not in df.columns:
        raise ValueError(f"Column '{signal_column}' not found.")

    if return_column not in df.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    if slippage < 0:
        raise ValueError("slippage must be non-negative.")

    result = df.copy()

    result["position"] = result[signal_column].shift(1)

    result["strategy_return"] = (
        result["position"] * result[return_column]
    )

    if not result.empty:
        result.loc[result.index[0], "strategy_return"] = float("nan")

    result["turnover"] = (
        result[signal_column]
        .diff()
        .abs()
        .shift(1)
        .fillna(0.0)
    )

    result["trading_cost"] = (
        result["turnover"]
        * (transaction_cost + slippage)
    )

    result["strategy_return"] = (
        result["strategy_return"]
        - result["trading_cost"]
    )

    result["equity"] = (
        1.0 + result["strategy_return"].fillna(0.0)
    ).cumprod()

    return result
