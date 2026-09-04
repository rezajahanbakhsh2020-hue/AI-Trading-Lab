import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_column: str = "signal",
    return_column: str = "return",
) -> pd.DataFrame:
    """
    Run a simple long-only backtest.

    The strategy holds the asset when signal == 1
    and stays out of the market when signal == 0.
    """
    result = df.copy()

    if signal_column not in result.columns:
        raise ValueError(f"Column '{signal_column}' not found.")

    if return_column not in result.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    result["strategy_return"] = (
        result[signal_column].shift(1) * result[return_column]
    )

    result["equity"] = (
        1 + result["strategy_return"].fillna(0)
    ).cumprod()

    return result
