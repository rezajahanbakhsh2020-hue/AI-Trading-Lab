import pandas as pd


def total_return(
    df: pd.DataFrame,
    equity_column: str = "equity",
) -> float:
    """
    Calculate total strategy return from the equity curve.
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    return float(df[equity_column].iloc[-1] - 1.0)


def max_drawdown(
    df: pd.DataFrame,
    equity_column: str = "equity",
) -> float:
    """
    Calculate maximum drawdown from the equity curve.

    Returns a negative value.
    Example:
        equity: 1.0 -> 1.2 -> 1.1 -> 0.9
        max drawdown = -0.25
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    equity = df[equity_column].astype(float)

    running_peak = equity.cummax()

    drawdown = (equity / running_peak) - 1.0

    return float(drawdown.min())
