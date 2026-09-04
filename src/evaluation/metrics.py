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
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    equity = df[equity_column].astype(float)

    running_peak = equity.cummax()

    drawdown = (equity / running_peak) - 1.0

    return float(drawdown.min())


def sharpe_ratio(
    df: pd.DataFrame,
    equity_column: str = "equity",
) -> float:
    """
    Calculate the Sharpe ratio from the equity curve.
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    equity = df[equity_column].astype(float)

    returns = equity.pct_change().dropna()

    if returns.empty:
        return 0.0

    std = returns.std()

    if std == 0 or pd.isna(std):
        return 0.0

    return float(returns.mean() / std)


def calmar_ratio(
    df: pd.DataFrame,
    equity_column: str = "equity",
) -> float:
    """
    Calculate the Calmar ratio.

    Calmar ratio = total return / absolute maximum drawdown.

    Returns 0.0 when maximum drawdown is zero.
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    total = total_return(df, equity_column)
    drawdown = max_drawdown(df, equity_column)

    if drawdown == 0.0:
        return 0.0

    return float(total / abs(drawdown))
