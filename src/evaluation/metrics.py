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

    Returns a negative value representing the largest percentage loss
    from a previous equity peak.
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    equity = df[equity_column]

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0

    return float(drawdown.min())


def win_rate(
    df: pd.DataFrame,
    return_column: str = "strategy_return",
) -> float:
    """
    Calculate the percentage of non-zero strategy returns that are profitable.
    """

    if return_column not in df.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    returns = df[return_column].dropna()
    trades = returns[returns != 0]

    if trades.empty:
        return 0.0

    return float((trades > 0).mean())


def sharpe_ratio(
    df: pd.DataFrame,
    return_column: str = "strategy_return",
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate the Sharpe ratio from strategy returns.

    Uses the sample standard deviation of returns.
    """

    if return_column not in df.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    returns = df[return_column].dropna()

    if returns.empty:
        return 0.0

    excess_returns = returns - risk_free_rate

    std = excess_returns.std()

    if std == 0:
        return 0.0

    return float(excess_returns.mean() / std)
