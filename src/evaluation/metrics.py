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


def sortino_ratio(
    df: pd.DataFrame,
    equity_column: str = "equity",
) -> float:
    """
    Calculate the Sortino ratio from the equity curve.

    Only downside returns are used to measure downside risk.
    """

    if equity_column not in df.columns:
        raise ValueError(f"Column '{equity_column}' not found.")

    if df.empty:
        return 0.0

    equity = df[equity_column].astype(float)

    returns = equity.pct_change().dropna()

    if returns.empty:
        return 0.0

    downside = returns.clip(upper=0.0)

    downside_deviation = (downside.pow(2).mean()) ** 0.5

    if downside_deviation == 0 or pd.isna(downside_deviation):
        return 0.0

    return float(returns.mean() / downside_deviation)


def exposure(
    df: pd.DataFrame,
    signal_column: str = "signal",
) -> float:
    """
    Calculate the percentage of periods spent in the market.

    Returns a value between 0.0 and 1.0.
    """

    if signal_column not in df.columns:
        raise ValueError(f"Column '{signal_column}' not found.")

    if df.empty:
        return 0.0

    signal = df[signal_column]

    return float((signal == 1).mean())


def win_rate(
    df: pd.DataFrame,
    return_column: str = "strategy_return",
) -> float:
    """
    Calculate the percentage of non-zero strategy returns
    that are profitable.

    Returns a value between 0.0 and 1.0.
    """

    if return_column not in df.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    if df.empty:
        return 0.0

    returns = df[return_column].astype(float)

    active_returns = returns[returns != 0]

    if active_returns.empty:
        return 0.0

    winning_returns = active_returns[active_returns > 0]

    return float(len(winning_returns) / len(active_returns))


def profit_factor(
    df: pd.DataFrame,
    return_column: str = "strategy_return",
) -> float:
    """
    Calculate the profit factor.

    Profit factor = gross profit / gross loss.

    Returns 0.0 when there are no winning returns
    or no losing returns.
    """

    if return_column not in df.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    if df.empty:
        return 0.0

    returns = df[return_column].astype(float)

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_profit == 0.0 or gross_loss == 0.0:
        return 0.0

    return float(gross_profit / gross_loss)
