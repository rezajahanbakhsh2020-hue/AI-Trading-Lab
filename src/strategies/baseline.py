import pandas as pd


def baseline_signal(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """
    Generate a simple moving-average baseline signal.

    Signal:
        1  -> fast MA above slow MA (BUY)
       -1  -> fast MA below slow MA (SELL)
        0  -> otherwise (NO_TRADE)
    """
    result = df.copy()

    if "close" not in result.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    result["fast_ma"] = result["close"].rolling(fast_window).mean()
    result["slow_ma"] = result["close"].rolling(slow_window).mean()

    valid_ma = result["fast_ma"].notna() & result["slow_ma"].notna()

    signal = pd.Series(0, index=result.index, dtype=int)
    signal.loc[valid_ma & (result["fast_ma"] > result["slow_ma"])] = 1
    signal.loc[valid_ma & (result["fast_ma"] < result["slow_ma"])] = -1

    result["signal"] = signal

    return result


def moving_average_signal(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """
    Generate a simple moving-average crossover signal for the moving_average candidate identity.

    Signal:
        1  -> fast MA above slow MA (BUY)
       -1  -> fast MA below slow MA (SELL)
        0  -> otherwise (NO_TRADE)
    """
    return baseline_signal(df, fast_window=fast_window, slow_window=slow_window)
