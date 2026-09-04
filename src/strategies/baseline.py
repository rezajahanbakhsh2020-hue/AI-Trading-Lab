import pandas as pd


def baseline_signal(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.DataFrame:
    """
    Generate a simple moving-average baseline signal.

    Signal:
        1  -> fast MA above slow MA
        0  -> otherwise
    """
    result = df.copy()

    if "close" not in result.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    result["fast_ma"] = result["close"].rolling(fast_window).mean()
    result["slow_ma"] = result["close"].rolling(slow_window).mean()

    result["signal"] = (
        result["fast_ma"] > result["slow_ma"]
    ).astype(int)

    return result
