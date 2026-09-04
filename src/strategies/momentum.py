import pandas as pd


def momentum_signal(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Generate a simple momentum trading signal.

    Signal:
        1 -> positive momentum
        0 -> otherwise

    Momentum is calculated from the percentage change
    over the specified lookback window.
    """

    result = df.copy()

    if "close" not in result.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")

    if window <= 0:
        raise ValueError("window must be positive.")

    result["momentum"] = result["close"].pct_change(window)

    result["signal"] = (
        result["momentum"] > 0
    ).astype(int)

    return result
