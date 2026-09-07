from __future__ import annotations

import pandas as pd


def breakout_signal(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Generate a simple breakout trading signal.

    Signal:
        1 -> close breaks above the previous rolling high
        0 -> otherwise

    The rolling high is shifted by one period to prevent
    look-ahead bias.
    """
    result = df.copy()

    required_columns = {"close", "high"}

    missing = required_columns - set(result.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")

    if window <= 0:
        raise ValueError("window must be positive.")

    result["breakout_high"] = (
        result["high"]
        .rolling(window)
        .max()
        .shift(1)
    )

    result["signal"] = (
        result["close"] > result["breakout_high"]
    ).astype(int)

    return result
