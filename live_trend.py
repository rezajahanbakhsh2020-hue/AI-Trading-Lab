from __future__ import annotations

import pandas as pd

from src.strategies.baseline import baseline_signal


DEFAULT_FAST_WINDOW = 20
DEFAULT_SLOW_WINDOW = 50


def generate_live_trend(
    data: pd.DataFrame,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
) -> pd.DataFrame:
    """
    Apply the existing AI-Trading-Lab moving-average baseline
    to live market data and expose a human-readable trend state.

    Trend:
        UP -> fast MA is above slow MA
        DOWN -> fast MA is below slow MA
        INSUFFICIENT DATA -> slow MA is not available yet

    The existing baseline strategy remains the source of truth
    for the moving-average calculations.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if "close" not in data.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    if not isinstance(fast_window, int):
        raise TypeError("fast_window must be an integer.")

    if not isinstance(slow_window, int):
        raise TypeError("slow_window must be an integer.")

    if fast_window <= 0:
        raise ValueError("fast_window must be positive.")

    if slow_window <= 0:
        raise ValueError("slow_window must be positive.")

    if fast_window >= slow_window:
        raise ValueError(
            "fast_window must be smaller than slow_window."
        )

    result = baseline_signal(
        data.copy(),
        fast_window=fast_window,
        slow_window=slow_window,
    )

    result["trend"] = "INSUFFICIENT DATA"

    valid_ma = (
        result["fast_ma"].notna()
        & result["slow_ma"].notna()
    )

    result.loc[
        valid_ma & (result["fast_ma"] > result["slow_ma"]),
        "trend",
    ] = "UP"

    result.loc[
        valid_ma & (result["fast_ma"] < result["slow_ma"]),
        "trend",
    ] = "DOWN"

    return result


def build_live_trend_snapshot(
    data: pd.DataFrame,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
) -> dict:
    """
    Build a compact human-readable snapshot from the latest trend row.
    """

    result = generate_live_trend(
        data=data,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    latest = result.iloc[-1]

    fast_ma = latest["fast_ma"]
    slow_ma = latest["slow_ma"]

    snapshot = {
        "trend": str(latest["trend"]),
        "fast_window": fast_window,
        "slow_window": slow_window,
        "fast_ma": (
            None
            if pd.isna(fast_ma)
            else float(fast_ma)
        ),
        "slow_ma": (
            None
            if pd.isna(slow_ma)
            else float(slow_ma)
        ),
    }

    close_value = latest["close"]

    if pd.isna(close_value):
        raise ValueError("Latest close price is invalid.")

    snapshot["close"] = float(close_value)

    if "openTime" in latest.index:
        timestamp = pd.to_datetime(
            latest["openTime"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(timestamp):
            raise ValueError("Latest openTime value is invalid.")

        snapshot["timestamp"] = timestamp.isoformat()

    elif "timestamp" in latest.index:
        timestamp = pd.to_datetime(
            latest["timestamp"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(timestamp):
            raise ValueError("Latest timestamp value is invalid.")

        snapshot["timestamp"] = timestamp.isoformat()

    return snapshot
