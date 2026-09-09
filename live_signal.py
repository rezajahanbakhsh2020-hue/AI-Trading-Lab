from __future__ import annotations

import pandas as pd

from src.strategies.momentum import momentum_signal


DEFAULT_MOMENTUM_WINDOW = 10


def generate_live_signal(
    data: pd.DataFrame,
    window: int = DEFAULT_MOMENTUM_WINDOW,
) -> pd.DataFrame:
    """
    Apply the existing AI-Trading-Lab momentum strategy to live market data.

    The existing strategy remains the single source of truth for the
    momentum calculation and numeric signal.

    Signal:
        1 -> BUY
        0 -> NO TRADE

    No SELL signal is invented here because the existing momentum strategy
    currently exposes only 1/0 semantics.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(window, int):
        raise TypeError("window must be an integer.")

    if window <= 0:
        raise ValueError("window must be positive.")

    if "close" not in data.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    result = momentum_signal(
        data.copy(),
        window=window,
    )

    result["signal_label"] = result["signal"].map(
        {
            1: "BUY",
            0: "NO TRADE",
        }
    )

    return result


def build_live_signal_snapshot(
    data: pd.DataFrame,
    window: int = DEFAULT_MOMENTUM_WINDOW,
) -> dict:
    """
    Build a compact, human-readable snapshot from the latest strategy row.

    The returned values come directly from the existing momentum strategy.
    """

    result = generate_live_signal(
        data=data,
        window=window,
    )

    latest = result.iloc[-1]

    signal_value = int(latest["signal"])
    signal_label = str(latest["signal_label"])

    momentum_value = latest["momentum"]

    if pd.isna(momentum_value):
        momentum = None
    else:
        momentum = float(momentum_value)

    close_value = latest["close"]

    if pd.isna(close_value):
        raise ValueError("Latest close price is invalid.")

    snapshot = {
        "signal": signal_value,
        "signal_label": signal_label,
        "momentum": momentum,
        "close": float(close_value),
        "strategy": "momentum",
        "window": window,
    }

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
