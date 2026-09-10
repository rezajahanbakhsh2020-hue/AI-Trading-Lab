from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")
SYMBOL = "XAUTUSDT"


def normalize_xaut_usdt_ohlc(
    data: pd.DataFrame | Iterable[dict],
) -> pd.DataFrame:
    """
    Normalize XAUT/USDT OHLC data into the project's standard schema.

    The adapter intentionally performs no strategy calculations and adds
    no external dependency. It only validates and normalizes market data
    so the existing trading engine can consume it.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"XAUT/USDT data is missing required columns: {missing}"
        )

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()

    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame.empty:
        raise ValueError("XAUT/USDT data cannot be empty")

    if frame[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise ValueError("XAUT/USDT OHLC data contains missing numeric values")

    timestamps = frame["timestamp"]

    if pd.api.types.is_numeric_dtype(timestamps):
        frame["timestamp"] = pd.to_datetime(
            timestamps,
            unit="s",
            utc=True,
        )
    else:
        frame["timestamp"] = pd.to_datetime(
            timestamps,
            utc=True,
            errors="raise",
        )

    if frame["timestamp"].duplicated().any():
        raise ValueError("XAUT/USDT timestamps must be unique")

    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp").reset_index(drop=True)

    invalid_ohlc = (
        (frame["high"] < frame["low"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
    )

    if invalid_ohlc.any():
        raise ValueError("XAUT/USDT data contains invalid OHLC relationships")

    return frame


def xaut_usdt_symbol() -> str:
    """Return the canonical symbol used by the adapter."""
    return SYMBOL
