from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
}


def prepare_chart_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare standardized market data for charting.

    The function creates an independent copy of the input data,
    validates the required OHLC structure, normalizes timestamps,
    sorts rows chronologically, and preserves optional columns
    such as volume, signal, and volatility regime.

    Required columns:
        timestamp
        open
        high
        low
        close

    Optional columns are preserved automatically.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains invalid datetime values."
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    if "volume" in result.columns:
        numeric_columns.append("volume")

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if result[numeric_columns].isna().any().any():
        raise ValueError(
            "OHLC or optional numeric columns contain "
            "invalid numeric values."
        )

    if (
        (result["high"] < result["low"])
        .any()
    ):
        raise ValueError(
            "high cannot be lower than low."
        )

    if (
        (result["high"] < result["open"])
        | (result["high"] < result["close"])
    ).any():
        raise ValueError(
            "high must be greater than or equal to "
            "open and close."
        )

    if (
        (result["low"] > result["open"])
        | (result["low"] > result["close"])
    ).any():
        raise ValueError(
            "low must be lower than or equal to "
            "open and close."
        )

    result = (
        result
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    return result
