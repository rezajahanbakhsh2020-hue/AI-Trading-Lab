from __future__ import annotations

import pandas as pd


def build_candlestick_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a standardized candlestick series.

    Required columns:
        timestamp
        open
        high
        low
        close

    Optional columns are preserved.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
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

    result = result.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    return result


def build_signal_markers(
    df: pd.DataFrame,
    signal_column: str = "signal",
) -> pd.DataFrame:
    """
    Build Buy/Sell marker data for chart visualization.

    Signal convention:
        1  -> Buy
       -1  -> Sell
        0  -> No marker
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        "timestamp",
        "close",
        signal_column,
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    result = df[
        [
            "timestamp",
            "close",
            signal_column,
        ]
    ].copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains invalid datetime values."
        )

    result["marker"] = result[signal_column].map(
        {
            1: "buy",
            -1: "sell",
        }
    )

    result = result[
        result[signal_column].isin([1, -1])
    ].copy()

    result = result.rename(
        columns={
            "close": "price",
        }
    )

    return result[
        [
            "timestamp",
            "price",
            "marker",
        ]
    ].reset_index(drop=True)


def build_volume_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a standardized volume series when volume exists.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        "timestamp",
        "volume",
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    result = df[
        [
            "timestamp",
            "volume",
        ]
    ].copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["volume"] = pd.to_numeric(
        result["volume"],
        errors="coerce",
    )

    if result[["timestamp", "volume"]].isna().any().any():
        raise ValueError(
            "timestamp or volume contains invalid values."
        )

    return (
        result
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
