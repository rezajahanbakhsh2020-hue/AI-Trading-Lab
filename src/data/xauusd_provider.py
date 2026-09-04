from datetime import date

import pandas as pd

from src.data.provider import DataProvider


def create_xauusd_csv_provider(
    path: str,
) -> DataProvider:
    """
    Create a provider for a local XAU/USD CSV file.

    The CSV must contain:
        timestamp, open, high, low, close

    Optional columns such as volume are preserved.
    """

    def provider() -> pd.DataFrame:
        df = pd.read_csv(path)

        required_columns = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        }

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        return df.copy()

    return provider


def filter_xauusd_date_range(
    df: pd.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Filter XAU/USD data by date range.

    The end date is inclusive.
    """

    result = df.copy()

    if "timestamp" not in result.columns:
        raise ValueError("Column 'timestamp' not found.")

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
        utc=True,
    )

    result = result.dropna(subset=["timestamp"])

    if start_date is not None:
        result = result[
            result["timestamp"].dt.date >= start_date
        ]

    if end_date is not None:
        result = result[
            result["timestamp"].dt.date <= end_date
        ]

    return result.reset_index(drop=True)
