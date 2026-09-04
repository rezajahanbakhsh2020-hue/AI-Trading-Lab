import pandas as pd


def standardize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize market data for downstream processing.

    - Converts timestamp to pandas datetime.
    - Removes rows with invalid timestamps.
    - Sorts data chronologically.
    - Removes duplicate timestamps.
    - Resets the DataFrame index.
    """

    result = df.copy()

    if "timestamp" not in result.columns:
        raise ValueError("Column 'timestamp' not found.")

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result = result.dropna(subset=["timestamp"])

    result = result.sort_values("timestamp")

    result = result.drop_duplicates(
        subset=["timestamp"],
        keep="first",
    )

    result = result.reset_index(drop=True)

    return result
