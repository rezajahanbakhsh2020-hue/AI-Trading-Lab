import pandas as pd

from configs.data import XAUUSD_CONFIG


def validate_xauusd_data(df: pd.DataFrame) -> None:
    """
    Validate market data according to the XAU/USD configuration.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        XAUUSD_CONFIG["timestamp_column"],
        *XAUUSD_CONFIG["price_columns"],
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

    if df.empty:
        raise ValueError("XAU/USD market data is empty.")

    timestamp_column = XAUUSD_CONFIG["timestamp_column"]

    if df[timestamp_column].isnull().any():
        raise ValueError(
            "XAU/USD timestamp contains missing values."
        )

    for column in XAUUSD_CONFIG["price_columns"]:
        if (df[column] <= 0).any():
            raise ValueError(
                f"XAU/USD {column} prices must be positive."
            )
