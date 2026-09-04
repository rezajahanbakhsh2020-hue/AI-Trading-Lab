import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]


def validate_market_data(df: pd.DataFrame) -> None:
    """
    Validate the structure and basic integrity of market data.

    Raises:
        ValueError: If the data is invalid.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Market data is empty.")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Market data contains missing values.")

    if (df["open"] <= 0).any():
        raise ValueError("Open prices must be positive.")

    if (df["high"] <= 0).any():
        raise ValueError("High prices must be positive.")

    if (df["low"] <= 0).any():
        raise ValueError("Low prices must be positive.")

    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    if (df["high"] < df["low"]).any():
        raise ValueError("High price cannot be lower than low price.")
