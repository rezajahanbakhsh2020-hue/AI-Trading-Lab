import pandas as pd


def add_returns(df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """
    Add simple percentage returns to the DataFrame.
    """
    result = df.copy()

    if price_column not in result.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame.")

    result["return"] = result[price_column].pct_change()

    return result
