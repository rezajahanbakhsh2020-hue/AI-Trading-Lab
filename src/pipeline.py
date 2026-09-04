import pandas as pd

from src.data.loader import load_csv
from src.data.validation import validate_market_data
from src.data.preprocessing import standardize_market_data


def load_and_prepare_market_data(
    path: str,
) -> pd.DataFrame:
    """
    Load, validate, and standardize market data.

    Pipeline:
        CSV -> Loader -> Validation -> Preprocessing
    """

    df = load_csv(path)

    validate_market_data(df)

    df = standardize_market_data(df)

    return df
