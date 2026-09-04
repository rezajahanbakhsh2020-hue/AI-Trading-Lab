import pandas as pd

from src.data.provider import DataProvider, load_from_provider
from src.data.validation import validate_market_data
from src.data.preprocessing import standardize_market_data


def load_market_data(
    provider: DataProvider,
) -> pd.DataFrame:
    """
    Load, validate, and standardize market data
    from a configurable data provider.

    Pipeline:
        Provider
        -> Validation
        -> Preprocessing
    """

    df = load_from_provider(provider)

    validate_market_data(df)

    df = standardize_market_data(df)

    return df
