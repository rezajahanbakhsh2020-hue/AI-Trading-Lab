from collections.abc import Callable

import pandas as pd


DataProvider = Callable[[], pd.DataFrame]


def validate_data_provider(
    provider: DataProvider,
) -> None:
    """
    Validate that the supplied object can be used as a data provider.
    """

    if not callable(provider):
        raise TypeError("provider must be callable.")


def load_from_provider(
    provider: DataProvider,
) -> pd.DataFrame:
    """
    Load market data from a callable data provider.
    """

    validate_data_provider(provider)

    df = provider()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Data provider must return a pandas DataFrame."
        )

    return df.copy()
