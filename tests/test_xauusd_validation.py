import pandas as pd
import pytest

from src.data.xauusd import validate_xauusd_data


def valid_xauusd_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
            "open": [2650.0, 2660.0, 2670.0],
            "high": [2660.0, 2670.0, 2680.0],
            "low": [2640.0, 2650.0, 2660.0],
            "close": [2655.0, 2665.0, 2675.0],
        }
    )


def test_valid_xauusd_data():
    df = valid_xauusd_data()

    validate_xauusd_data(df)


def test_xauusd_data_accepts_volume():
    df = valid_xauusd_data()

    df["volume"] = [1000, 1100, 1200]

    validate_xauusd_data(df)


def test_xauusd_data_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match="df must be a pandas DataFrame.",
    ):
        validate_xauusd_data("invalid")


def test_xauusd_data_rejects_empty_data():
    df = valid_xauusd_data().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="XAU/USD market data is empty.",
    ):
        validate_xauusd_data(df)


def test_xauusd_data_rejects_missing_column():
    df = valid_xauusd_data().drop(columns=["close"])

    with pytest.raises(
        ValueError,
        match="Missing required columns:",
    ):
        validate_xauusd_data(df)


def test_xauusd_data_rejects_missing_timestamp():
    df = valid_xauusd_data()
    df.loc[1, "timestamp"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="XAU/USD timestamp contains missing values.",
    ):
        validate_xauusd_data(df)


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
    ],
)
def test_xauusd_data_rejects_non_positive_prices(column):
    df = valid_xauusd_data()
    df.loc[1, column] = 0.0

    with pytest.raises(
        ValueError,
        match=f"XAU/USD {column} prices must be positive.",
    ):
        validate_xauusd_data(df)
