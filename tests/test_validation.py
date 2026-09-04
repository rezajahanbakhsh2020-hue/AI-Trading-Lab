import pandas as pd
import pytest

from src.data.validation import validate_market_data


def valid_data():
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "open": [2650.0, 2660.0],
            "high": [2670.0, 2680.0],
            "low": [2640.0, 2650.0],
            "close": [2660.0, 2670.0],
        }
    )


def test_valid_market_data():
    df = valid_data()

    validate_market_data(df)


def test_valid_market_data_allows_volume():
    df = valid_data()
    df["volume"] = [1000, 1200]

    validate_market_data(df)


def test_missing_column():
    df = valid_data().drop(columns=["close"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_market_data(df)


def test_empty_data():
    df = valid_data().iloc[0:0]

    with pytest.raises(ValueError, match="Market data is empty"):
        validate_market_data(df)


def test_missing_values():
    df = valid_data()
    df.loc[0, "close"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_market_data(df)


def test_invalid_high_low():
    df = valid_data()
    df.loc[0, "high"] = 2630.0

    with pytest.raises(ValueError, match="High price cannot be lower"):
        validate_market_data(df)


def test_equal_high_low_is_valid():
    df = valid_data()
    df.loc[0, "high"] = 2640.0
    df.loc[0, "low"] = 2640.0

    validate_market_data(df)


def test_zero_open_price():
    df = valid_data()
    df.loc[0, "open"] = 0.0

    with pytest.raises(ValueError, match="Open prices must be positive"):
        validate_market_data(df)


def test_negative_open_price():
    df = valid_data()
    df.loc[0, "open"] = -1.0

    with pytest.raises(ValueError, match="Open prices must be positive"):
        validate_market_data(df)


def test_zero_high_price():
    df = valid_data()
    df.loc[0, "high"] = 0.0

    with pytest.raises(ValueError, match="High prices must be positive"):
        validate_market_data(df)


def test_zero_low_price():
    df = valid_data()
    df.loc[0, "low"] = 0.0

    with pytest.raises(ValueError, match="Low prices must be positive"):
        validate_market_data(df)


def test_negative_low_price():
    df = valid_data()
    df.loc[0, "low"] = -1.0

    with pytest.raises(ValueError, match="Low prices must be positive"):
        validate_market_data(df)


def test_zero_close_price():
    df = valid_data()
    df.loc[0, "close"] = 0.0

    with pytest.raises(ValueError, match="Close prices must be positive"):
        validate_market_data(df)


def test_negative_close_price():
    df = valid_data()
    df.loc[0, "close"] = -1.0

    with pytest.raises(ValueError, match="Close prices must be positive"):
        validate_market_data(df)
