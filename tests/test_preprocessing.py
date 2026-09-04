import pandas as pd
import pytest

from src.data.preprocessing import standardize_market_data


def test_standardize_market_data():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
            ],
            "open": [2670.0, 2650.0, 2660.0],
            "high": [2680.0, 2660.0, 2670.0],
            "low": [2660.0, 2640.0, 2650.0],
            "close": [2675.0, 2655.0, 2665.0],
        }
    )

    result = standardize_market_data(df)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["timestamp"].is_monotonic_increasing
    assert len(result) == 3
    assert result["timestamp"].iloc[0] == pd.Timestamp("2026-01-01")


def test_invalid_timestamps_are_removed():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "not-a-date",
                "2026-01-02",
            ],
            "close": [100.0, 101.0, 102.0],
        }
    )

    result = standardize_market_data(df)

    assert len(result) == 2
    assert result["timestamp"].isna().sum() == 0


def test_duplicate_timestamps_are_removed():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "close": [100.0, 101.0, 102.0],
        }
    )

    result = standardize_market_data(df)

    assert len(result) == 2
    assert result["timestamp"].is_unique


def test_missing_timestamp_column():
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="timestamp"):
        standardize_market_data(df)
