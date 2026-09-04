import pandas as pd
import pytest

from src.data.provider import (
    load_from_provider,
    validate_data_provider,
)


def sample_provider():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-02",
            ],
            "open": [2650.0, 2660.0],
            "high": [2670.0, 2680.0],
            "low": [2640.0, 2650.0],
            "close": [2660.0, 2670.0],
        }
    )


def test_validate_data_provider_accepts_callable():
    validate_data_provider(sample_provider)


def test_validate_data_provider_rejects_non_callable():
    with pytest.raises(
        TypeError,
        match="provider must be callable.",
    ):
        validate_data_provider("invalid")


def test_load_from_provider_returns_dataframe():
    result = load_from_provider(sample_provider)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]


def test_load_from_provider_returns_copy():
    result = load_from_provider(sample_provider)

    result.loc[0, "close"] = 9999.0

    original = sample_provider()

    assert original.loc[0, "close"] == 2660.0


def test_load_from_provider_rejects_invalid_result():
    def invalid_provider():
        return "invalid"

    with pytest.raises(
        TypeError,
        match="Data provider must return a pandas DataFrame.",
    ):
        load_from_provider(invalid_provider)
