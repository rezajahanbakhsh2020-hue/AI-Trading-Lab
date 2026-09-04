import pandas as pd
import pytest

from src.data.market_data import load_market_data


def sample_market_provider():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
            ],
            "open": [
                2670.0,
                2650.0,
                2660.0,
            ],
            "high": [
                2680.0,
                2660.0,
                2670.0,
            ],
            "low": [
                2660.0,
                2640.0,
                2650.0,
            ],
            "close": [
                2675.0,
                2655.0,
                2665.0,
            ],
        }
    )


def test_load_market_data_validates_and_standardizes():
    result = load_market_data(sample_market_provider)

    assert len(result) == 3
    assert list(result["timestamp"]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
    ]


def test_load_market_data_preserves_optional_volume():
    def provider():
        df = sample_market_provider()
        df["volume"] = [1000, 1100, 1200]
        return df

    result = load_market_data(provider)

    assert "volume" in result.columns


def test_load_market_data_rejects_invalid_market_data():
    def provider():
        return pd.DataFrame(
            {
                "timestamp": ["2026-01-01"],
                "open": [2650.0],
                "high": [2670.0],
                "low": [2640.0],
            }
        )

    with pytest.raises(
        ValueError,
        match="Missing required columns:",
    ):
        load_market_data(provider)


def test_load_market_data_does_not_modify_provider_data():
    original = sample_market_provider()

    def provider():
        return original.copy()

    load_market_data(provider)

    assert list(original["timestamp"]) == [
        "2026-01-03",
        "2026-01-01",
        "2026-01-02",
    ]
