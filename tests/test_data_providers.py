import pandas as pd
import pytest

from src.data.providers import csv_provider


def test_csv_provider_loads_data(tmp_path):
    path = tmp_path / "xauusd.csv"

    df = pd.DataFrame(
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

    df.to_csv(path, index=False)

    provider = csv_provider(path)
    result = provider()

    pd.testing.assert_frame_equal(result, df)


def test_csv_provider_accepts_string_path(tmp_path):
    path = tmp_path / "xauusd.csv"

    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "open": [2650.0],
            "high": [2670.0],
            "low": [2640.0],
            "close": [2660.0],
        }
    )

    df.to_csv(path, index=False)

    provider = csv_provider(str(path))
    result = provider()

    pd.testing.assert_frame_equal(result, df)


def test_csv_provider_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.csv"

    provider = csv_provider(path)

    with pytest.raises(
        FileNotFoundError,
        match="File not found:",
    ):
        provider()


def test_csv_provider_returns_new_dataframe(tmp_path):
    path = tmp_path / "xauusd.csv"

    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "open": [2650.0],
            "high": [2670.0],
            "low": [2640.0],
            "close": [2660.0],
        }
    )

    df.to_csv(path, index=False)

    provider = csv_provider(path)

    first = provider()
    first.loc[0, "close"] = 9999.0

    second = provider()

    assert second.loc[0, "close"] == 2660.0
