import pandas as pd
import pytest

from src.data.loader import load_csv


def test_load_csv(tmp_path):
    csv_file = tmp_path / "test.csv"

    pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.0, 99.0],
            "close": [100.0, 101.0],
        }
    ).to_csv(csv_file, index=False)

    df = load_csv(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]
    assert df["close"].iloc[0] == 100.0


def test_load_csv_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_file)


def test_load_csv_missing_required_column(tmp_path):
    csv_file = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "open": [99.0],
            "high": [101.0],
            "low": [98.0],
        }
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(csv_file)
