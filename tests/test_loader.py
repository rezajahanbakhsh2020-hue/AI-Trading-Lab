import pandas as pd
import pytest

from src.data.loader import load_csv


def test_load_csv(tmp_path):
    csv_file = tmp_path / "xauusd.csv"

    csv_file.write_text(
        "timestamp,open,high,low,close\n"
        "2026-01-01 00:00:00,2650,2660,2640,2655\n"
        "2026-01-01 01:00:00,2655,2670,2650,2665\n"
    )

    result = load_csv(csv_file)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]
    assert len(result) == 2
    assert result["close"].iloc[0] == 2655


def test_load_csv_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_file)


def test_load_csv_missing_required_column(tmp_path):
    csv_file = tmp_path / "invalid.csv"

    csv_file.write_text(
        "timestamp,open,high,low\n"
        "2026-01-01 00:00:00,2650,2660,2640\n"
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(csv_file)
