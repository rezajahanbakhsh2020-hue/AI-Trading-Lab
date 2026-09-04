import pandas as pd

from src.data.loader import load_csv


def test_load_csv(tmp_path):
    csv_file = tmp_path / "test.csv"

    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "close": [100.0, 101.0],
        }
    ).to_csv(csv_file, index=False)

    df = load_csv(csv_file)

    assert len(df) == 2
    assert list(df.columns) == ["date", "close"]
    assert df["close"].iloc[0] == 100.0
