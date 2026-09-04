import pandas as pd

from src.pipeline import load_and_prepare_market_data


def test_load_and_prepare_market_data(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [1000, 1100],
        }
    )

    data.to_csv(csv_file, index=False)

    result = load_and_prepare_market_data(str(csv_file))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
