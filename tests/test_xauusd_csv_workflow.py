import pandas as pd

from src.data.market_data import load_market_data
from src.data.providers import csv_provider
from src.features.indicators import add_returns
from src.evaluation.strategy_suite import run_default_strategy_suite


def test_xauusd_csv_full_workflow(tmp_path):
    path = tmp_path / "xauusd.csv"

    source = pd.DataFrame(
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

    source.to_csv(path, index=False)

    provider = csv_provider(path)

    df = load_market_data(provider)
    df = add_returns(df)

    results = run_default_strategy_suite(df)

    assert len(df) == 3

    assert list(df["timestamp"]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
    ]

    assert "return" in df.columns

    assert set(results.keys()) == {
        "moving_average",
        "momentum",
    }

    for report in results.values():
        assert isinstance(report, dict)
        assert "total_return" in report
        assert "max_drawdown" in report
