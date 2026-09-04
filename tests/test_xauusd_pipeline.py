import pandas as pd

from src.data.xauusd import validate_xauusd_data
from src.data.preprocessing import standardize_market_data
from src.features.indicators import add_returns
from src.evaluation.strategy_suite import run_default_strategy_suite


def create_xauusd_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=60,
                freq="D",
            ),
            "open": [2650 + i for i in range(60)],
            "high": [2655 + i for i in range(60)],
            "low": [2645 + i for i in range(60)],
            "close": [2652 + i for i in range(60)],
        }
    )


def test_xauusd_data_pipeline():
    df = create_xauusd_data()

    validate_xauusd_data(df)

    df = standardize_market_data(df)
    df = add_returns(df)

    results = run_default_strategy_suite(df)

    assert len(df) == 60
    assert "return" in df.columns

    assert set(results.keys()) == {
        "moving_average",
        "momentum",
    }

    for report in results.values():
        assert isinstance(report, dict)
        assert "total_return" in report
