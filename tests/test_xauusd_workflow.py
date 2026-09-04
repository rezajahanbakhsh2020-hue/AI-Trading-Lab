import pandas as pd

from src.data.preprocessing import standardize_market_data
from src.features.indicators import add_returns
from src.evaluation.strategy_suite import run_default_strategy_suite


def create_xauusd_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=60,
                freq="D",
            ),
            "open": [
                2650 + i for i in range(60)
            ],
            "high": [
                2655 + i for i in range(60)
            ],
            "low": [
                2645 + i for i in range(60)
            ],
            "close": [
                2652 + i for i in range(60)
            ],
        }
    )


def test_xauusd_sample_has_standard_market_columns():
    df = create_xauusd_sample()

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    assert required_columns.issubset(df.columns)


def test_xauusd_workflow_creates_returns():
    df = create_xauusd_sample()

    df = standardize_market_data(df)
    df = add_returns(df)

    assert "return" in df.columns
    assert len(df) == 60
    assert pd.isna(df["return"].iloc[0])
    assert df["return"].iloc[1] > 0


def test_xauusd_workflow_runs_strategy_suite():
    df = create_xauusd_sample()

    df = standardize_market_data(df)
    df = add_returns(df)

    results = run_default_strategy_suite(df)

    assert set(results.keys()) == {
        "moving_average",
        "momentum",
    }


def test_xauusd_workflow_produces_complete_reports():
    df = create_xauusd_sample()

    df = standardize_market_data(df)
    df = add_returns(df)

    results = run_default_strategy_suite(df)

    expected_metrics = {
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
        "exposure",
        "win_rate",
        "profit_factor",
    }

    for report in results.values():
        assert expected_metrics.issubset(report.keys())
