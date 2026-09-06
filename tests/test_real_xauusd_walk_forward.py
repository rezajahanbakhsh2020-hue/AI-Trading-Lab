from pathlib import Path

import pandas as pd

from configs.strategies import (
    BACKTEST_CONFIG,
    MOVING_AVERAGE_CONFIG,
)
from src.data.preprocessing import standardize_market_data
from src.data.validation import validate_xauusd_data
from src.evaluation.walk_forward_report import evaluate_walk_forward
from src.evaluation.walk_forward_runner import run_walk_forward_strategy
from src.features.indicators import add_returns
from src.strategies.baseline import baseline_signal


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)


def test_real_xauusd_walk_forward_pipeline():
    df = pd.read_csv(DATA_PATH)

    validate_xauusd_data(df)

    df = standardize_market_data(df)
    df = add_returns(df)

    assert not df.empty
    assert len(df) >= 100
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique

    strategy = lambda data: baseline_signal(
        data,
        fast_window=MOVING_AVERAGE_CONFIG["fast_window"],
        slow_window=MOVING_AVERAGE_CONFIG["slow_window"],
    )

    results = run_walk_forward_strategy(
        df=df,
        strategy=strategy,
        train_size=60,
        test_size=20,
        step=20,
        transaction_cost=BACKTEST_CONFIG["transaction_cost"],
        slippage=BACKTEST_CONFIG["slippage"],
    )

    assert results

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    assert not combined.empty
    assert "timestamp" in combined.columns
    assert "signal" in combined.columns
    assert "position" in combined.columns
    assert "strategy_return" in combined.columns
    assert "equity" in combined.columns
    assert "oos_equity" in combined.columns

    report = evaluate_walk_forward(results)

    assert report["windows"] == len(results)
    assert report["observations"] == len(combined)

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

    assert expected_metrics.issubset(report)

    for metric in expected_metrics:
        assert isinstance(
            report[metric],
            (int, float),
        )
