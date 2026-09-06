import pandas as pd

from configs.strategies import (
    BACKTEST_CONFIG,
    MOVING_AVERAGE_CONFIG,
)
from src.data.preprocessing import standardize_market_data
from src.evaluation.walk_forward_report import evaluate_walk_forward
from src.evaluation.walk_forward_runner import run_walk_forward_strategy
from src.features.indicators import add_returns
from src.strategies.baseline import baseline_signal


def create_xauusd_walk_forward_sample() -> pd.DataFrame:
    periods = 120

    close = pd.Series(
        [2650.0 + i * 2.0 + (i % 7) * 0.5 for i in range(periods)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=periods,
                freq="D",
            ),
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
        }
    )


def test_xauusd_walk_forward_end_to_end():
    df = create_xauusd_walk_forward_sample()

    df = standardize_market_data(df)
    df = add_returns(df)

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

    assert len(results) == 3

    for result in results:
        assert len(result) == 20
        assert "timestamp" in result.columns
        assert "signal" in result.columns
        assert "position" in result.columns
        assert "strategy_return" in result.columns
        assert "equity" in result.columns
        assert "oos_equity" in result.columns

    combined_timestamps = pd.concat(
        [result["timestamp"] for result in results],
        ignore_index=True,
    )

    assert len(combined_timestamps) == 60
    assert combined_timestamps.is_monotonic_increasing
    assert combined_timestamps.is_unique

    report = evaluate_walk_forward(results)

    assert report["windows"] == 3
    assert report["observations"] == 60

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

    assert expected_metrics.issubset(report.keys())

    for metric in expected_metrics:
        assert isinstance(report[metric], (int, float))
