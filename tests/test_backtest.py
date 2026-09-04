import pandas as pd

from src.backtest.engine import run_backtest
from src.features.indicators import add_returns
from src.strategies.baseline import baseline_signal


def test_baseline_signal():
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105],
        }
    )

    result = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    assert "fast_ma" in result.columns
    assert "slow_ma" in result.columns
    assert "signal" in result.columns

    assert result["signal"].iloc[-1] == 1


def test_run_backtest():
    df = pd.DataFrame(
        {
            "signal": [0, 1, 1],
            "return": [0.00, 0.10, -0.05],
        }
    )

    result = run_backtest(df)

    assert "strategy_return" in result.columns
    assert "equity" in result.columns

    assert result["strategy_return"].iloc[1] == 0.0
    assert result["strategy_return"].iloc[2] == -0.05


def test_full_backtest_pipeline():
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 104.0, 106.0, 108.0],
        }
    )

    df = add_returns(df)
    df = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )
    result = run_backtest(df)

    assert "return" in result.columns
    assert "signal" in result.columns
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

    assert result["equity"].iloc[-1] > 0
