import pandas as pd

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
from src.backtest.engine import run_backtest


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
