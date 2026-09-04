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
