import pandas as pd
import pytest

from src.evaluation.compare import (
    compare_strategies,
    comparison_dataframe,
)
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal


def test_compare_ma_and_momentum_strategies():
    df = pd.DataFrame({
        "close": [
            100,
            101,
            102,
            103,
            102,
            104,
            106,
            105,
            107,
            109,
            108,
            110,
        ],
    })

    strategies = {
        "moving_average": lambda data: baseline_signal(
            data,
            fast_window=2,
            slow_window=4,
        ),
        "momentum": lambda data: momentum_signal(
            data,
            window=3,
        ),
    }

    results = compare_strategies(
        df,
        strategies,
    )

    assert set(results.keys()) == {
        "moving_average",
        "momentum",
    }

    for report in results.values():
        assert isinstance(report, dict)
        assert "total_return" in report
        assert "max_drawdown" in report
        assert "sharpe_ratio" in report
        assert "calmar_ratio" in report
        assert "sortino_ratio" in report
        assert "exposure" in report
        assert "win_rate" in report
        assert "profit_factor" in report


def test_comparison_dataframe_contains_both_strategies():
    comparison = {
        "moving_average": {
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.2,
        },
        "momentum": {
            "total_return": 0.15,
            "max_drawdown": -0.04,
            "sharpe_ratio": 1.4,
        },
    }

    result = comparison_dataframe(comparison)

    assert list(result.index) == [
        "moving_average",
        "momentum",
    ]

    assert result.loc[
        "moving_average",
        "total_return",
    ] == pytest.approx(0.10)

    assert result.loc[
        "momentum",
        "total_return",
    ] == pytest.approx(0.15)


def test_strategies_do_not_modify_common_input():
    df = pd.DataFrame({
        "close": [
            100,
            101,
            102,
            103,
            104,
            105,
        ],
    })

    original = df.copy()

    strategies = {
        "moving_average": lambda data: baseline_signal(
            data,
            fast_window=2,
            slow_window=3,
        ),
        "momentum": lambda data: momentum_signal(
            data,
            window=2,
        ),
    }

    compare_strategies(
        df,
        strategies,
    )

    pd.testing.assert_frame_equal(df, original)
