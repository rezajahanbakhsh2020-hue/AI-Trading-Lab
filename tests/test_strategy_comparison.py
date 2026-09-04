import pandas as pd
import pytest

from src.backtest.runner import run_strategy
from src.evaluation.compare import (
    compare_strategies,
    comparison_dataframe,
)
from src.strategy.baseline import generate_baseline_signal


def always_long_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test strategy that is always in the market.
    """

    result = df.copy()
    result["signal"] = 1

    return result


def always_flat_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test strategy that is always out of the market.
    """

    result = df.copy()
    result["signal"] = 0

    return result


def test_run_strategy_returns_backtest_and_report():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
    })

    result, report = run_strategy(
        df,
        generate_baseline_signal,
    )

    assert isinstance(result, pd.DataFrame)
    assert isinstance(report, dict)

    assert "signal" in result.columns
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

    assert "total_return" in report
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report
    assert "calmar_ratio" in report
    assert "sortino_ratio" in report
    assert "exposure" in report
    assert "win_rate" in report
    assert "profit_factor" in report


def test_run_strategy_does_not_modify_input():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20],
    })

    original = df.copy()

    run_strategy(
        df,
        always_long_strategy,
    )

    pd.testing.assert_frame_equal(df, original)


def test_run_strategy_rejects_invalid_input():
    with pytest.raises(TypeError):
        run_strategy(
            "not a dataframe",
            always_long_strategy,
        )


def test_run_strategy_rejects_non_callable_strategy():
    df = pd.DataFrame({
        "return": [0.10, -0.05],
    })

    with pytest.raises(TypeError):
        run_strategy(
            df,
            "not a strategy",
        )


def test_run_strategy_requires_signal_column():
    def invalid_strategy(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    df = pd.DataFrame({
        "return": [0.10, -0.05],
    })

    with pytest.raises(ValueError):
        run_strategy(
            df,
            invalid_strategy,
        )


def test_compare_strategies_runs_all_strategies():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
    })

    strategies = {
        "always_long": always_long_strategy,
        "always_flat": always_flat_strategy,
    }

    results = compare_strategies(
        df,
        strategies,
    )

    assert isinstance(results, dict)

    assert set(results.keys()) == {
        "always_long",
        "always_flat",
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


def test_compare_strategies_uses_same_input_data():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
    })

    original = df.copy()

    strategies = {
        "long": always_long_strategy,
        "flat": always_flat_strategy,
    }

    compare_strategies(
        df,
        strategies,
    )

    pd.testing.assert_frame_equal(df, original)


def test_compare_strategies_rejects_empty_strategy_collection():
    df = pd.DataFrame({
        "return": [0.10, -0.05],
    })

    with pytest.raises(ValueError):
        compare_strategies(
            df,
            {},
        )


def test_compare_strategies_rejects_invalid_strategy_collection():
    df = pd.DataFrame({
        "return": [0.10, -0.05],
    })

    with pytest.raises(TypeError):
        compare_strategies(
            df,
            ["strategy"],
        )


def test_comparison_dataframe_creates_expected_structure():
    comparison = {
        "strategy_a": {
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.2,
        },
        "strategy_b": {
            "total_return": 0.20,
            "max_drawdown": -0.10,
            "sharpe_ratio": 1.5,
        },
    }

    result = comparison_dataframe(comparison)

    assert isinstance(result, pd.DataFrame)

    assert list(result.index) == [
        "strategy_a",
        "strategy_b",
    ]

    assert set(result.columns) == {
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
    }

    assert result.loc["strategy_a", "total_return"] == pytest.approx(
        0.10
    )

    assert result.loc["strategy_b", "total_return"] == pytest.approx(
        0.20
    )


def test_comparison_dataframe_handles_empty_comparison():
    result = comparison_dataframe({})

    assert isinstance(result, pd.DataFrame)
    assert result.empty
