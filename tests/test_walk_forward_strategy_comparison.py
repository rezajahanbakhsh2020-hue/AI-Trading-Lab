import pandas as pd
import pytest

from src.evaluation.compare import (
    compare_walk_forward_strategies,
)


def create_market_data(
    size: int = 30,
) -> pd.DataFrame:
    close = [
        100.0,
        101.0,
        102.0,
        101.0,
        103.0,
        104.0,
        105.0,
        104.0,
        106.0,
        107.0,
        108.0,
        107.0,
        109.0,
        110.0,
        111.0,
        110.0,
        112.0,
        113.0,
        114.0,
        113.0,
        115.0,
        116.0,
        117.0,
        116.0,
        118.0,
        119.0,
        120.0,
        119.0,
        121.0,
        122.0,
    ][:size]

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=size,
                freq="D",
            ),
            "open": [
                value - 0.5
                for value in close
            ],
            "high": [
                value + 1.0
                for value in close
            ],
            "low": [
                value - 1.0
                for value in close
            ],
            "close": close,
        }
    )


def long_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = 1.0
    return result


def flat_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = 0.0
    return result


def test_compare_walk_forward_strategies():
    df = create_market_data()

    comparison = compare_walk_forward_strategies(
        df=df,
        strategies={
            "long": long_strategy,
            "flat": flat_strategy,
        },
        train_size=10,
        test_size=5,
    )

    assert set(comparison) == {
        "long",
        "flat",
    }

    for report in comparison.values():
        assert isinstance(report, dict)

        assert "windows" in report
        assert "observations" in report
        assert "total_return" in report
        assert "max_drawdown" in report
        assert "sharpe_ratio" in report
        assert "calmar_ratio" in report
        assert "sortino_ratio" in report
        assert "exposure" in report
        assert "win_rate" in report
        assert "profit_factor" in report

        assert "window_returns" in report
        assert "profitable_windows" in report
        assert "losing_windows" in report
        assert "positive_window_rate" in report


def test_compare_walk_forward_strategies_uses_same_configuration():
    df = create_market_data()

    comparison = compare_walk_forward_strategies(
        df=df,
        strategies={
            "long": long_strategy,
            "flat": flat_strategy,
        },
        train_size=10,
        test_size=5,
        step=5,
    )

    assert (
        comparison["long"]["windows"]
        == comparison["flat"]["windows"]
    )

    assert (
        comparison["long"]["observations"]
        == comparison["flat"]["observations"]
    )


def test_compare_walk_forward_strategies_flat_strategy():
    df = create_market_data()

    comparison = compare_walk_forward_strategies(
        df=df,
        strategies={
            "flat": flat_strategy,
        },
        train_size=10,
        test_size=5,
    )

    report = comparison["flat"]

    assert report["exposure"] == 0.0
    assert report["total_return"] == 0.0
    assert report["positive_window_rate"] == 0.0


def test_compare_walk_forward_strategies_requires_dictionary():
    df = create_market_data()

    with pytest.raises(TypeError):
        compare_walk_forward_strategies(
            df=df,
            strategies=[],
            train_size=10,
            test_size=5,
        )


def test_compare_walk_forward_strategies_requires_non_empty_dictionary():
    df = create_market_data()

    with pytest.raises(ValueError):
        compare_walk_forward_strategies(
            df=df,
            strategies={},
            train_size=10,
            test_size=5,
        )


def test_compare_walk_forward_strategies_requires_positive_sizes():
    df = create_market_data()

    with pytest.raises(ValueError):
        compare_walk_forward_strategies(
            df=df,
            strategies={
                "long": long_strategy,
            },
            train_size=0,
            test_size=5,
        )

    with pytest.raises(ValueError):
        compare_walk_forward_strategies(
            df=df,
            strategies={
                "long": long_strategy,
            },
            train_size=10,
            test_size=0,
        )
