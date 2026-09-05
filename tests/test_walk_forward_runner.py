import pandas as pd
import pytest

from src.evaluation.walk_forward_runner import (
    run_walk_forward_strategy,
)


def create_sample_data(size: int = 30) -> pd.DataFrame:
    close = pd.Series(range(100, 100 + size), dtype=float)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=size,
                freq="D",
            ),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "return": close.pct_change().fillna(0.0),
        }
    )


def always_long_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = 1
    return result


def previous_return_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["signal"] = (
        result["return"]
        .shift(1)
        .gt(0)
        .astype(int)
    )

    return result


def test_walk_forward_runs_strategy_on_oos_windows():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
    )

    assert len(results) == 4

    for result in results:
        assert len(result) == 5
        assert "signal" in result.columns
        assert "position" in result.columns
        assert "strategy_return" in result.columns
        assert "oos_equity" in result.columns


def test_walk_forward_oos_results_have_correct_boundaries():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
    )

    assert (
        results[0]["walk_forward_train_start"].iloc[0]
        == 0
    )
    assert (
        results[0]["walk_forward_train_end"].iloc[0]
        == 10
    )
    assert (
        results[0]["walk_forward_test_start"].iloc[0]
        == 10
    )
    assert (
        results[0]["walk_forward_test_end"].iloc[0]
        == 15
    )

    assert (
        results[1]["walk_forward_train_start"].iloc[0]
        == 5
    )
    assert (
        results[1]["walk_forward_test_start"].iloc[0]
        == 15
    )


def test_walk_forward_oos_equity_starts_from_one():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
    )

    for result in results:
        assert result["oos_equity"].iloc[0] == pytest.approx(
            1.0 + result["strategy_return"].iloc[0]
        )


def test_walk_forward_preserves_strategy_history():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=previous_return_strategy,
        train_size=10,
        test_size=5,
    )

    assert len(results) == 4

    for result in results:
        assert len(result) == 5
        assert "signal" in result.columns


def test_walk_forward_rejects_invalid_strategy():
    df = create_sample_data(30)

    with pytest.raises(TypeError):
        run_walk_forward_strategy(
            df=df,
            strategy=None,
            train_size=10,
            test_size=5,
        )


def test_walk_forward_rejects_strategy_without_signal():
    df = create_sample_data(30)

    def invalid_strategy(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    with pytest.raises(ValueError):
        run_walk_forward_strategy(
            df=df,
            strategy=invalid_strategy,
            train_size=10,
            test_size=5,
        )


def test_walk_forward_returns_empty_for_short_data():
    df = create_sample_data(14)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
    )

    assert results == []
