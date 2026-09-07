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


def alternating_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["signal"] = (
        pd.Series(
            range(len(result)),
            index=result.index,
        )
        .mod(2)
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

    def invalid_strategy(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
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


def test_walk_forward_supports_custom_step_smaller_than_test_size():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        step=2,
    )

    assert len(results) == 8

    for result in results:
        assert len(result) == 5


def test_walk_forward_custom_step_preserves_oos_boundaries():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        step=2,
    )

    assert (
        results[0]["walk_forward_test_start"].iloc[0]
        == 10
    )

    assert (
        results[1]["walk_forward_test_start"].iloc[0]
        == 12
    )

    assert (
        results[2]["walk_forward_test_start"].iloc[0]
        == 14
    )


def test_walk_forward_supports_step_larger_than_test_size():
    df = create_sample_data(30)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        step=8,
    )

    assert len(results) == 2

    assert (
        results[0]["walk_forward_test_start"].iloc[0]
        == 10
    )

    assert (
        results[1]["walk_forward_test_start"].iloc[0]
        == 18
    )


def test_walk_forward_applies_transaction_cost():
    df = create_sample_data(30)

    results_without_cost = run_walk_forward_strategy(
        df=df,
        strategy=alternating_strategy,
        train_size=10,
        test_size=5,
        transaction_cost=0.0,
    )

    results_with_cost = run_walk_forward_strategy(
        df=df,
        strategy=alternating_strategy,
        train_size=10,
        test_size=5,
        transaction_cost=0.01,
    )

    for without_cost, with_cost in zip(
        results_without_cost,
        results_with_cost,
    ):
        assert (
            with_cost["trading_cost"].sum()
            > without_cost["trading_cost"].sum()
        )

        assert (
            with_cost["strategy_return"].sum()
            < without_cost["strategy_return"].sum()
        )


def test_walk_forward_applies_slippage():
    df = create_sample_data(30)

    results_without_slippage = run_walk_forward_strategy(
        df=df,
        strategy=alternating_strategy,
        train_size=10,
        test_size=5,
        slippage=0.0,
    )

    results_with_slippage = run_walk_forward_strategy(
        df=df,
        strategy=alternating_strategy,
        train_size=10,
        test_size=5,
        slippage=0.01,
    )

    for without_slippage, with_slippage in zip(
        results_without_slippage,
        results_with_slippage,
    ):
        assert (
            with_slippage["trading_cost"].sum()
            > without_slippage["trading_cost"].sum()
        )

        assert (
            with_slippage["strategy_return"].sum()
            < without_slippage["strategy_return"].sum()
        )


def test_walk_forward_rejects_negative_transaction_cost():
    df = create_sample_data(30)

    with pytest.raises(ValueError):
        run_walk_forward_strategy(
            df=df,
            strategy=always_long_strategy,
            train_size=10,
            test_size=5,
            transaction_cost=-0.01,
        )


def test_walk_forward_rejects_negative_slippage():
    df = create_sample_data(30)

    with pytest.raises(ValueError):
        run_walk_forward_strategy(
            df=df,
            strategy=always_long_strategy,
            train_size=10,
            test_size=5,
            slippage=-0.01,
        )
