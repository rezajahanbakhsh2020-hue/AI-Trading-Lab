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

    assert results[0]["walk_forward_train_start"].iloc[0] == 0
    assert results[0]["walk_forward_train_end"].iloc[0] == 10
    assert results[0]["walk_forward_test_start"].iloc[0] == 10
    assert results[0]["walk_forward_test_end"].iloc[0] == 15

    assert results[1]["walk_forward_train_start"].iloc[0] == 5
    assert results[1]["walk_forward_test_start"].iloc[0] == 15


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


def test_walk_forward_rejects_invalid_dataframe():
    with pytest.raises(
        TypeError,
        match="df must be a pandas DataFrame.",
    ):
        run_walk_forward_strategy(
            df="invalid",
            strategy=always_long_strategy,
            train_size=10,
            test_size=5,
        )


def test_walk_forward_rejects_invalid_strategy():
    df = create_sample_data(30)

    with pytest.raises(
        TypeError,
        match="strategy must be callable.",
    ):
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

    with pytest.raises(
        ValueError,
        match="strategy result must contain a 'signal' column.",
    ):
        run_walk_forward_strategy(
            df=df,
            strategy=invalid_strategy,
            train_size=10,
            test_size=5,
        )


def test_walk_forward_rejects_strategy_with_wrong_length():
    df = create_sample_data(30)

    def invalid_strategy(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        result = data.copy()
        result["signal"] = 1
        return result.iloc[:-1]

    with pytest.raises(
        ValueError,
        match="strategy must return the same number of rows",
    ):
        run_walk_forward_strategy(
            df=df,
            strategy=invalid_strategy,
            train_size=10,
            test_size=5,
        )


def test_walk_forward_rejects_negative_transaction_cost():
    df = create_sample_data(30)

    with pytest.raises(
        ValueError,
        match="transaction_cost must be non-negative.",
    ):
        run_walk_forward_strategy(
            df=df,
            strategy=always_long_strategy,
            train_size=10,
            test_size=5,
            transaction_cost=-0.001,
        )


def test_walk_forward_rejects_negative_slippage():
    df = create_sample_data(30)

    with pytest.raises(
        ValueError,
        match="slippage must be non-negative.",
    ):
        run_walk_forward_strategy(
            df=df,
            strategy=always_long_strategy,
            train_size=10,
            test_size=5,
            slippage=-0.001,
        )


def test_walk_forward_short_data_returns_empty():
    df = create_sample_data(14)

    results = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
    )

    assert results == []


def test_walk_forward_transaction_cost_reduces_performance():
    df = create_sample_data(30)

    without_cost = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        transaction_cost=0.0,
    )

    with_cost = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        transaction_cost=0.01,
    )

    no_cost_return = sum(
        result["strategy_return"].sum()
        for result in without_cost
    )

    cost_return = sum(
        result["strategy_return"].sum()
        for result in with_cost
    )

    assert cost_return < no_cost_return


def test_walk_forward_slippage_reduces_performance():
    df = create_sample_data(30)

    without_slippage = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        slippage=0.0,
    )

    with_slippage = run_walk_forward_strategy(
        df=df,
        strategy=always_long_strategy,
        train_size=10,
        test_size=5,
        slippage=0.01,
    )

    no_slippage_return = sum(
        result["strategy_return"].sum()
        for result in without_slippage
    )

    slippage_return = sum(
        result["strategy_return"].sum()
        for result in with_slippage
    )

    assert slippage_return < no_slippage_return
