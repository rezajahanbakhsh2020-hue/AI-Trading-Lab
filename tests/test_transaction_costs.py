import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.runner import run_strategy
from src.evaluation.compare import compare_strategies


def always_long_strategy(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = 1

    return result


def alternating_strategy(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = [
        0,
        1,
        0,
        1,
        0,
    ]

    return result


def test_default_costs_preserve_previous_behavior():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(df)

    expected_strategy_return = [
        float("nan"),
        -0.05,
        0.0,
        -0.10,
    ]

    assert pd.isna(result["strategy_return"].iloc[0])

    assert result["strategy_return"].iloc[1:].tolist() == pytest.approx(
        expected_strategy_return[1:]
    )


def test_position_is_previous_period_signal():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(df)

    assert result["position"].tolist() == pytest.approx([
        0.0,
        1.0,
        0.0,
        1.0,
    ])


def test_turnover_is_based_on_position_changes():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(df)

    assert result["turnover"].tolist() == pytest.approx([
        0.0,
        1.0,
        1.0,
        1.0,
    ])


def test_transaction_cost_reduces_strategy_return():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(
        df,
        transaction_cost=0.01,
    )

    assert result["trading_cost"].tolist() == pytest.approx([
        0.0,
        0.01,
        0.01,
        0.01,
    ])

    assert pd.isna(result["strategy_return"].iloc[0])

    assert result["strategy_return"].iloc[1:].tolist() == pytest.approx([
        -0.06,
        -0.01,
        -0.11,
    ])


def test_slippage_reduces_strategy_return():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(
        df,
        slippage=0.005,
    )

    assert result["trading_cost"].tolist() == pytest.approx([
        0.0,
        0.005,
        0.005,
        0.005,
    ])

    assert pd.isna(result["strategy_return"].iloc[0])

    assert result["strategy_return"].iloc[1:].tolist() == pytest.approx([
        -0.055,
        -0.005,
        -0.105,
    ])


def test_transaction_cost_and_slippage_are_combined():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(
        df,
        transaction_cost=0.01,
        slippage=0.005,
    )

    assert result["trading_cost"].tolist() == pytest.approx([
        0.0,
        0.015,
        0.015,
        0.015,
    ])

    assert pd.isna(result["strategy_return"].iloc[0])

    assert result["strategy_return"].iloc[1:].tolist() == pytest.approx([
        -0.065,
        -0.015,
        -0.115,
    ])


def test_gross_return_is_separate_from_net_return():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20],
        "signal": [1, 1, 1],
    })

    result = run_backtest(
        df,
        transaction_cost=0.01,
    )

    assert "gross_strategy_return" in result.columns
    assert "strategy_return" in result.columns
    assert "trading_cost" in result.columns

    assert pd.isna(result["gross_strategy_return"].iloc[0])

    assert result["gross_strategy_return"].iloc[1:].tolist() == pytest.approx([
        -0.05,
        0.20,
    ])

    assert pd.isna(result["strategy_return"].iloc[0])

    assert result["strategy_return"].iloc[1:].tolist() == pytest.approx([
        -0.06,
        0.20,
    ])


def test_no_cost_is_charged_when_position_remains_unchanged():
    df = pd.DataFrame({
        "return": [0.10, 0.02, 0.03, 0.04],
        "signal": [0, 0, 0, 0],
    })

    result = run_backtest(
        df,
        transaction_cost=0.01,
        slippage=0.005,
    )

    assert result["turnover"].tolist() == pytest.approx([
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    assert result["trading_cost"].tolist() == pytest.approx([
        0.0,
        0.0,
        0.0,
        0.0,
    ])


def test_costs_are_applied_to_each_position_change():
    df = pd.DataFrame({
        "return": [0.01, 0.01, 0.01, 0.01, 0.01],
        "signal": [0, 1, 0, 1, 0],
    })

    result = run_backtest(
        df,
        transaction_cost=0.01,
    )

    assert result["turnover"].tolist() == pytest.approx([
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ])

    assert result["trading_cost"].tolist() == pytest.approx([
        0.0,
        0.0,
        0.01,
        0.01,
        0.01,
    ])


def test_negative_transaction_cost_is_rejected():
    df = pd.DataFrame({
        "return": [0.01, 0.02],
        "signal": [1, 1],
    })

    with pytest.raises(ValueError):
        run_backtest(
            df,
            transaction_cost=-0.01,
        )


def test_negative_slippage_is_rejected():
    df = pd.DataFrame({
       
