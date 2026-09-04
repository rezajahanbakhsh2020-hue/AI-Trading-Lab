import pandas as pd
import pytest

from src.evaluation.report import evaluate_backtest


def test_evaluate_backtest_returns_all_metrics():
    df = pd.DataFrame(
        {
            "signal": [0, 1, 1, 0, 1],
            "strategy_return": [0.0, 0.10, -0.05, 0.0, 0.08],
            "equity": [1.0, 1.10, 1.045, 1.045, 1.1286],
        }
    )

    report = evaluate_backtest(df)

    expected_keys = {
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
        "exposure",
        "win_rate",
        "profit_factor",
    }

    assert set(report.keys()) == expected_keys


def test_evaluate_backtest_returns_correct_basic_metrics():
    df = pd.DataFrame(
        {
            "signal": [0, 1, 1, 0, 1],
            "strategy_return": [0.0, 0.10, -0.05, 0.0, 0.08],
            "equity": [1.0, 1.10, 1.045, 1.045, 1.1286],
        }
    )

    report = evaluate_backtest(df)

    assert report["total_return"] == pytest.approx(0.1286)
    assert report["max_drawdown"] == pytest.approx(-0.05)
    assert report["exposure"] == pytest.approx(0.6)
    assert report["win_rate"] == pytest.approx(2 / 3)
    assert report["profit_factor"] == pytest.approx(0.18 / 0.05)


def test_evaluate_backtest_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "signal",
            "strategy_return",
            "equity",
        ]
    )

    report = evaluate_backtest(df)

    assert report["total_return"] == 0.0
    assert report["max_drawdown"] == 0.0
    assert report["sharpe_ratio"] == 0.0
    assert report["calmar_ratio"] == 0.0
    assert report["sortino_ratio"] == 0.0
    assert report["exposure"] == 0.0
    assert report["win_rate"] == 0.0
    assert report["profit_factor"] == 0.0


def test_evaluate_backtest_rejects_non_dataframe():
    with pytest.raises(TypeError):
        evaluate_backtest([1, 2, 3])


def test_evaluate_backtest_supports_custom_columns():
    df = pd.DataFrame(
        {
            "my_signal": [0, 1, 1, 0],
            "my_return": [0.0, 0.10, -0.05, 0.05],
            "my_equity": [1.0, 1.10, 1.045, 1.09725],
        }
    )

    report = evaluate_backtest(
        df,
        equity_column="my_equity",
        signal_column="my_signal",
        return_column="my_return",
    )

    assert report["total_return"] == pytest.approx(0.09725)
    assert report["exposure"] == pytest.approx(0.5)
    assert report["win_rate"] == pytest.approx(2 / 3)
    assert report["profit_factor"] == pytest.approx(0.15 / 0.05)


def test_evaluate_backtest_missing_equity_column():
    df = pd.DataFrame(
        {
            "signal": [0, 1],
            "strategy_return": [0.0, 0.1],
        }
    )

    with pytest.raises(ValueError):
        evaluate_backtest(df)


def test_evaluate_backtest_missing_signal_column():
    df = pd.DataFrame(
        {
            "equity": [1.0, 1.1],
            "strategy_return": [0.0, 0.1],
        }
    )

    with pytest.raises(ValueError):
        evaluate_backtest(df)


def test_evaluate_backtest_missing_return_column():
    df = pd.DataFrame(
        {
            "signal": [0, 1],
            "equity": [1.0, 1.1],
        }
    )

    with pytest.raises(ValueError):
        evaluate_backtest(df)
