import pandas as pd
import pytest

from src.evaluation.report import evaluate_backtest


def test_evaluate_backtest():
    df = pd.DataFrame({
        "signal": [1, 1, 0, 1, 1],
        "strategy_return": [0.02, -0.01, 0.0, 0.03, -0.02],
        "equity": [1.0, 1.02, 1.0098, 1.0098, 1.040094],
    })

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

    assert report["total_return"] > 0
    assert report["max_drawdown"] <= 0
    assert report["exposure"] == 0.8
    assert report["win_rate"] == pytest.approx(0.5)
    assert report["profit_factor"] == pytest.approx(1.6666666667)


def test_evaluate_backtest_custom_columns():
    df = pd.DataFrame({
        "position": [1, 0, 1, 1],
        "pnl": [0.01, 0.0, -0.005, 0.02],
        "balance": [1.0, 1.01, 1.01, 1.00495],
    })

    report = evaluate_backtest(
        df,
        equity_column="balance",
        signal_column="position",
        return_column="pnl",
    )

    assert report["exposure"] == 0.75
    assert report["win_rate"] == pytest.approx(2 / 3)
    assert report["profit_factor"] == pytest.approx(6.0)


def test_evaluate_backtest_requires_dataframe():
    with pytest.raises(TypeError):
        evaluate_backtest(None)


def test_evaluate_backtest_missing_required_column():
    df = pd.DataFrame({
        "equity": [1.0, 1.1],
        "signal": [1, 1],
    })

    with pytest.raises(ValueError):
        evaluate_backtest(df)
