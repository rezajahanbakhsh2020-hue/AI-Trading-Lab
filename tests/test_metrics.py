import pandas as pd
import pytest

from src.evaluation.metrics import (
    total_return,
    max_drawdown,
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    exposure,
    win_rate,
    profit_factor,
)


def test_total_return():
    df = pd.DataFrame({
        "equity": [1.0, 1.1, 1.2],
    })

    assert round(total_return(df), 10) == 0.2


def test_max_drawdown():
    df = pd.DataFrame({
        "equity": [1.0, 1.2, 1.1, 1.3],
    })

    assert round(max_drawdown(df), 10) == round(-0.0833333333, 10)


def test_sharpe_ratio():
    df = pd.DataFrame({
        "equity": [1.0, 1.1, 1.21],
    })

    result = sharpe_ratio(df)

    assert result > 0


def test_calmar_ratio():
    df = pd.DataFrame({
        "equity": [1.0, 1.2, 1.1, 1.3],
    })

    result = calmar_ratio(df)

    assert result > 0


def test_sortino_ratio():
    df = pd.DataFrame({
        "equity": [1.0, 1.1, 1.05, 1.2, 1.15],
    })

    result = sortino_ratio(df)

    assert result > 0


def test_exposure():
    df = pd.DataFrame({
        "signal": [1, 1, 0, 0, 1],
    })

    assert exposure(df) == 0.6


def test_win_rate():
    df = pd.DataFrame({
        "strategy_return": [0.02, -0.01, 0.03, 0.0, -0.02],
    })

    assert win_rate(df) == 0.5


def test_profit_factor():
    df = pd.DataFrame({
        "strategy_return": [0.02, -0.01, 0.03, 0.0, -0.02],
    })

    assert profit_factor(df) == pytest.approx(1.6666666667)


def test_total_return_empty_dataframe():
    df = pd.DataFrame({
        "equity": [],
    })

    assert total_return(df) == 0.0


def test_max_drawdown_empty_dataframe():
    df = pd.DataFrame({
        "equity": [],
    })

    assert max_drawdown(df) == 0.0


def test_sharpe_ratio_empty_dataframe():
    df = pd.DataFrame({
        "equity": [],
    })

    assert sharpe_ratio(df) == 0.0


def test_calmar_ratio_empty_dataframe():
    df = pd.DataFrame({
        "equity": [],
    })

    assert calmar_ratio(df) == 0.0


def test_sortino_ratio_empty_dataframe():
    df = pd.DataFrame({
        "equity": [],
    })

    assert sortino_ratio(df) == 0.0


def test_exposure_empty_dataframe():
    df = pd.DataFrame({
        "signal": [],
    })

    assert exposure(df) == 0.0


def test_win_rate_no_trades():
    df = pd.DataFrame({
        "strategy_return": [0.0, 0.0, 0.0],
    })

    assert win_rate(df) == 0.0


def test_profit_factor_no_losses():
    df = pd.DataFrame({
        "strategy_return": [0.01, 0.02, 0.0],
    })

    assert profit_factor(df) == 0.0


def test_profit_factor_no_wins():
    df = pd.DataFrame({
        "strategy_return": [-0.01, -0.02, 0.0],
    })

    assert profit_factor(df) == 0.0


def test_sharpe_ratio_constant_equity():
    df = pd.DataFrame({
        "equity": [1.0, 1.0, 1.0, 1.0],
    })

    assert sharpe_ratio(df) == 0.0


def test_sortino_ratio_without_downside():
    df = pd.DataFrame({
        "equity": [1.0, 1.1, 1.2, 1.3],
    })

    assert sortino_ratio(df) == 0.0


def test_calmar_ratio_without_drawdown():
    df = pd.DataFrame({
        "equity": [1.0, 1.1, 1.2, 1.3],
    })

    assert calmar_ratio(df) == 0.0


def test_missing_equity_column():
    df = pd.DataFrame({
        "price": [1.0, 2.0],
    })

    with pytest.raises(ValueError):
        total_return(df)


def test_missing_equity_column_for_max_drawdown():
    df = pd.DataFrame({
        "price": [1.0, 2.0],
    })

    with pytest.raises(ValueError):
        max_drawdown(df)


def test_missing_equity_column_for_sharpe_ratio():
    df = pd.DataFrame({
        "price": [1.0, 2.0],
    })

    with pytest.raises(ValueError):
        sharpe_ratio(df)


def test_missing_equity_column_for_calmar_ratio():
    df = pd.DataFrame({
        "price": [1.0, 2.0],
    })

    with pytest.raises(ValueError):
        calmar_ratio(df)


def test_missing_equity_column_for_sortino_ratio():
    df = pd.DataFrame({
        "price": [1.0, 2.0],
    })

    with pytest.raises(ValueError):
        sortino_ratio(df)


def test_missing_signal_column():
    df = pd.DataFrame({
        "strategy_return": [0.01, -0.01],
    })

    with pytest.raises(ValueError):
        exposure(df)


def test_missing_strategy_return_column_for_win_rate():
    df = pd.DataFrame({
        "equity": [1.0, 1.1],
    })

    with pytest.raises(ValueError):
        win_rate(df)


def test_missing_strategy_return_column_for_profit_factor():
    df = pd.DataFrame({
        "equity": [1.0, 1.1],
    })

    with pytest.raises(ValueError):
        profit_factor(df)
