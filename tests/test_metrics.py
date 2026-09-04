import pandas as pd

from src.evaluation.metrics import (
    total_return,
    max_drawdown,
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    exposure,
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


def test_total_return_empty_dataframe():
    df = pd.DataFrame({
        "equity": [],
    })

    assert total_return(df) == 0.0


def test_exposure_empty_dataframe():
    df = pd.DataFrame({
        "signal": [],
    })

    assert exposure(df) == 0.0
