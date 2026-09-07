from pathlib import Path

import pandas as pd

from src.evaluation.production_backtest import (
    build_production_strategy,
)
from src.evaluation.production_report import (
    build_production_report,
)


def test_build_production_strategy_moving_average():
    strategy = build_production_strategy(
        "moving_average"
    )

    assert callable(strategy)


def test_build_production_strategy_momentum():
    strategy = build_production_strategy(
        "momentum"
    )

    assert callable(strategy)


def test_unknown_production_strategy_raises():
    try:
        build_production_strategy("unknown")
    except ValueError as exc:
        assert "Unknown strategy" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_production_report_returns_metrics():
    backtest = pd.DataFrame(
        {
            "strategy_return": [
                0.01,
                -0.005,
                0.02,
                0.01,
            ]
        }
    )

    report = build_production_report(
        backtest
    )

    assert report["observations"] == 4
    assert "total_return" in report
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report


def test_production_report_handles_empty_data():
    backtest = pd.DataFrame(
        columns=["strategy_return"]
    )

    report = build_production_report(
        backtest
    )

    assert report["observations"] == 0
    assert report["total_return"] == 0.0
    assert report["max_drawdown"] == 0.0
    assert report["sharpe_ratio"] == 0.0
