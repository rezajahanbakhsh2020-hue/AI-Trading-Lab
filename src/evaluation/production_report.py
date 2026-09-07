from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import (
    max_drawdown,
    sharpe_ratio,
    total_return,
)


REQUIRED_COLUMNS = {
    "strategy_return",
}


def build_production_report(
    backtest: pd.DataFrame,
) -> dict:
    if not isinstance(backtest, pd.DataFrame):
        raise TypeError(
            "backtest must be a pandas DataFrame."
        )

    missing = REQUIRED_COLUMNS - set(
        backtest.columns
    )

    if missing:
        raise ValueError(
            "backtest is missing required columns: "
            f"{sorted(missing)}"
        )

    if backtest.empty:
        return {
            "observations": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    returns = backtest[
        "strategy_return"
    ].fillna(0.0)

    return {
        "observations": int(len(backtest)),
        "total_return": float(
            total_return(backtest)
        ),
        "max_drawdown": float(
            max_drawdown(backtest)
        ),
        "sharpe_ratio": float(
            sharpe_ratio(backtest)
        ),
    }
