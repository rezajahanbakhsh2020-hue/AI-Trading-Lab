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

    result = backtest.copy()

    if "equity" not in result.columns:
        result["equity"] = (
            1.0
            + result["strategy_return"]
            .fillna(0.0)
        ).cumprod()

    return {
        "observations": int(len(result)),
        "total_return": float(
            total_return(result)
        ),
        "max_drawdown": float(
            max_drawdown(result)
        ),
        "sharpe_ratio": float(
            sharpe_ratio(result)
        ),
    }
