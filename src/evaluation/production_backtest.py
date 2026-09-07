from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.strategies import (
    BACKTEST_CONFIG,
    MOVING_AVERAGE_CONFIG,
    MOMENTUM_CONFIG,
)
from src.backtest.engine import run_backtest
from src.data.loader import load_csv
from src.data.preprocessing import standardize_market_data
from src.data.validation import validate_market_data
from src.evaluation.production_result_store import (
    save_production_result,
)
from src.evaluation.production_selection import (
    select_production_strategy,
)
from src.features.indicators import add_returns
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def build_production_strategy(
    strategy_name: str,
):
    if strategy_name == "moving_average":
        return lambda data: baseline_signal(
            data,
            fast_window=MOVING_AVERAGE_CONFIG[
                "fast_window"
            ],
            slow_window=MOVING_AVERAGE_CONFIG[
                "slow_window"
            ],
        )

    if strategy_name == "momentum":
        return lambda data: momentum_signal(
            data,
            window=MOMENTUM_CONFIG["window"],
        )

    raise ValueError(
        f"Unknown strategy: {strategy_name}"
    )


def run_production_backtest(
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    save_result: bool = True,
) -> dict:
    selection = select_production_strategy(
        results_dir=results_dir
    )

    strategy_name = selection["strategy"]

    if strategy_name is None:
        raise ValueError(
            "No stable production strategy "
            "is available."
        )

    df = load_csv(str(data_path))
    validate_market_data(df)
    df = standardize_market_data(df)
    df = add_returns(df)

    strategy = build_production_strategy(
        strategy_name
    )

    strategy_result = strategy(df)

    backtest_result = run_backtest(
        strategy_result,
        transaction_cost=BACKTEST_CONFIG[
            "transaction_cost"
        ],
        slippage=BACKTEST_CONFIG[
            "slippage"
        ],
    )

    result = {
        "strategy": strategy_name,
        "stability_score": selection[
            "stability_score"
        ],
        "stability_report": selection[
            "stability_report"
        ],
        "backtest": backtest_result,
    }

    if save_result:
        result["saved_result"] = (
            save_production_result(result)
        )

    return result
