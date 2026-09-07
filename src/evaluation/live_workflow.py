from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.strategies import (
    BACKTEST_CONFIG,
    MOVING_AVERAGE_CONFIG,
    MOMENTUM_CONFIG,
)
from src.data.loader import load_csv
from src.data.preprocessing import standardize_market_data
from src.data.validation import validate_market_data
from src.evaluation.compare import (
    compare_walk_forward_strategies,
)
from src.evaluation.final_report import (
    build_final_strategy_report,
    get_best_strategy,
)
from src.evaluation.result_store import (
    save_walk_forward_result,
)
from src.evaluation.strategy_selection import (
    select_eligible_strategies,
)
from src.features.indicators import add_returns
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal


DEFAULT_TRAIN_SIZE = 180
DEFAULT_TEST_SIZE = 60
DEFAULT_STEP = 30


def prepare_xauusd_data(
    path: str | Path,
) -> pd.DataFrame:
    df = load_csv(str(path))

    validate_market_data(df)

    df = standardize_market_data(df)
    df = add_returns(df)

    return df


def build_default_strategies() -> dict:
    return {
        "moving_average": lambda data: baseline_signal(
            data,
            fast_window=MOVING_AVERAGE_CONFIG[
                "fast_window"
            ],
            slow_window=MOVING_AVERAGE_CONFIG[
                "slow_window"
            ],
        ),
        "momentum": lambda data: momentum_signal(
            data,
            window=MOMENTUM_CONFIG["window"],
        ),
    }


def run_xauusd_walk_forward(
    path: str | Path,
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    step: int | None = DEFAULT_STEP,
    metric: str = "total_return",
    ascending: bool = False,
    min_total_return: float = 0.0,
    max_drawdown: float = 0.20,
    min_sharpe_ratio: float = 0.0,
    min_positive_window_rate: float = 0.50,
    save_result: bool = True,
) -> dict:
    df = prepare_xauusd_data(path)

    strategies = build_default_strategies()

    comparison = compare_walk_forward_strategies(
        df=df,
        strategies=strategies,
        train_size=train_size,
        test_size=test_size,
        step=step,
        transaction_cost=BACKTEST_CONFIG[
            "transaction_cost"
        ],
        slippage=BACKTEST_CONFIG[
            "slippage"
        ],
    )

    final_report = build_final_strategy_report(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )

    eligible = select_eligible_strategies(
        final_report,
        min_total_return=min_total_return,
        max_drawdown=max_drawdown,
        min_sharpe_ratio=min_sharpe_ratio,
        min_positive_window_rate=(
            min_positive_window_rate
        ),
    )

    best_strategy = get_best_strategy(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )

    result = {
        "data": df,
        "comparison": comparison,
        "final_report": final_report,
        "eligible_strategies": eligible,
        "best_strategy": best_strategy,
    }

    if save_result:
        result["saved_result"] = (
            save_walk_forward_result(result)
        )

    return result
