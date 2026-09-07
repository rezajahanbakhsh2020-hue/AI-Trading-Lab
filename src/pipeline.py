import pandas as pd

from src.data.loader import load_csv
from src.data.validation import validate_market_data
from src.data.preprocessing import standardize_market_data

from src.features.indicators import add_returns

from src.strategy.baseline import generate_baseline_signal

from src.backtest.engine import run_backtest
from src.evaluation.report import evaluate_backtest
from src.evaluation.walk_forward_report import evaluate_walk_forward
from src.evaluation.walk_forward_runner import run_walk_forward_strategy
from src.evaluation.final_report import build_final_strategy_report
from src.evaluation.strategy_suite import run_default_strategy_suite
from src.evaluation.compare import compare_walk_forward_strategies
from configs.strategies import (
    BACKTEST_CONFIG,
    MOVING_AVERAGE_CONFIG,
    MOMENTUM_CONFIG,
)
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal


def load_and_prepare_market_data(
    path: str,
) -> pd.DataFrame:
    """
    Load, validate, and standardize market data.

    Pipeline:
    CSV -> Loader -> Validation -> Preprocessing
    """
    df = load_csv(path)
    validate_market_data(df)
    df = standardize_market_data(df)

    return df


def run_strategy_backtest(
    path: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete baseline strategy pipeline.

    Pipeline:
    CSV
    -> Loader
    -> Validation
    -> Preprocessing
    -> Returns
    -> Strategy Signal
    -> Backtest
    -> Evaluation
    """
    df = load_and_prepare_market_data(path)
    df = add_returns(df)
    df = generate_baseline_signal(df)

    df = run_backtest(df)

    report = evaluate_backtest(df)

    return df, report


def run_default_strategy_suite_pipeline(
    path: str,
) -> dict[str, dict]:
    """
    Run the default configured strategy suite through the main pipeline.

    Pipeline:
    CSV
    -> Loader
    -> Validation
    -> Preprocessing
    -> Returns
    -> Default Strategy Suite
    -> Strategy Comparison

    The default suite currently contains:
    - moving_average
    - momentum
    """
    df = load_and_prepare_market_data(path)
    df = add_returns(df)

    return run_default_strategy_suite(df)


def run_walk_forward_backtest(
    path: str,
    train_size: int,
    test_size: int,
    step: int | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> tuple[list[pd.DataFrame], dict]:
    """
    Run the baseline strategy through chronological walk-forward
    out-of-sample evaluation.

    Pipeline:
    CSV
    -> Loader
    -> Validation
    -> Preprocessing
    -> Returns
    -> Walk-Forward OOS Backtest
    -> Walk-Forward Evaluation
    """
    df = load_and_prepare_market_data(path)
    df = add_returns(df)

    def baseline_strategy(
        window_df: pd.DataFrame,
    ) -> pd.DataFrame:
        return generate_baseline_signal(window_df)

    oos_results = run_walk_forward_strategy(
        df=df,
        strategy=baseline_strategy,
        train_size=train_size,
        test_size=test_size,
        step=step,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )

    report = evaluate_walk_forward(oos_results)

    return oos_results, report


def run_default_walk_forward_pipeline(
    path: str,
    train_size: int,
    test_size: int,
    step: int | None = None,
    metric: str = "total_return",
    ascending: bool = False,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Run the complete default strategy suite through walk-forward
    out-of-sample evaluation and final ranking.

    Pipeline:
    CSV
    -> Loader
    -> Validation
    -> Preprocessing
    -> Returns
    -> Default Strategies
    -> Walk-Forward OOS
    -> Strategy Comparison
    -> Final Ranking

    Returns
    -------
    tuple[dict[str, dict], pd.DataFrame]
        Walk-forward comparison results and final ranked report.
    """
    df = load_and_prepare_market_data(path)
    df = add_returns(df)

    strategies = {
        "moving_average": lambda data: baseline_signal(
            data,
            fast_window=MOVING_AVERAGE_CONFIG["fast_window"],
            slow_window=MOVING_AVERAGE_CONFIG["slow_window"],
        ),
        "momentum": lambda data: momentum_signal(
            data,
            window=MOMENTUM_CONFIG["window"],
        ),
    }

    comparison = compare_walk_forward_strategies(
        df=df,
        strategies=strategies,
        train_size=train_size,
        test_size=test_size,
        step=step,
        transaction_cost=BACKTEST_CONFIG["transaction_cost"],
        slippage=BACKTEST_CONFIG["slippage"],
    )

    final_report = build_final_strategy_report(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )

    return comparison, final_report


def build_final_report(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Build the final ranked strategy report from a strategy comparison.
    """
    return build_final_strategy_report(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )
