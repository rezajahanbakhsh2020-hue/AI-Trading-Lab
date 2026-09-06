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

    Returns
    -------
    tuple[list[pd.DataFrame], dict]
        OOS results for each walk-forward window and an aggregate report.
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


def build_final_report(
    comparison: dict[str, dict],
    metric: str = "total_return",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Build the final ranked strategy report from a strategy comparison.

    This function connects the final strategy reporting layer
    to the main pipeline without changing the existing backtest
    or walk-forward workflows.

    Parameters
    ----------
    comparison:
        Output of compare_walk_forward_strategies().

    metric:
        Primary metric used for ranking.

    ascending:
        Ranking direction for the primary metric.

    Returns
    -------
    pd.DataFrame
        Final ranked strategy report.
    """
    return build_final_strategy_report(
        comparison=comparison,
        metric=metric,
        ascending=ascending,
    )
