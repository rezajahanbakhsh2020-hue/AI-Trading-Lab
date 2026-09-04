import pandas as pd

from src.data.loader import load_csv
from src.data.validation import validate_market_data
from src.data.preprocessing import standardize_market_data
from src.features.indicators import add_returns
from src.strategy.baseline import generate_baseline_signal
from src.backtest.engine import run_backtest
from src.evaluation.report import evaluate_backtest


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
