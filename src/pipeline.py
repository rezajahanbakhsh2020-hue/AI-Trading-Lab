import pandas as pd

from configs.strategies import BACKTEST_CONFIG
from src.data.loader import load_csv
from src.data.validation import validate_market_data
from src.data.preprocessing import standardize_market_data
from src.features.indicators import add_returns
from src.evaluation.report import evaluate_backtest
from src.strategies.baseline import baseline_signal


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
    Run the baseline strategy through the standard backtest pipeline.

    Pipeline:
        CSV
        -> Loader
        -> Validation
        -> Preprocessing
        -> Returns
        -> Strategy
        -> Backtest
        -> Evaluation
    """

    from src.backtest.runner import run_strategy

    df = load_and_prepare_market_data(path)

    df = add_returns(df)

    strategy = lambda data: baseline_signal(data)

    result, report = run_strategy(
        df=df,
        strategy=strategy,
        transaction_cost=BACKTEST_CONFIG["transaction_cost"],
        slippage=BACKTEST_CONFIG["slippage"],
    )

    return result, report
