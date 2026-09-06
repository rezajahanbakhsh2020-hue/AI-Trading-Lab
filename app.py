from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from configs.strategies import (
    BACKTEST_CONFIG,
    MOVING_AVERAGE_CONFIG,
    MOMENTUM_CONFIG,
)
from src.backtest.runner import run_strategy
from src.evaluation.market_regime import classify_volatility_regime
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal
from src.visualization.chart_data import prepare_chart_data
from src.visualization.chart_engine import build_signal_markers


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Trading Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data
def load_market_data(path: str) -> pd.DataFrame:
    """
    Load XAU/USD market data from the project data directory.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Market data file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = sorted(
        required_columns.difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    timestamp_series = data["timestamp"]

    if pd.api.types.is_numeric_dtype(timestamp_series):
        data["timestamp"] = pd.to_datetime(
            timestamp_series,
            unit="s",
            errors="coerce",
        )
    else:
        data["timestamp"] = pd.to_datetime(
            timestamp_series,
            errors="coerce",
        )

    if data["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains invalid datetime values."
        )

    return prepare_chart_data(data)


# ---------------------------------------------------------------------
# Market calculations
# ---------------------------------------------------------------------

def add_market_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add basic market metrics used by the dashboard.
    """

    result = df.copy()

    result["return"] = (
        result["close"]
        .pct_change()
    )

    result["daily_change"] = (
        result["close"]
        .diff()
    )

    result["daily_change_pct"] = (
        result["close"]
        .pct_change()
        * 100.0
    )

    return result


def prepare_regime_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add volatility regime information.
    """

    return classify_volatility_regime(
        df,
        return_column="return",
        window=20,
    )


# ---------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------

def moving_average_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the project's configured Moving Average strategy.
    """

    return baseline_signal(
        df,
        fast_window=MOVING_AVERAGE_CONFIG[
            "fast_window"
        ],
        slow_window=MOVING_AVERAGE_CONFIG[
            "slow_window"
        ],
    )


def momentum_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the project's configured Momentum strategy.
    """

    return momentum_signal(
        df,
        window=MOMENTUM_CONFIG[
            "window"
        ],
    )


STRATEGIES = {
    "Moving Average": moving_average_strategy,
    "Momentum": momentum_strategy,
}


# ---------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------

@st.cache_data
def run_selected_strategy(
    df: pd.DataFrame,
    strategy_name: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the selected strategy through the project's
    standard Strategy -> Backtest -> Evaluation pipeline.
    """

    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    strategy_function = STRATEGIES[
        strategy_name
    ]

    result, report = run_strategy(
        df=df,
        strategy=strategy_function,
        transaction_cost=BACKTEST_CONFIG[
            "transaction_cost"
        ],
        slippage=BACKTEST_CONFIG[
           
