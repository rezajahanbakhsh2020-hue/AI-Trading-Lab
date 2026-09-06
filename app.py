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
from src.evaluation.compare import (
    compare_walk_forward_strategies,
)
from src.evaluation.final_report import (
    build_final_strategy_report,
    get_best_strategy,
)
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
    Load and validate XAU/USD market data.
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
    Add basic market metrics.
    """

    result = df.copy()

    result["return"] = result["close"].pct_change()

    result["daily_change"] = result["close"].diff()

    result["daily_change_pct"] = (
        result["close"].pct_change() * 100.0
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
    Run the configured Moving Average strategy.
    """

    return baseline_signal(
        df,
        fast_window=MOVING_AVERAGE_CONFIG["fast_window"],
        slow_window=MOVING_AVERAGE_CONFIG["slow_window"],
    )


def momentum_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the configured Momentum strategy.
    """

    return momentum_signal(
        df,
        window=MOMENTUM_CONFIG["window"],
    )


STRATEGIES = {
    "moving_average": moving_average_strategy,
    "momentum": momentum_strategy,
}


DISPLAY_STRATEGY_NAMES = {
    "moving_average": "Moving Average",
    "momentum": "Momentum",
}


# ---------------------------------------------------------------------
# Standard strategy execution
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

    if strategy_name not in DISPLAY_STRATEGY_NAMES.values():
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    internal_name = next(
        key
        for key, value in DISPLAY_STRATEGY_NAMES.items()
        if value == strategy_name
    )

    result, report = run_strategy(
        df=df,
        strategy=STRATEGIES[internal_name],
        transaction_cost=BACKTEST_CONFIG[
            "transaction_cost"
        ],
        slippage=BACKTEST_CONFIG[
            "slippage"
        ],
    )

    return result, report


# ---------------------------------------------------------------------
# Walk-forward execution
# ---------------------------------------------------------------------

@st.cache_data
def run_walk_forward_comparison(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int,
) -> dict[str, dict]:
    """
    Run the real project walk-forward comparison.

    Every strategy receives the same chronological
    train/test/step configuration.
    """

    return compare_walk_forward_strategies(
        df=df,
        strategies=STRATEGIES,
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


# ---------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------

def build_price_chart(
    df: pd.DataFrame,
    show_volume: bool,
    show_signals: bool,
    show_regime: bool,
) -> go.Figure:
    """
    Build the main interactive market chart.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="XAU/USD",
        )
    )

    if show_signals and "signal" in df.columns:
        markers = build_signal_markers(
            df,
            signal_column="signal",
        )

        buy_markers = markers[
            markers["marker"] == "buy"
        ]

        sell_markers = markers[
            markers["marker"] == "sell"
        ]

        if not buy_markers.empty:
            figure.add_trace(
                go.Scatter(
                    x=buy_markers["timestamp"],
                    y=buy_markers["price"],
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="triangle-up",
                        size=11,
                    ),
                )
            )

        if not sell_markers.empty:
            figure.add_trace(
                go.Scatter(
                    x=sell_markers["timestamp"],
                    y=sell_markers["price"],
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="triangle-down",
                        size=11,
                    ),
                )
            )

    if show_volume and "volume" in df.columns:
        figure.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.30,
            )
        )

        figure.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    if show_regime and "volatility_regime" in df.columns:
        high_regime = df[
            df["volatility_regime"]
            == "high_volatility"
        ]

        if not high_regime.empty:
            figure.add_trace(
                go.Scatter(
                    x=high_regime["timestamp"],
                    y=high_regime["high"],
                    mode="markers",
                    name="High Volatility",
                    marker=dict(
                        symbol="circle-open",
                        size=8,
                    ),
                )
            )

    figure.update_layout(
        title="XAU/USD Market Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=620,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return figure


def build_equity_chart(
    df: pd.DataFrame,
) -> go.Figure:
    """
    Build the strategy equity curve.
    """

    figure = go.Figure()

    if "equity" not in df.columns:
        return figure

    figure.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["equity"],
            mode="lines",
            name="Equity",
        )
    )

    figure.update_layout(
        title="Strategy Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=360,
        hovermode="x unified",
    )

    return figure


def build_drawdown_chart(
    df: pd.DataFrame,
) -> go.F
