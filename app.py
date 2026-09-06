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
    Add basic market metrics used by the dashboard.
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
    Run the project's configured Moving Average strategy.
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
    Run the project's configured Momentum strategy.
    """

    return momentum_signal(
        df,
        window=MOMENTUM_CONFIG["window"],
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

    strategy_function = STRATEGIES[strategy_name]

    result, report = run_strategy(
        df=df,
        strategy=strategy_function,
        transaction_cost=BACKTEST_CONFIG["transaction_cost"],
        slippage=BACKTEST_CONFIG["slippage"],
    )

    return result, report


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
            df["volatility_regime"] == "high_volatility"
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
) -> go.Figure:
    """
    Build the strategy drawdown curve.
    """

    figure = go.Figure()

    if "equity" not in df.columns:
        return figure

    equity = pd.to_numeric(
        df["equity"],
        errors="coerce",
    )

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1.0
    ) * 100.0

    figure.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        )
    )

    figure.update_layout(
        title="Strategy Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=320,
        hovermode="x unified",
    )

    return figure


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def format_percent(
    value: float | int | None,
) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_ratio(
    value: float | int | None,
) -> str:
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)

        if pd.isna(numeric_value):
            return "N/A"

        if numeric_value == float("inf"):
            return "∞"

        if numeric_value == float("-inf"):
            return "-∞"

        return f"{numeric_value:.2f}"

    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

market_data = load_market_data(str(DATA_PATH))
market_data = add_market_metrics(market_data)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("AI Trading Lab")

st.sidebar.subheader("Chart")

history = st.sidebar.slider(
    "Chart History",
    min_value=30,
    max_value=len(market_data),
    value=min(150, len(market_data)),
)

show_volume = st.sidebar.checkbox(
    "Show Volume",
    value=False,
)

show_signals = st.sidebar.checkbox(
    "Show Buy / Sell Signals",
    value=True,
)

show_regime = st.sidebar.checkbox(
    "Show Market Regime",
    value=False,
)

st.sidebar.subheader("Strategy")

strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    options=list(STRATEGIES.keys()),
)


# ---------------------------------------------------------------------
# Prepare strategy data
# ---------------------------------------------------------------------

selected_data = market_data.tail(history).copy()

backtest_result, report = run_selected_strategy(
    market_data,
    strategy_name,
)

display_result = backtest_result.tail(history).copy()

display_result = prepare_regime_data(
    display_result
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("📈 AI Trading Lab")

st.caption(
    "XAU/USD Strategy Research & Backtest Dashboard"
)

st.divider()


# ---------------------------------------------------------------------
# Strategy performance
# ---------------------------------------------------------------------

st.subheader(
    f"Strategy Performance — {strategy_name}"
)

metric_columns = st.columns(4)

with metric_columns[0]:
    st.metric(
        "Total Return",
        format_percent(
            report.get("total_return")
        ),
    )

with metric_columns[1]:
    st.metric(
        "Max Drawdown",
        format_percent(
            report.get("max_drawdown")
        ),
    )

with metric_columns[2]:
    st.metric(
        "Sharpe Ratio",
        format_ratio(
            report.get("sharpe_ratio")
        ),
    )

with metric_columns[3]:
    st.metric(
        "Calmar Ratio",
        format_ratio(
            report.get("calmar_ratio")
        ),
    )


secondary_metrics = st.columns(4)

with secondary_metrics[0]:
    st.metric(
        "Sortino Ratio",
        format_ratio(
            report.get("sortino_ratio")
        ),
    )

with secondary_metrics[1]:
    st.metric(
        "Win Rate",
        format_percent(
            report.get("win_rate")
        ),
    )

with secondary_metrics[2]:
    st.metric(
        "Exposure",
        format_percent(
            report.get("exposure")
        ),
    )

with secondary_metrics[3]:
    st.metric(
        "Profit Factor",
        format_ratio(
            report.get("profit_factor")
        ),
    )


# ---------------------------------------------------------------------
# Main market chart
# ---------------------------------------------------------------------

st.subheader("Market")

market_chart = build_price_chart(
    display_result,
    show_volume=show_volume,
    show_signals=show_signals,
    show_regime=show_regime,
)

st.plotly_chart(
    market_chart,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Equity and drawdown
# ---------------------------------------------------------------------

st.subheader("Performance Analysis")

equity_column, drawdown_column = st.columns(2)

with equity_column:
    equity_chart = build_equity_chart(
        display_result
    )

    st.plotly_chart(
        equity_chart,
        use_container_width=True,
    )

with drawdown_column:
    drawdown_chart = build_drawdown_chart(
        display_result
    )

    st.plotly_chart(
        drawdown_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------

st.subheader("Strategy Configuration")

if strategy_name == "Moving Average":
    config_data = {
        "Strategy": strategy_name,
        "Fast Window": MOVING_AVERAGE_CONFIG[
            "fast_window"
        ],
        "Slow Window": MOVING_AVERAGE_CONFIG[
            "slow_window"
        ],
        "Transaction Cost": BACKTEST_CONFIG[
            "transaction_cost"
        ],
        "Slippage": BACKTEST_CONFIG[
            "slippage"
        ],
    }

else:
    config_data = {
        "Strategy": strategy_name,
        "Momentum Window": MOMENTUM_CONFIG[
            "window"
        ],
        "Transaction Cost": BACKTEST_CONFIG[
            "transaction_cost"
        ],
        "Slippage": BACKTEST_CONFIG[
            "slippage"
        ],
    }

st.dataframe(
    pd.DataFrame(
        [config_data]
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Market statistics
# ---------------------------------------------------------------------

st.subheader("Market Statistics")

latest = market_data.iloc[-1]

market_columns = st.columns(4)

with market_columns[0]:
    st.metric(
        "Latest Close",
        f"{latest['close']:.2f}",
    )

with market_columns[1]:
    st.metric(
        "Daily Change",
        f"{latest['daily_change']:.2f}",
    )

with market_columns[2]:
    st.metric(
        "Daily Change %",
        f"{latest['daily_change_pct']:.2f}%",
    )

with market_columns[3]:
    st.metric(
        "Data Points",
        f"{len(market_data):,}",
    )


# ---------------------------------------------------------------------
# Dataset information
# ---------------------------------------------------------------------

st.subheader("Dataset")

dataset_columns = st.columns(3)

with dataset_columns[0]:
    st.write(
        "**Symbol:** XAU/USD"
    )

with dataset_columns[1]:
    st.write(
        "**Timeframe:** Daily"
    )

with dataset_columns[2]:
    st.write(
        f"**From:** "
        f"{market_data['timestamp'].min().date()} "
        f"**To:** "
        f"{market_data['timestamp'].max().date()}"
    )


# ---------------------------------------------------------------------
# Raw strategy result
# ---------------------------------------------------------------------

with st.expander(
    "View Backtest Data"
):
    st.dataframe(
        display_result,
        use_container_width=True,
        hide_index=True,
    )
