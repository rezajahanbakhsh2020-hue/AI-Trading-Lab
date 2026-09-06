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
    comparison_dataframe,
)
from src.evaluation.final_report import (
    build_final_strategy_report,
    get_best_strategy,
)
from src.evaluation.market_regime import (
    classify_volatility_regime,
)
from src.evaluation.strategy_selection import (
    select_eligible_strategies,
)
from src.evaluation.strategy_suite import (
    run_default_strategy_suite,
)
from src.strategies.baseline import baseline_signal
from src.strategies.momentum import momentum_signal
from src.visualization.chart_data import (
    prepare_chart_data,
)
from src.visualization.chart_engine import (
    build_signal_markers,
)


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
    Run the project's walk-forward strategy comparison.
    """

    from src.evaluation.compare import (
        compare_walk_forward_strategies,
    )

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
        numeric_value = float(value)

        if pd.isna(numeric_value):
            return "N/A"

        return f"{numeric_value * 100:.2f}%"

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


def format_number(
    value: float | int | None,
) -> str:
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)

        if pd.isna(numeric_value):
            return "N/A"

        return f"{numeric_value:,.2f}"

    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------
# Safe report helpers
# ---------------------------------------------------------------------

def report_value(
    report: dict,
    key: str,
    default: object = None,
) -> object:
    """
    Safely retrieve a value from a strategy report.
    """

    if not isinstance(report, dict):
        return default

    return report.get(key, default)


def report_metric_frame(
    reports: dict,
) -> pd.DataFrame:
    """
    Convert strategy reports to a compact DataFrame.
    """

    rows = []

    for strategy_name, report in reports.items():
        if not isinstance(report, dict):
            continue

        rows.append(
            {
                "strategy": DISPLAY_STRATEGY_NAMES.get(
                    strategy_name,
                    strategy_name,
                ),
                "total_return": report.get(
                    "total_return"
                ),
                "max_drawdown": report.get(
                    "max_drawdown"
                ),
                "sharpe_ratio": report.get(
                    "sharpe_ratio"
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def normalize_walk_forward_results(
    results: dict,
) -> pd.DataFrame:
    """
    Convert walk-forward results into a display DataFrame.

    The function accepts the project's dictionary result while
    remaining defensive about the exact report structure.
    """

    rows = []

    if not isinstance(results, dict):
        return pd.DataFrame()

    for strategy_name, value in results.items():
        display_name = DISPLAY_STRATEGY_NAMES.get(
            strategy_name,
            strategy_name,
        )

        if isinstance(value, dict):
            row = {
                "strategy": display_name,
                "total_return": value.get(
                    "total_return"
                ),
                "max_drawdown": value.get(
                    "max_drawdown"
                ),
                "sharpe_ratio": value.get(
                    "sharpe_ratio"
                ),
                "positive_window_rate": value.get(
                    "positive_window_rate"
                ),
                "window_count": value.get(
                    "window_count"
                ),
            }

            rows.append(row)

        elif isinstance(value, pd.DataFrame):
            frame = value.copy()

            if "strategy" not in frame.columns:
                frame.insert(
                    0,
                    "strategy",
                    display_name,
                )

            rows.append(frame)

    if not rows:
        return pd.DataFrame()

    if all(
        isinstance(item, dict)
        for item in rows
    ):
        return pd.DataFrame(rows)

    frames = [
        item
        for item in rows
        if isinstance(item, pd.DataFrame)
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

st.title("📈 AI Trading Lab")
st.caption(
    "XAU/USD strategy research, backtesting, "
    "walk-forward evaluation and strategy selection"
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("Controls")

selected_strategy = st.sidebar.selectbox(
    "Strategy",
    options=list(DISPLAY_STRATEGY_NAMES.values()),
    index=0,
)

st.sidebar.subheader("Chart")

show_volume = st.sidebar.checkbox(
    "Show Volume",
    value=True,
)

show_signals = st.sidebar.checkbox(
    "Show Buy/Sell Signals",
    value=True,
)

show_regime = st.sidebar.checkbox(
    "Show High Volatility",
    value=False,
)

st.sidebar.subheader("Strategy Selection Gate")

min_total_return = st.sidebar.number_input(
    "Minimum Total Return",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.01,
    format="%.2f",
)

max_drawdown = st.sidebar.number_input(
    "Maximum Drawdown",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01,
    format="%.2f",
)

min_sharpe_ratio = st.sidebar.number_input(
    "Minimum Sharpe Ratio",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.10,
    format="%.2f",
)

min_positive_window_rate = st.sidebar.number_input(
    "Minimum Positive Window Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    format="%.2f",
)

st.sidebar.subheader("Walk-Forward")

default_train_size = min(
    180,
    max(20, len(pd.read_csv(DATA_PATH)) // 2)
    if DATA_PATH.exists()
    else 180,
)

default_test_size = min(
    60,
    max(10, default_train_size // 3),
)

default_step = min(
    30,
    max(5, default_test_size // 2),
)

train_size = st.sidebar.number_input(
    "Train Size",
    min_value=20,
    max_value=1000,
    value=int(default_train_size),
    step=10,
)

test_size = st.sidebar.number_input(
    "Test Size",
    min_value=5,
    max_value=500,
    value=int(default_test_size),
    step=5,
)

step = st.sidebar.number_input(
    "Step",
    min_value=1,
    max_value=500,
    value=int(default_step),
    step=5,
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

try:
    market_data = load_market_data(
        str(DATA_PATH)
    )
except Exception as exc:
    st.error(
        f"Unable to load market data: {exc}"
    )
    st.stop()


market_data = add_market_metrics(
    market_data
)

market_data = prepare_regime_data(
    market_data
)


# ---------------------------------------------------------------------
# Market overview
# ---------------------------------------------------------------------

st.subheader("Market Overview")

latest_close = market_data["close"].iloc[-1]
first_close = market_data["close"].iloc[0]

total_market_return = (
    latest_close / first_close - 1.0
)

latest_daily_change = market_data[
    "daily_change"
].iloc[-1]

latest_daily_change_pct = market_data[
    "daily_change_pct"
].iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Close",
    format_number(latest_close),
)

col2.metric(
    "Market Return",
    format_percent(total_market_return),
)

col3.metric(
    "Daily Change",
    format_number(latest_daily_change),
)

col4.metric(
    "Daily Change %",
    (
        format_number(latest_daily_change_pct)
        + "%"
        if pd.notna(latest_daily_change_pct)
        else "N/A"
    ),
)


# ---------------------------------------------------------------------
# Main market chart
# ---------------------------------------------------------------------

st.subheader("Market Chart")

st.plotly_chart(
    build_price_chart(
        market_data,
        show_volume=show_volume,
        show_signals=False,
        show_regime=show_regime,
    ),
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Selected strategy backtest
# ---------------------------------------------------------------------

st.subheader(
    f"{selected_strategy} Strategy Backtest"
)

try:
    selected_result, selected_report = (
        run_selected_strategy(
            market_data,
            selected_strategy,
        )
    )

    chart_data = selected_result.copy()

    if "volatility_regime" not in chart_data.columns:
        chart_data = prepare_regime_data(
            chart_data
        )

    st.plotly_chart(
        build_price_chart(
            chart_data,
            show_volume=show_volume,
            show_signals=show_signals,
            show_regime=show_regime,
        ),
        use_container_width=True,
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Total Return",
        format_percent(
            report_value(
                selected_report,
                "total_return",
            )
        ),
    )

    metric2.metric(
        "Max Drawdown",
        format_percent(
            report_value(
                selected_report,
                "max_drawdown",
            )
        ),
    )

    metric3.metric(
        "Sharpe Ratio",
        format_ratio(
            report_value(
                selected_report,
                "sharpe_ratio",
            )
        ),
    )

    metric4.metric(
        "Trades",
        format_number(
            report_value(
                selected_report,
                "trades",
                report_value(
                    selected_report,
                    "number_of_trades",
                ),
            )
        ),
    )

    st.plotly_chart(
        build_equity_chart(
            chart_data
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        build_drawdown_chart(
            chart_data
        ),
        use_container_width=True,
    )

except Exception as exc:
    st.error(
        f"Strategy backtest failed: {exc}"
    )


# ---------------------------------------------------------------------
# Strategy suite comparison
# ---------------------------------------------------------------------

st.subheader("Strategy Suite Comparison")

try:
    suite_result = run_default_strategy_suite(
        market_data
    )

    if isinstance(suite_result, tuple):
        suite_reports = suite_result[1]
    else:
        suite_reports = suite_result

    if isinstance(suite_reports, dict):
        suite_frame = report_metric_frame(
            suite_reports
        )

        if not suite_frame.empty:
            st.dataframe(
                suite_frame,
                use_container_width=True,
                hide_index=True,
            )

except Exception as exc:
    st.warning(
        f"Strategy suite comparison unavailable: {exc}"
    )


# ---------------------------------------------------------------------
# Final strategy report
# ---------------------------------------------------------------------

st.subheader("Final Strategy Report")

try:
    final_report = build_final_strategy_report(
        market_data
    )

    if isinstance(final_report, pd.DataFrame):
        st.dataframe(
            final_report,
            use_container_width=True,
            hide_index=True,
        )
    elif isinstance(final_report, dict):
        final_frame = pd.DataFrame(
            final_report
        )

        st.dataframe(
            final_frame,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write(final_report)

except Exception as exc:
    st.warning(
        f"Final strategy report unavailable: {exc}"
    )


# ---------------------------------------------------------------------
# Strategy selection gate
# ---------------------------------------------------------------------

st.subheader("Strategy Selection Gate")

try:
    gate_source = None

    if isinstance(
        final_report,
        pd.DataFrame,
    ):
        gate_source = final_report

    elif isinstance(
        final_report,
        dict,
    ):
        gate_source = pd.DataFrame(
            final_report
        )

    if gate_source is None:
        st.info(
            "No final strategy report is available "
            "for eligibility filtering."
        )
    else:
        eligible = select_eligible_strategies(
            gate_source,
            min_total_return=min_total_return,
            max_drawdown=max_drawdown,
            min_sharpe_ratio=min_sharpe_ratio,
            min_positive_window_rate=(
                min_positive_window_rate
            ),
        )

        st.caption(
            "Strategies passing all configured "
            "performance gates."
        )

        if eligible.empty:
            st.warning(
                "No strategy currently passes "
                "all configured gates."
            )
        else:
            st.success(
                f"{len(eligible)} strategy(s) "
                "passed the selection gate."
            )

            st.dataframe(
                eligible,
                use_container_width=True,
                hide_index=True,
            )

except Exception as exc:
    st.warning(
        f"Strategy selection gate unavailable: {exc}"
    )


# ---------------------------------------------------------------------
# Best strategy
# ---------------------------------------------------------------------

st.subheader("Best Strategy")

try:
    if isinstance(
        final_report,
        pd.DataFrame,
    ):
        best_strategy = get_best_strategy(
            final_report
        )
    else:
        best_strategy = get_best_strategy(
            final_report
        )

    st.info(
        f"Best strategy: {best_strategy}"
    )

except Exception as exc:
    st.warning(
        f"Best strategy unavailable: {exc}"
    )


# ---------------------------------------------------------------------
# Walk-forward comparison
# ---------------------------------------------------------------------

st.subheader(
    "Walk-Forward Out-of-Sample Comparison"
)

st.caption(
    "The walk-forward evaluation uses sequential "
    "train/test windows and evaluates strategies "
    "out-of-sample."
)

if (
    train_size + test_size > len(market_data)
):
    st.warning(
        "Train Size + Test Size exceeds the "
        "available dataset length."
    )
else:
    run_walk_forward = st.button(
        "Run Walk-Forward Comparison",
        type="primary",
    )

    if run_walk_forward:
        try:
            with st.spinner(
                "Running walk-forward evaluation..."
            ):
                walk_forward_results = (
                    run_walk_forward_comparison(
                        market_data,
                        train_size=int(
                            train_size
                        ),
                        test_size=int(
                            test_size
                        ),
                        step=int(step),
                    )
                )

            st.success(
                "Walk-forward evaluation completed."
            )

            walk_forward_frame = (
                normalize_walk_forward_results(
                    walk_forward_results
                )
            )

            if not walk_forward_frame.empty:
                st.dataframe(
                    walk_forward_frame,
                    use_container_width=True,
                    hide_index=True,
                )

                numeric_columns = [
                    column
                    for column in [
                        "total_return",
                        "max_drawdown",
                        "sharpe_ratio",
                        "positive_window_rate",
                    ]
                    if column
                    in walk_forward_frame.columns
                ]

                if numeric_columns:
                    st.subheader(
                        "Walk-Forward Metrics"
                    )

                    display_frame = (
                        walk_forward_frame[
                            [
                                "strategy"
                            ]
                            + numeric_columns
                        ]
                        .copy()
                    )

                    if (
                        "total_return"
                        in display_frame.columns
                    ):
                        display_frame[
                            "total_return"
                        ] = display_frame[
                            "total_return"
                        ].map(
                            format_percent
                        )

                    if (
                        "max_drawdown"
                        in display_frame.columns
                    ):
                        display_frame[
                            "max_drawdown"
                        ] = display_frame[
                            "max_drawdown"
                        ].map(
                            format_percent
                        )

                    if (
                        "sharpe_ratio"
                        in display_frame.columns
                    ):
                        display_frame[
                            "sharpe_ratio"
                        ] = display_frame[
                            "sharpe_ratio"
                        ].map(
                            format_ratio
