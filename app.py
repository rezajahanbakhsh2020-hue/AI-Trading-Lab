from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.visualization.chart_data import prepare_chart_data
from src.visualization.chart_engine import build_signal_markers
from src.evaluation.market_regime import classify_volatility_regime


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

    The loader supports both Unix timestamps and normal
    datetime strings.
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
    Add volatility regime information to market data.
    """

    return classify_volatility_regime(
        df,
        return_column="return",
        window=20,
    )


# ---------------------------------------------------------------------
# Chart construction
# ---------------------------------------------------------------------

def build_market_chart(
    df: pd.DataFrame,
    show_volume: bool = True,
) -> go.Figure:
    """
    Build the main interactive XAU/USD chart.
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
            increasing_line_width=1,
            decreasing_line_width=1,
        )
    )

    if "signal" in df.columns:
        markers = build_signal_markers(
            df,
            signal_column="signal",
        )

        buys = markers[
            markers["marker"] == "buy"
        ]

        sells = markers[
            markers["marker"] == "sell"
        ]

        if not buys.empty:
            figure.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy",
                    marker={
                        "symbol": "triangle-up",
                        "size": 12,
                    },
                )
            )

        if not sells.empty:
            figure.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell",
                    marker={
                        "symbol": "triangle-down",
                        "size": 12,
                    },
                )
            )

    if (
        show_volume
        and "volume" in df.columns
    ):
        figure.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                opacity=0.30,
                yaxis="y2",
            )
        )

        figure.update_layout(
            yaxis2={
                "title": "Volume",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            }
        )

    figure.update_layout(
        title={
            "text": "XAU/USD — Interactive Market Chart",
            "x": 0.02,
        },
        height=680,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("⚙️ Research Controls")

st.sidebar.caption(
    "AI Trading Lab"
)

st.sidebar.divider()

display_days = st.sidebar.slider(
    "Chart History",
    min_value=30,
    max_value=365,
    value=120,
    step=10,
)

show_volume = st.sidebar.checkbox(
    "Show Volume",
    value=True,
)

show_regime = st.sidebar.checkbox(
    "Show Market Regime",
    value=True,
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
        "Unable to load market data."
    )
    st.code(str(exc))
    st.stop()


market_data = add_market_metrics(
    market_data
)


if show_regime:
    market_data = prepare_regime_data(
        market_data
    )


display_data = market_data.tail(
    display_days
).copy()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("📈 AI Trading Lab")

st.caption(
    "Research • Backtest • Market Regime • Strategy Analysis"
)

st.divider()


# ---------------------------------------------------------------------
# Current market information
# ---------------------------------------------------------------------

latest = market_data.iloc[-1]

latest_price = float(
    latest["close"]
)

latest_change = float(
    latest["daily_change_pct"]
)

start_price = float(
    market_data.iloc[0]["close"]
)

period_return = (
    (latest_price / start_price) - 1.0
) * 100.0


if (
    show_regime
    and pd.notna(
        latest.get("volatility_regime")
    )
):
    regime = str(
        latest["volatility_regime"]
    )
else:
    regime = "Not available"


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Instrument",
        "XAU/USD",
    )

with col2:
    st.metric(
        "Last Price",
        f"{latest_price:,.2f}",
        f"{latest_change:+.2f}%",
    )

with col3:
    st.metric(
        "Period Return",
        f"{period_return:+.2f}%",
    )

with col4:
    st.metric(
        "Market Regime",
        regime.replace(
            "_",
            " ",
        ).title(),
    )

with col5:
    st.metric(
        "Data Points",
        f"{len(market_data):,}",
    )


st.divider()


# ---------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------

st.subheader("Market")

figure = build_market_chart(
    display_data,
    show_volume=show_volume,
)

st.plotly_chart(
    figure,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
    },
)


# ---------------------------------------------------------------------
# Market statistics
# ---------------------------------------------------------------------

st.divider()

st.subheader("Market Statistics")

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:
    st.metric(
        "Period High",
        f"{display_data['high'].max():,.2f}",
    )

with stat2:
    st.metric(
        "Period Low",
        f"{display_data['low'].min():,.2f}",
    )

with stat3:
    volatility = (
        display_data["return"]
        .std()
        * 100.0
    )

    st.metric(
        "Daily Volatility",
        (
            f"{volatility:.2f}%"
            if pd.notna(volatility)
            else "N/A"
        ),
    )

with stat4:
    if "volume" in display_data.columns:
        st.metric(
            "Avg Volume",
            f"{display_data['volume'].mean():,.0f}",
        )
    else:
        st.metric(
            "Volume",
            "Unavailable",
        )


# ---------------------------------------------------------------------
# Research panels
# ---------------------------------------------------------------------

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("🔬 Market Regime")

    if show_regime:
        regime_counts = (
            display_data[
                "volatility_regime"
            ]
            .value_counts()
        )

        if regime_counts.empty:
            st.info(
                "Regime data is not available yet."
            )
        else:
            regime_table = (
                regime_counts
                .rename("Days")
                .to_frame()
            )

            st.dataframe(
                regime_table,
                width="stretch",
            )
    else:
        st.info(
            "Market regime display is disabled."
        )


with right:
    st.subheader("🎯 Strategy")

    strategy = st.selectbox(
        "Strategy",
        [
            "Moving Average",
            "Momentum",
        ],
    )

    st.write(
        f"Selected strategy: **{strategy}**"
    )

    st.info(
        "Strategy execution will be connected "
        "to the real backtest pipeline in the "
        "next dashboard stage."
    )


# ---------------------------------------------------------------------
# Data information
# ---------------------------------------------------------------------

st.divider()

st.subheader("📊 Dataset")

data_col1, data_col2, data_col3 = st.columns(3)

with data_col1:
    st.write(
        "**Start:** "
        f"{market_data['timestamp'].min():%Y-%m-%d}"
    )

with data_col2:
    st.write(
        "**End:** "
        f"{market_data['timestamp'].max():%Y-%m-%d}"
    )

with data_col3:
    st.write(
        "**Columns:** "
        + ", ".join(
            market_data.columns
        )
    )


st.caption(
    "AI Trading Lab — Research Dashboard"
)
