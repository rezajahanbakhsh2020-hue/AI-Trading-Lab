from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.visualization.chart_data import prepare_chart_data
from src.visualization.chart_engine import build_signal_markers


st.set_page_config(
    page_title="AI Trading Lab",
    page_icon="📈",
    layout="wide",
)


def build_demo_data() -> pd.DataFrame:
    """Create a small deterministic dataset for the dashboard."""

    dates = pd.date_range(
        start="2026-01-01",
        periods=30,
        freq="D",
    )

    close = [
        2650,
        2662,
        2648,
        2675,
        2690,
        2682,
        2705,
        2720,
        2712,
        2735,
        2750,
        2742,
        2768,
        2780,
        2772,
        2795,
        2810,
        2802,
        2825,
        2840,
        2832,
        2855,
        2870,
        2862,
        2885,
        2900,
        2892,
        2915,
        2930,
        2922,
    ]

    open_prices = [
        close[0] - 5,
        *[value - 4 for value in close[1:]],
    ]

    high = [
        max(o, c) + 8
        for o, c in zip(open_prices, close)
    ]

    low = [
        min(o, c) - 8
        for o, c in zip(open_prices, close)
    ]

    volume = [
        100 + index * 7
        for index in range(len(dates))
    ]

    signal = [0] * len(dates)
    signal[5] = 1
    signal[14] = -1
    signal[20] = 1
    signal[26] = -1

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "signal": signal,
        }
    )


def build_candlestick_chart(
    df: pd.DataFrame,
) -> go.Figure:
    """Build the main interactive market chart."""

    chart_data = prepare_chart_data(df)

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=chart_data["timestamp"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="XAU/USD",
        )
    )

    if "signal" in chart_data.columns:
        markers = build_signal_markers(
            chart_data,
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

    figure.update_layout(
        title="XAU/USD Market Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    return figure


st.title("📈 AI Trading Lab")

st.caption(
    "Research • Backtest • Market Regime • Strategy Analysis"
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Instrument",
        "XAU/USD",
    )

with col2:
    st.metric(
        "Timeframe",
        "Daily",
    )

with col3:
    st.metric(
        "Market Regime",
        "Normal",
    )

with col4:
    st.metric(
        "Status",
        "Research",
    )

st.divider()

data = build_demo_data()

figure = build_candlestick_chart(data)

st.plotly_chart(
    figure,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
    },
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Strategy")

    strategy = st.selectbox(
        "Select Strategy",
        [
            "Moving Average",
            "Momentum",
        ],
    )

    st.write(
        f"Selected strategy: **{strategy}**"
    )

with right:
    st.subheader("Research Status")

    st.success(
        "Dashboard foundation is ready."
    )

st.divider()

st.subheader("Future Modules")

future_modules = [
    "Backtest Engine",
    "Walk-Forward Analysis",
    "Market Regime",
    "Risk Management",
    "Strategy Selection",
    "Performance Analytics",
    "Machine Learning",
]

for module in future_modules:
    st.write(f"• {module}")
