from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_live import fetch_xauusd_ohlc, fetch_xauusd_quote
from live_signal import (
    DEFAULT_MOMENTUM_WINDOW,
    build_live_signal_snapshot,
)
from live_trend import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    build_live_trend_snapshot,
)


DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 200
DEFAULT_REFRESH_SECONDS = 15

SUPPORTED_INTERVALS = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)


def build_signal_trend_snapshot(
    data: pd.DataFrame,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
) -> dict:
    """Combine the existing live signal and live trend outputs."""

    signal_snapshot = build_live_signal_snapshot(
        data,
        window=momentum_window,
    )

    trend_snapshot = build_live_trend_snapshot(
        data,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    snapshot = {
        "signal": signal_snapshot["signal"],
        "signal_label": signal_snapshot["signal_label"],
        "momentum": signal_snapshot["momentum"],
        "strategy": signal_snapshot["strategy"],
        "momentum_window": signal_snapshot["window"],
        "trend": trend_snapshot["trend"],
        "fast_window": trend_snapshot["fast_window"],
        "slow_window": trend_snapshot["slow_window"],
        "fast_ma": trend_snapshot["fast_ma"],
        "slow_ma": trend_snapshot["slow_ma"],
        "close": trend_snapshot["close"],
    }

    if "timestamp" in signal_snapshot:
        snapshot["timestamp"] = signal_snapshot["timestamp"]
    elif "timestamp" in trend_snapshot:
        snapshot["timestamp"] = trend_snapshot["timestamp"]

    return snapshot


def build_candlestick_chart(data: pd.DataFrame) -> go.Figure:
    """Build a candlestick chart from the supplied OHLC data."""

    required_columns = {
        "openTime",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required chart columns: {sorted(missing)}"
        )

    chart_data = data.copy()
    chart_data["openTime"] = pd.to_datetime(
        chart_data["openTime"],
        utc=True,
        errors="coerce",
    )

    if chart_data["openTime"].isna().any():
        raise ValueError("Invalid openTime values in chart data.")

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=chart_data["openTime"],
                open=chart_data["open"],
                high=chart_data["high"],
                low=chart_data["low"],
                close=chart_data["close"],
                name="XAU/USD",
            )
        ]
    )

    figure.update_layout(
        title="XAU/USD Live Market",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
    )

    return figure


def _render_dashboard(
    interval: str,
    limit: int,
    momentum_window: int,
    fast_window: int,
    slow_window: int,
) -> None:
    st.title("XAU/USD Live Signal + Trend")

    st.caption(
        "Real XAU/USD market data connected to the existing "
        "AI-Trading-Lab signal and trend engines."
    )

    try:
        data = fetch_xauusd_ohlc(
            interval=interval,
            limit=limit,
        )

        quote = fetch_xauusd_quote()

        snapshot = build_signal_trend_snapshot(
            data,
            momentum_window=momentum_window,
            fast_window=fast_window,
            slow_window=slow_window,
        )

    except Exception as exc:
        st.error(f"Live market data error: {exc}")
        return

    market_state = str(
        quote.get("marketState", "unknown")
    ).upper()

    if bool(quote.get("stale", False)):
        st.warning(
            "The latest quote is marked as stale by the data source."
        )

    st.subheader("Current Decision State")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Signal",
            snapshot["signal_label"],
        )

    with col2:
        st.metric(
            "Trend",
            snapshot["trend"],
        )

    with col3:
        st.metric(
            "Price",
            f'{snapshot["close"]:.2f}',
        )

    with col4:
        st.metric(
            "Market",
            market_state,
        )

    st.plotly_chart(
        build_candlestick_chart(data),
        use_container_width=True,
    )

    st.subheader("Engine Output")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Signal Engine**")
        st.write(
            f"Strategy: `{snapshot['strategy']}`"
        )
        st.write(
            f"Momentum window: `{snapshot['momentum_window']}`"
        )
        st.write(
            f"Momentum: `{snapshot['momentum']:.6f}`"
        )
        st.write(
            f"Signal: `{snapshot['signal_label']}`"
        )

    with col2:
        st.write("**Trend Engine**")
        st.write(
            f"Fast MA: `{snapshot['fast_ma']}`"
        )
        st.write(
            f"Slow MA: `{snapshot['slow_ma']}`"
        )
        st.write(
            f"Fast window: `{snapshot['fast_window']}`"
        )
        st.write(
            f"Slow window: `{snapshot['slow_window']}`"
        )
        st.write(
            f"Trend: `{snapshot['trend']}`"
        )

    st.subheader("Human-readable State")

    signal_label = snapshot["signal_label"]
    trend = snapshot["trend"]

    if trend == "INSUFFICIENT DATA":
        explanation = (
            f"Signal is {signal_label}, but there is not enough "
            "history yet to determine the moving-average trend."
        )
    else:
        explanation = (
            f"Signal engine: {signal_label}. "
            f"Trend engine: {trend}. "
            f"Current XAU/USD price: {snapshot['close']:.2f}."
        )

    st.info(explanation)

    if "timestamp" in snapshot:
        st.caption(
            f"Latest candle time: {snapshot['timestamp']}"
        )

    quote_age = quote.get("quoteAgeSeconds")

    if quote_age is not None:
        st.caption(
            f"Quote age: {quote_age} seconds"
        )

    st.caption("Data source: BiQuote XAU/USD")


def main() -> None:
    st.set_page_config(
        page_title="XAU/USD Live Signal + Trend",
        layout="wide",
    )

    st.sidebar.header("Live Settings")

    interval = st.sidebar.selectbox(
        "Timeframe",
        SUPPORTED_INTERVALS,
        index=SUPPORTED_INTERVALS.index(DEFAULT_INTERVAL),
    )

    limit = st.sidebar.number_input(
        "Candle count",
        min_value=50,
        max_value=1000,
        value=DEFAULT_LIMIT,
        step=50,
    )

    momentum_window = st.sidebar.number_input(
        "Momentum window",
        min_value=1,
        max_value=200,
        value=DEFAULT_MOMENTUM_WINDOW,
        step=1,
    )

    fast_window = st.sidebar.number_input(
        "Fast MA window",
        min_value=1,
        max_value=200,
        value=DEFAULT_FAST_WINDOW,
        step=1,
    )

    slow_window = st.sidebar.number_input(
        "Slow MA window",
        min_value=2,
        max_value=500,
        value=DEFAULT_SLOW_WINDOW,
        step=1,
    )

    refresh_seconds = st.sidebar.number_input(
        "Refresh seconds",
        min_value=5,
        max_value=300,
        value=DEFAULT_REFRESH_SECONDS,
        step=5,
    )

    if fast_window >= slow_window:
        st.sidebar.error(
            "Fast MA window must be smaller than slow MA window."
        )
        return

    render_kwargs = {
        "interval": interval,
        "limit": int(limit),
        "momentum_window": int(momentum_window),
        "fast_window": int(fast_window),
        "slow_window": int(slow_window),
    }

    fragment = getattr(st, "fragment", None)

    if fragment is not None:
        @fragment(run_every=int(refresh_seconds))
        def live_fragment() -> None:
            _render_dashboard(**render_kwargs)

        live_fragment()
    else:
        _render_dashboard(**render_kwargs)


if __name__ == "__main__":
    main()
