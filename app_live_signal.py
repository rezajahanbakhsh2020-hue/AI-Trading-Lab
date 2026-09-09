from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MIN_LIMIT,
    SUPPORTED_INTERVALS,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from live_signal import (
    DEFAULT_MOMENTUM_WINDOW,
    build_live_signal_snapshot,
)


def build_signal_chart(data):
    """Build the live XAU/USD candlestick chart."""

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=data["openTime"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
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
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return figure


def signal_description(snapshot: dict) -> str:
    """Create a human-readable explanation from real strategy output."""

    signal = snapshot["signal_label"]
    momentum = snapshot["momentum"]
    close = snapshot["close"]

    if momentum is None:
        return (
            f"Current price is {close:,.2f}. "
            "There is not enough historical data yet to calculate "
            "the momentum value."
        )

    if signal == "BUY":
        return (
            f"Momentum is positive ({momentum:+.4%}). "
            f"The existing momentum strategy currently produces BUY "
            f"at {close:,.2f}."
        )

    return (
        f"Momentum is non-positive ({momentum:+.4%}). "
        f"The existing momentum strategy currently produces NO TRADE "
        f"at {close:,.2f}."
    )


def render_signal_dashboard(
    interval: str,
    limit: int,
    window: int,
) -> None:
    """Render real market data and the existing strategy signal."""

    try:
        data = fetch_xauusd_ohlc(
            interval=interval,
            limit=limit,
        )
        quote = fetch_xauusd_quote()

        snapshot = build_live_signal_snapshot(
            data=data,
            window=window,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        st.error(f"Live signal error: {exc}")
        return

    latest_price = float(quote["mid"])
    signal = snapshot["signal_label"]
    market_state = str(
        quote.get("marketState", "unknown")
    ).upper()

    quote_age = quote.get("quoteAgeSeconds")
    stale = bool(quote.get("stale", False))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "XAU/USD",
            f"{latest_price:,.2f}",
        )

    with col2:
        st.metric(
            "Signal",
            signal,
        )

    with col3:
        st.metric(
            "Strategy",
            snapshot["strategy"].upper(),
        )

    with col4:
        st.metric(
            "Market",
            market_state,
        )

    if stale:
        st.warning(
            "The latest XAU/USD quote is marked as stale "
            "by the data provider."
        )

    if signal == "BUY":
        st.success("BUY signal from the existing Momentum strategy.")
    else:
        st.info(
            "NO TRADE: the existing Momentum strategy "
            "does not currently produce a BUY signal."
        )

    st.subheader("Market Chart")

    figure = build_signal_chart(data)

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.subheader("Strategy Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        momentum = snapshot["momentum"]

        if momentum is None:
            momentum_text = "Not enough data"
        else:
            momentum_text = f"{momentum:+.4%}"

        st.metric(
            "Momentum",
            momentum_text,
        )

    with status_col2:
        st.metric(
            "Entry Price",
            f"{snapshot['close']:,.2f}",
        )

    with status_col3:
        if quote_age is None:
            age_text = "Unknown"
        else:
            age_text = f"{float(quote_age):.1f}s"

        st.metric(
            "Quote Age",
            age_text,
        )

    st.subheader("Human Explanation")

    st.write(
        signal_description(snapshot)
    )

    st.caption(
        f"Strategy: {snapshot['strategy']} | "
        f"Momentum window: {snapshot['window']} | "
        f"Timeframe: {interval} | "
        f"Candles: {len(data)}"
    )

    if "timestamp" in snapshot:
        st.caption(
            f"Signal timestamp: {snapshot['timestamp']}"
        )

    st.caption(
        "Data source: BiQuote | "
        "Signal source: existing AI-Trading-Lab Momentum strategy"
    )


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | Live Signal",
        page_icon="📈",
        layout="wide",
    )

    st.title("AI-Trading-Lab — XAU/USD Live Signal")

    st.caption(
        "Real XAU/USD market data connected to the existing "
        "AI-Trading-Lab Momentum strategy."
    )

    sidebar = st.sidebar

    sidebar.header("Live Signal")

    interval = sidebar.selectbox(
        "Timeframe",
        options=SUPPORTED_INTERVALS,
        index=SUPPORTED_INTERVALS.index(DEFAULT_INTERVAL),
    )

    limit = sidebar.number_input(
        "Candles",
        min_value=MIN_LIMIT,
        max_value=MAX_LIMIT,
        value=DEFAULT_LIMIT,
        step=10,
    )

    window = sidebar.number_input(
        "Momentum Window",
        min_value=1,
        max_value=100,
        value=DEFAULT_MOMENTUM_WINDOW,
        step=1,
    )

    refresh_seconds = sidebar.number_input(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=300,
        value=15,
        step=5,
    )

    sidebar.caption(
        "The signal is calculated from real XAU/USD candles "
        "using the existing Momentum strategy."
    )

    if hasattr(st, "fragment"):

        @st.fragment(run_every=f"{int(refresh_seconds)}s")
        def live_fragment() -> None:
            render_signal_dashboard(
                interval=interval,
                limit=int(limit),
                window=int(window),
            )

        live_fragment()

    else:
        render_signal_dashboard(
            interval=interval,
            limit=int(limit),
            window=int(window),
        )

        st.info(
            "Automatic refresh is unavailable in this "
            "Streamlit version. Reload the page to refresh."
        )


if __name__ == "__main__":
    main()
