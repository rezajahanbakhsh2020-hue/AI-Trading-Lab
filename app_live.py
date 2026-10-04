from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.biquote import (
    BIQUOTE_BASE_URL,
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    DEFAULT_TIMEOUT,
    MAX_LIMIT,
    MIN_LIMIT,
    SUPPORTED_INTERVALS,
    XAUUSD_SYMBOL,
    _get_json,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)


def build_live_chart(data: pd.DataFrame) -> go.Figure:
    """Build a Plotly XAU/USD candlestick chart."""

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
            "Chart data is missing required columns: "
            + ", ".join(sorted(missing))
        )

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
        title="XAU/USD Live Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return figure


def _price_change(data: pd.DataFrame) -> tuple[float, float]:
    if len(data) < 2:
        return 0.0, 0.0

    previous_close = float(data.iloc[-2]["close"])
    latest_close = float(data.iloc[-1]["close"])

    change = latest_close - previous_close

    if previous_close == 0:
        change_percent = 0.0
    else:
        change_percent = (change / previous_close) * 100.0

    return change, change_percent


def render_live_market(
    interval: str,
    limit: int,
) -> None:
    """Render the current XAU/USD market view."""

    try:
        data = fetch_xauusd_ohlc(
            interval=interval,
            limit=limit,
        )
        quote = fetch_xauusd_quote()
    except (RuntimeError, ValueError) as exc:
        st.error(f"Live market data error: {exc}")
        return

    latest_price = float(quote["mid"])
    change, change_percent = _price_change(data)

    market_state = str(
        quote.get("marketState", "unknown")
    ).upper()

    stale = bool(quote.get("stale", False))
    quote_age = quote.get("quoteAgeSeconds")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "XAU/USD",
            f"{latest_price:,.2f}",
            f"{change:+,.2f} ({change_percent:+.2f}%)",
        )

    with col2:
        st.metric(
            "Market",
            market_state,
        )

    with col3:
        if quote_age is None:
            age_text = "Unknown"
        else:
            age_text = f"{float(quote_age):.1f}s"

        st.metric(
            "Quote Age",
            age_text,
        )

    if stale:
        st.warning(
            "The latest XAU/USD quote is marked as stale by the data provider."
        )

    latest_candle = data.iloc[-1]

    candle_status = "OPEN" if bool(
        latest_candle.get("isOpen", False)
    ) else "CLOSED"

    st.caption(
        "Latest candle: "
        f"{candle_status} | "
        f"Open {float(latest_candle['open']):,.2f} | "
        f"High {float(latest_candle['high']):,.2f} | "
        f"Low {float(latest_candle['low']):,.2f} | "
        f"Close {float(latest_candle['close']):,.2f}"
    )

    figure = build_live_chart(data)

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.caption(
        f"Source: BiQuote | {XAUUSD_SYMBOL} | "
        f"{interval} | {len(data)} candles"
    )


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | XAU/USD Live",
        page_icon="📈",
        layout="wide",
    )

    st.title("XAU/USD Live Chart")
    st.caption(
        "AI-Trading-Lab — real XAU/USD market data"
    )

    sidebar = st.sidebar

    sidebar.header("Live Chart")

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

    refresh_seconds = sidebar.number_input(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=300,
        value=15,
        step=5,
    )

    sidebar.caption(
        "The chart uses real XAU/USD OHLC data. "
        "Automatic refresh is used for this first live proof stage."
    )

    if hasattr(st, "fragment"):
        @st.fragment(run_every=f"{int(refresh_seconds)}s")
        def live_fragment() -> None:
            render_live_market(
                interval=interval,
                limit=int(limit),
            )

        live_fragment()
    else:
        render_live_market(
            interval=interval,
            limit=int(limit),
        )

        st.info(
            "Automatic fragment refresh is unavailable in this "
            "Streamlit version. Reload the page to refresh the data."
        )


if __name__ == "__main__":
    main()
