from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


BIQUOTE_BASE_URL = "https://biquote.io"
XAUUSD_SYMBOL = "XAUUSD"

SUPPORTED_INTERVALS = {
    "1m": "1 Minute",
    "5m": "5 Minutes",
    "15m": "15 Minutes",
    "30m": "30 Minutes",
    "1h": "1 Hour",
    "4h": "4 Hours",
    "1d": "1 Day",
}


def fetch_xauusd_ohlc(
    interval: str = "5m",
    limit: int = 200,
    timeout: int = 10,
) -> pd.DataFrame:
    """Fetch real XAU/USD OHLC candles from BiQuote."""

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")

    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000.")

    url = (
        f"{BIQUOTE_BASE_URL}/api/"
        f"{XAUUSD_SYMBOL}/ohlc"
        f"?interval={interval}&limit={limit}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "AI-Trading-Lab/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        raise RuntimeError(
            f"BiQuote HTTP error: {exc.code}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Unable to reach BiQuote: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "BiQuote request timed out."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "BiQuote returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "BiQuote response must be a JSON object."
        )

    if payload.get("symbol") != XAUUSD_SYMBOL:
        raise ValueError(
            "BiQuote returned an unexpected symbol."
        )

    bars = payload.get("bars")

    if not isinstance(bars, list):
        raise ValueError(
            "BiQuote response does not contain bars."
        )

    if not bars:
        raise ValueError(
            "BiQuote returned no XAU/USD candles."
        )

    data = pd.DataFrame(bars)

    required_columns = {
        "openTime",
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
            "Missing required candle columns: "
            f"{missing_columns}"
        )

    data["timestamp"] = pd.to_datetime(
        data["openTime"],
        errors="coerce",
        utc=True,
    )

    if data["timestamp"].isna().any():
        raise ValueError(
            "BiQuote returned invalid candle timestamps."
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[numeric_columns].isna().any().any():
        raise ValueError(
            "BiQuote returned invalid OHLC values."
        )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return data


def fetch_xauusd_quote(
    timeout: int = 10,
) -> dict:
    """Fetch the latest real XAU/USD quote."""

    url = (
        f"{BIQUOTE_BASE_URL}/api/"
        f"{XAUUSD_SYMBOL}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "AI-Trading-Lab/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        raise RuntimeError(
            f"BiQuote quote HTTP error: {exc.code}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Unable to reach BiQuote quote endpoint: "
            f"{exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "BiQuote quote request timed out."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "BiQuote quote response is invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "BiQuote quote response must be an object."
        )

    if payload.get("symbol") != XAUUSD_SYMBOL:
        raise ValueError(
            "BiQuote returned an unexpected quote symbol."
        )

    if "mid" not in payload:
        raise ValueError(
            "BiQuote quote does not contain mid price."
        )

    return payload


def build_live_chart(
    data: pd.DataFrame,
) -> go.Figure:
    """Build the live XAU/USD candlestick chart."""

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="XAU/USD",
        )
    )

    figure.update_layout(
        title="XAU/USD Live Candlestick",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (USD)",
        height=650,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=10,
        ),
    )

    return figure


def render_live_market(
    interval: str,
    limit: int,
) -> None:
    """Render one live market refresh."""

    try:
        quote = fetch_xauusd_quote()
        candles = fetch_xauusd_ohlc(
            interval=interval,
            limit=limit,
        )
    except Exception as exc:
        st.error(
            f"Live XAU/USD data unavailable: {exc}"
        )
        return

    latest_price = float(quote["mid"])

    previous_close = None

    if len(candles) >= 2:
        previous_close = float(
            candles["close"].iloc[-2]
        )

    change = None
    change_percent = None

    if (
        previous_close is not None
        and previous_close != 0
    ):
        change = latest_price - previous_close
        change_percent = (
            change / previous_close
        ) * 100.0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "XAU/USD",
        f"{latest_price:,.2f}",
    )

    if change is not None:
        col2.metric(
            "Change",
            f"{change:+,.2f}",
        )
    else:
        col2.metric(
            "Change",
            "N/A",
        )

    if change_percent is not None:
        col3.metric(
            "Change %",
            f"{change_percent:+.2f}%",
        )
    else:
        col3.metric(
            "Change %",
            "N/A",
        )

    market_state = quote.get(
        "marketState",
        "unknown",
    )

    col4.metric(
        "Market",
        str(market_state).upper(),
    )

    quote_time = quote.get(
        "timestamp",
        "N/A",
    )

    st.caption(
        "Real XAU/USD data | "
        f"BiQuote | Quote time: {quote_time}"
    )

    st.plotly_chart(
        build_live_chart(candles),
        use_container_width=True,
    )

    latest_bar = candles.iloc[-1]

    st.subheader("Latest Candle")

    candle_columns = st.columns(5)

    candle_columns[0].metric(
        "Open",
        f"{float(latest_bar['open']):,.2f}",
    )

    candle_columns[1].metric(
        "High",
        f"{float(latest_bar['high']):,.2f}",
    )

    candle_columns[2].metric(
        "Low",
        f"{float(latest_bar['low']):,.2f}",
    )

    candle_columns[3].metric(
        "Close",
        f"{float(latest_bar['close']):,.2f}",
    )

    candle_columns[4].metric(
        "Bars",
        str(len(candles)),
    )

    st.caption(
        "The chart uses real OHLC candles from "
        "the selected XAU/USD timeframe."
    )


def main() -> None:
    st.set_page_config(
        page_title="AI Trading Lab - Live XAU/USD",
        page_icon="📈",
        layout="wide",
    )

    st.title(
        "📈 AI Trading Lab — Live XAU/USD"
    )

    st.caption(
        "First practical proof: "
        "real XAU/USD market data → "
        "real candlesticks → visual dashboard"
    )

    st.sidebar.header(
        "Live Market Controls"
    )

    interval = st.sidebar.selectbox(
        "Timeframe",
        options=list(
            SUPPORTED_INTERVALS.keys()
        ),
        index=1,
        format_func=lambda value: (
            f"{value} — "
            f"{SUPPORTED_INTERVALS[value]}"
        ),
    )

    limit = st.sidebar.slider(
        "Number of candles",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
    )

    refresh_interval = st.sidebar.selectbox(
        "Refresh interval",
        options=[5, 10, 15, 30],
        index=1,
        format_func=lambda value: (
            f"Every {value} seconds"
        ),
    )

    try:
        fragment = st.fragment
    except AttributeError:
        fragment = None

    if fragment is None:
        st.warning(
            "Automatic refresh requires a recent "
            "Streamlit version. Use the browser "
            "refresh button for a manual update."
        )

        render_live_market(
            interval,
            limit,
        )
        return

    @fragment(
        run_every=f"{refresh_interval}s"
    )
    def live_market_fragment() -> None:
        render_live_market(
            interval,
            limit,
        )

    live_market_fragment()


if __name__ == "__main__":
    main()
