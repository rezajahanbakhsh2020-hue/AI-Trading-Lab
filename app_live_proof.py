from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_live import fetch_live_data
from live_signal import build_live_signal
from live_trend import build_live_trend
from live_risk_levels import calculate_risk_levels


DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 200
DEFAULT_MOMENTUM_WINDOW = 10
DEFAULT_FAST_WINDOW = 20
DEFAULT_SLOW_WINDOW = 50
DEFAULT_STOP_LOSS_PCT = 0.01
DEFAULT_TAKE_PROFIT_PCT = 0.02


def build_live_proof_view(
    candles: pd.DataFrame,
    quote: dict[str, Any],
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> dict[str, Any]:
    """Build the complete human-readable live proof view."""
    signal = build_live_signal(
        candles,
        symbol=symbol,
        interval=interval,
        window=momentum_window,
    )

    trend = build_live_trend(
        candles,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    entry_price = float(
        quote.get(
            "latest_mid",
            candles["close"].iloc[-1],
        )
    )

    risk = calculate_risk_levels(
        entry_price=entry_price,
        signal=signal["signal"],
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "candle_count": int(len(candles)),
        "latest_candle": candles.iloc[-1].to_dict(),
        "signal": signal,
        "trend": trend,
        "risk": risk,
        "quote": quote,
    }


def build_candlestick_chart(
    candles: pd.DataFrame,
    proof: dict[str, Any],
) -> go.Figure:
    """Build the visual live-proof chart."""
    frame = candles.copy()

    timestamp_column = next(
        (
            column
            for column in ("timestamp", "datetime", "time", "date")
            if column in frame.columns
        ),
        None,
    )

    if timestamp_column is None:
        x_values = frame.index
    else:
        x_values = frame[timestamp_column]

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=x_values,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="XAU/USD",
        )
    )

    quote = proof["quote"]
    entry_price = proof["risk"].get("entry_price")

    if entry_price is not None:
        figure.add_hline(
            y=float(entry_price),
            line_dash="dash",
            annotation_text="Entry",
        )

    stop_loss = proof["risk"].get("stop_loss")
    if stop_loss is not None:
        figure.add_hline(
            y=float(stop_loss),
            line_dash="dash",
            annotation_text="SL",
        )

    take_profit = proof["risk"].get("take_profit")
    if take_profit is not None:
        figure.add_hline(
            y=float(take_profit),
            line_dash="dash",
            annotation_text="TP",
        )

    latest_mid = quote.get("latest_mid")
    if latest_mid is not None:
        figure.add_hline(
            y=float(latest_mid),
            line_dash="dot",
            annotation_text="Market",
        )

    figure.update_layout(
        title="XAU/USD Live Proof",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
    )

    return figure


def build_human_explanation(proof: dict[str, Any]) -> str:
    """Convert the live proof result into a human-readable explanation."""
    signal = proof["signal"]
    trend = proof["trend"]
    risk = proof["risk"]

    signal_label = signal.get("signal_label", "UNKNOWN")
    trend_label = trend.get("trend", "UNKNOWN")

    if signal_label == "BUY":
        explanation = (
            f"Signal is BUY while the detected trend is {trend_label}. "
            f"Entry is {risk.get('entry_price')}."
        )

        if risk.get("stop_loss") is not None:
            explanation += (
                f" Stop Loss is {risk['stop_loss']} and "
                f"Take Profit is {risk['take_profit']}."
            )

        return explanation

    return (
        f"Signal is {signal_label}. "
        f"Detected trend is {trend_label}. "
        "No trade risk levels are active."
    )


def load_saved_snapshot(
    path: str | Path = "results/live/live_snapshot.json",
) -> dict[str, Any] | None:
    """Load an existing persisted live snapshot when available."""
    snapshot_path = Path(path)

    if not snapshot_path.exists():
        return None

    with snapshot_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("saved snapshot must contain a JSON object")

    return data


def run_app() -> None:
    """Render the Streamlit live-proof dashboard."""
    st.set_page_config(
        page_title="AI Trading Lab - Live Proof",
        layout="wide",
    )

    st.title("AI Trading Lab — XAU/USD Live Proof")
    st.caption(
        "Real market data → candle → signal → trend → risk levels"
    )

    symbol = st.sidebar.text_input(
        "Symbol",
        value=DEFAULT_SYMBOL,
    )

    interval = st.sidebar.selectbox(
        "Interval",
        options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=1,
    )

    limit = st.sidebar.slider(
        "Candles",
        min_value=50,
        max_value=1000,
        value=DEFAULT_LIMIT,
        step=50,
    )

    refresh = st.button("Refresh live data")

    if refresh or "live_proof_data" not in st.session_state:
        with st.spinner("Loading real XAU/USD market data..."):
            candles, quote = fetch_live_data(
                symbol=symbol,
                interval=interval,
                limit=limit,
            )

            st.session_state["live_proof_data"] = build_live_proof_view(
                candles=candles,
                quote=quote,
                symbol=symbol,
                interval=interval,
            )

    proof = st.session_state["live_proof_data"]

    signal = proof["signal"]
    trend = proof["trend"]
    risk = proof["risk"]
    quote = proof["quote"]

    columns = st.columns(5)

    columns[0].metric(
        "Market",
        proof["symbol"],
    )

    columns[1].metric(
        "Signal",
        signal.get("signal_label", "UNKNOWN"),
    )

    columns[2].metric(
        "Trend",
        trend.get("trend", "UNKNOWN"),
    )

    columns[3].metric(
        "Entry",
        risk.get("entry_price", "—"),
    )

    columns[4].metric(
        "Candles",
        proof["candle_count"],
    )

    st.subheader("Live Chart")
    st.plotly_chart(
        build_candlestick_chart(
            proof.get("candles", pd.DataFrame()),
            proof,
        ),
        use_container_width=True,
    )

    st.subheader("Decision")

    if signal.get("signal_label") == "BUY":
        st.success(build_human_explanation(proof))
    else:
        st.info(build_human_explanation(proof))

    risk_columns = st.columns(4)

    risk_columns[0].metric(
        "Entry",
        risk.get("entry_price", "—"),
    )

    risk_columns[1].metric(
        "Stop Loss",
        risk.get("stop_loss", "—"),
    )

    risk_columns[2].metric(
        "Take Profit",
        risk.get("take_profit", "—"),
    )

    risk_columns[3].metric(
        "Risk / Reward",
        risk.get("risk_reward_ratio", "—"),
    )

    st.subheader("Market Status")

    status = {
        "market_state": quote.get("market_state"),
        "quote_stale": quote.get("quote_stale"),
        "quote_age_seconds": quote.get("quote_age_seconds"),
        "latest_bid": quote.get("latest_bid"),
        "latest_ask": quote.get("latest_ask"),
        "latest_mid": quote.get("latest_mid"),
    }

    st.json(status)

    with st.expander("Raw Live Proof"):
        st.json(proof)


if __name__ == "__main__":
    run_app()
