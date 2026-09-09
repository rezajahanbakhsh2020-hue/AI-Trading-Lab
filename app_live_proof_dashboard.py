"""Human-readable visual dashboard for the AI-Trading-Lab live proof."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    SUPPORTED_INTERVALS,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from live_risk_levels import (
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    build_live_risk_levels,
)
from live_signal import DEFAULT_MOMENTUM_WINDOW, build_live_signal_snapshot
from live_trend import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    build_live_trend_snapshot,
)


DEFAULT_REFRESH_SECONDS = 15


def build_live_proof_snapshot(
    data: pd.DataFrame,
    quote: dict[str, Any],
    *,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> dict[str, Any]:
    """Combine the existing live engines into one visual proof snapshot."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(quote, dict):
        raise ValueError("quote must be a dictionary.")

    signal = build_live_signal_snapshot(
        data,
        window=momentum_window,
    )

    trend = build_live_trend_snapshot(
        data,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    risk = build_live_risk_levels(
        data,
        momentum_window=momentum_window,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    latest_candle = data.iloc[-1]

    return {
        "symbol": "XAUUSD",
        "interval": None,
        "candle_count": int(len(data)),
        "signal": signal["signal"],
        "signal_label": signal["signal_label"],
        "strategy": signal["strategy"],
        "momentum": signal["momentum"],
        "trend": trend["trend"],
        "fast_ma": trend["fast_ma"],
        "slow_ma": trend["slow_ma"],
        "entry_price": risk["entry_price"],
        "stop_loss": risk["stop_loss"],
        "take_profit": risk["take_profit"],
        "risk_reward_ratio": risk["risk_reward_ratio"],
        "stop_loss_pct": risk["stop_loss_pct"],
        "take_profit_pct": risk["take_profit_pct"],
        "timestamp": signal["timestamp"],
        "latest_candle": {
            "open": float(latest_candle["open"]),
            "high": float(latest_candle["high"]),
            "low": float(latest_candle["low"]),
            "close": float(latest_candle["close"]),
        },
        "market_state": str(
            quote.get("marketState", "UNKNOWN")
        ).upper(),
        "quote_stale": bool(
            quote.get("stale", False)
        ),
        "quote_age_seconds": quote.get(
            "quoteAgeSeconds"
        ),
        "bid": quote.get("bid"),
        "ask": quote.get("ask"),
        "mid": quote.get("mid"),
    }


def build_live_proof_chart(
    data: pd.DataFrame,
    snapshot: dict[str, Any],
) -> go.Figure:
    """Build the candlestick chart with active system levels."""

    required = {
        "openTime",
        "open",
        "high",
        "low",
        "close",
    }

    missing = sorted(
        required.difference(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required chart columns: {missing}"
        )

    if data.empty:
        raise ValueError("data must not be empty.")

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=data["openTime"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="XAU/USD",
        )
    )

    levels = (
        ("Entry", snapshot.get("entry_price")),
        ("SL", snapshot.get("stop_loss")),
        ("TP", snapshot.get("take_profit")),
    )

    for label, value in levels:
        if value is not None:
            figure.add_hline(
                y=float(value),
                line_dash="dash",
                annotation_text=label,
            )

    market_price = snapshot.get("mid")

    if market_price is not None:
        figure.add_hline(
            y=float(market_price),
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


def build_human_explanation(
    snapshot: dict[str, Any],
) -> str:
    """Create a concise human-readable explanation."""

    signal = snapshot["signal_label"]
    trend = snapshot["trend"]

    if signal == "BUY":
        text = (
            f"BUY signal detected. "
            f"Current trend is {trend}. "
            f"Entry price is "
            f"{snapshot['entry_price']:.2f}."
        )

        if snapshot["stop_loss"] is not None:
            text += (
                f" Stop Loss: "
                f"{snapshot['stop_loss']:.2f}."
            )

        if snapshot["take_profit"] is not None:
            text += (
                f" Take Profit: "
                f"{snapshot['take_profit']:.2f}."
            )

        return text

    return (
        f"NO TRADE. "
        f"Current trend is {trend}. "
        "The current strategy has not produced an active "
        "BUY condition, so SL/TP are not issued."
    )


def _render_dashboard() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | Live Proof",
        page_icon="📊",
        layout="wide",
    )

    st.title("AI-Trading-Lab — XAU/USD Live Proof")

    st.caption(
        "Real XAU/USD → candle → strategy → signal → "
        "trend → risk → human-readable output"
    )

    with st.sidebar:
        st.header("Live Settings")

        interval = st.selectbox(
            "Timeframe",
            SUPPORTED_INTERVALS,
            index=SUPPORTED_INTERVALS.index(
                DEFAULT_INTERVAL
            ),
        )

        limit = st.number_input(
            "Candle count",
            min_value=1,
            max_value=MAX_LIMIT,
            value=DEFAULT_LIMIT,
            step=1,
        )

    try:
        data = fetch_xauusd_ohlc(
            interval=interval,
            limit=int(limit),
        )

        quote = fetch_xauusd_quote()

        snapshot = build_live_proof_snapshot(
            data,
            quote,
        )

        snapshot["interval"] = interval

    except Exception as exc:
        st.error(
            f"Unable to build live proof: {exc}"
        )
        return

    if snapshot["quote_stale"]:
        st.warning(
            "The latest quote is marked as stale."
        )

    top1, top2, top3, top4, top5 = st.columns(5)

    top1.metric(
        "XAU/USD",
        f"{snapshot['mid']:.2f}"
        if snapshot["mid"] is not None
        else "N/A",
    )

    top2.metric(
        "Signal",
        snapshot["signal_label"],
    )

    top3.metric(
        "Trend",
        snapshot["trend"],
    )

    top4.metric(
        "Entry",
        f"{snapshot['entry_price']:.2f}",
    )

    top5.metric(
        "Market",
        snapshot["market_state"],
    )

    st.plotly_chart(
        build_live_proof_chart(
            data,
            snapshot,
        ),
        use_container_width=True,
    )

    st.subheader("System Decision")

    explanation = build_human_explanation(
        snapshot
    )

    if snapshot["signal_label"] == "BUY":
        st.success(explanation)
    else:
        st.info(explanation)

    st.subheader("Trading Levels")

    level1, level2, level3, level4 = st.columns(4)

    level1.metric(
        "Entry",
        f"{snapshot['entry_price']:.2f}",
    )

    level2.metric(
        "Stop Loss",
        (
            f"{snapshot['stop_loss']:.2f}"
            if snapshot["stop_loss"] is not None
            else "N/A"
        ),
    )

    level3.metric(
        "Take Profit",
        (
            f"{snapshot['take_profit']:.2f}"
            if snapshot["take_profit"] is not None
            else "N/A"
        ),
    )

    level4.metric(
        "Risk / Reward",
        (
            f"{snapshot['risk_reward_ratio']:.2f}"
            if snapshot["risk_reward_ratio"] is not None
            else "N/A"
        ),
    )

    st.subheader("Engine Output")

    engine1, engine2, engine3 = st.columns(3)

    engine1.metric(
        "Momentum",
        (
            f"{snapshot['momentum']:.6f}"
            if snapshot["momentum"] is not None
            else "N/A"
        ),
    )

    engine2.metric(
        "Fast MA",
        (
            f"{snapshot['fast_ma']:.2f}"
            if snapshot["fast_ma"] is not None
            else "N/A"
        ),
    )

    engine3.metric(
        "Slow MA",
        (
            f"{snapshot['slow_ma']:.2f}"
            if snapshot["slow_ma"] is not None
            else "N/A"
        ),
    )

    st.subheader("Market Status")

    status1, status2, status3, status4 = st.columns(4)

    status1.metric(
        "Quote Age",
        (
            f"{float(snapshot['quote_age_seconds']):.1f}s"
            if snapshot["quote_age_seconds"] is not None
            else "N/A"
        ),
    )

    status2.metric(
        "Bid",
        (
            f"{float(snapshot['bid']):.2f}"
            if snapshot["bid"] is not None
            else "N/A"
        ),
    )

    status3.metric(
        "Ask",
        (
            f"{float(snapshot['ask']):.2f}"
            if snapshot["ask"] is not None
            else "N/A"
        ),
    )

    status4.metric(
        "Candles",
        snapshot["candle_count"],
    )

    st.caption(
        f"Strategy: {snapshot['strategy']} | "
        f"Timeframe: {snapshot['interval']} | "
        f"Candle timestamp: {snapshot['timestamp']} | "
        f"Source: BiQuote"
    )


def main() -> None:
    fragment = getattr(
        st,
        "fragment",
        None,
    )

    if fragment is not None:

        @fragment(
            run_every=f"{DEFAULT_REFRESH_SECONDS}s"
        )
        def live_view() -> None:
            _render_dashboard()

        live_view()
    else:
        _render_dashboard()


if __name__ == "__main__":
    main()
