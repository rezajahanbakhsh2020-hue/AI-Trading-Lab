"""Live XAU/USD dashboard combining signal, trend, and risk levels."""

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
from live_trend import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    build_live_trend_snapshot,
)
from live_signal import DEFAULT_MOMENTUM_WINDOW


DEFAULT_REFRESH_SECONDS = 15


def build_signal_trend_risk_snapshot(
    data: pd.DataFrame,
    *,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> dict[str, Any]:
    """Combine existing signal, trend, and risk engines."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    signal_risk = build_live_risk_levels(
        data,
        momentum_window=momentum_window,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    trend = build_live_trend_snapshot(
        data,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    return {
        "signal": signal_risk["signal"],
        "signal_label": signal_risk["signal_label"],
        "strategy": signal_risk["strategy"],
        "momentum": signal_risk["momentum"],
        "trend": trend["trend"],
        "fast_ma": trend["fast_ma"],
        "slow_ma": trend["slow_ma"],
        "entry_price": signal_risk["entry_price"],
        "stop_loss": signal_risk["stop_loss"],
        "take_profit": signal_risk["take_profit"],
        "risk_reward_ratio": signal_risk["risk_reward_ratio"],
        "stop_loss_pct": signal_risk["stop_loss_pct"],
        "take_profit_pct": signal_risk["take_profit_pct"],
        "momentum_window": signal_risk["window"],
        "fast_window": trend["fast_window"],
        "slow_window": trend["slow_window"],
        "timestamp": signal_risk["timestamp"],
    }


def build_candlestick_chart(
    data: pd.DataFrame,
    *,
    snapshot: dict[str, Any] | None = None,
) -> go.Figure:
    """Build the live candlestick chart with optional risk levels."""

    required = {"openTime", "open", "high", "low", "close"}

    missing = sorted(required.difference(data.columns))

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

    if snapshot is not None:
        entry_price = snapshot.get("entry_price")
        stop_loss = snapshot.get("stop_loss")
        take_profit = snapshot.get("take_profit")

        if entry_price is not None:
            figure.add_hline(
                y=float(entry_price),
                annotation_text="Entry",
            )

        if stop_loss is not None:
            figure.add_hline(
                y=float(stop_loss),
                annotation_text="SL",
            )

        if take_profit is not None:
            figure.add_hline(
                y=float(take_profit),
                annotation_text="TP",
            )

    figure.update_layout(
        title="XAU/USD Live Market",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )

    return figure


def _render_dashboard() -> None:
    st.set_page_config(
        page_title="XAU/USD Live Signal Trend Risk",
        layout="wide",
    )

    st.title("XAU/USD Live Signal + Trend + Risk")

    st.caption(
        "Real market data connected to the existing signal, "
        "trend, and risk engines."
    )

    with st.sidebar:
        interval = st.selectbox(
            "Timeframe",
            SUPPORTED_INTERVALS,
            index=SUPPORTED_INTERVALS.index(DEFAULT_INTERVAL),
        )

        limit = st.number_input(
            "Candle count",
            min_value=1,
            max_value=MAX_LIMIT,
            value=DEFAULT_LIMIT,
            step=1,
        )

        momentum_window = st.number_input(
            "Momentum window",
            min_value=1,
            value=DEFAULT_MOMENTUM_WINDOW,
            step=1,
        )

        fast_window = st.number_input(
            "Fast MA window",
            min_value=1,
            value=DEFAULT_FAST_WINDOW,
            step=1,
        )

        slow_window = st.number_input(
            "Slow MA window",
            min_value=2,
            value=DEFAULT_SLOW_WINDOW,
            step=1,
        )

        stop_loss_pct = st.number_input(
            "Stop loss %",
            min_value=0.0001,
            value=float(DEFAULT_STOP_LOSS_PCT),
            step=0.001,
            format="%.4f",
        )

        take_profit_pct = st.number_input(
            "Take profit %",
            min_value=0.0001,
            value=float(DEFAULT_TAKE_PROFIT_PCT),
            step=0.001,
            format="%.4f",
        )

    try:
        data = fetch_xauusd_ohlc(
            interval=interval,
            limit=int(limit),
        )

        quote = fetch_xauusd_quote()

        snapshot = build_signal_trend_risk_snapshot(
            data,
            momentum_window=int(momentum_window),
            fast_window=int(fast_window),
            slow_window=int(slow_window),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
        )

    except Exception as exc:
        st.error(f"Unable to build live system output: {exc}")
        return

    if bool(quote.get("stale", False)):
        st.warning("Live quote is currently marked as stale.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Signal",
        snapshot["signal_label"],
    )

    col2.metric(
        "Trend",
        snapshot["trend"],
    )

    col3.metric(
        "Entry Price",
        f"{snapshot['entry_price']:.2f}",
    )

    col4.metric(
        "Market",
        str(quote.get("marketState", "UNKNOWN")),
    )

    st.plotly_chart(
        build_candlestick_chart(
            data,
            snapshot=snapshot,
        ),
        use_container_width=True,
    )

    st.subheader("System Output")

    output_col1, output_col2, output_col3 = st.columns(3)

    output_col1.metric(
        "Momentum",
        f"{snapshot['momentum']:.6f}",
    )

    output_col2.metric(
        "Fast MA",
        (
            f"{snapshot['fast_ma']:.2f}"
            if snapshot["fast_ma"] is not None
            else "N/A"
        ),
    )

    output_col3.metric(
        "Slow MA",
        (
            f"{snapshot['slow_ma']:.2f}"
            if snapshot["slow_ma"] is not None
            else "N/A"
        ),
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risk_col1.metric(
        "Stop Loss",
        (
            f"{snapshot['stop_loss']:.2f}"
            if snapshot["stop_loss"] is not None
            else "N/A"
        ),
    )

    risk_col2.metric(
        "Take Profit",
        (
            f"{snapshot['take_profit']:.2f}"
            if snapshot["take_profit"] is not None
            else "N/A"
        ),
    )

    risk_col3.metric(
        "Risk / Reward",
        (
            f"{snapshot['risk_reward_ratio']:.2f}"
            if snapshot["risk_reward_ratio"] is not None
            else "N/A"
        ),
    )

    if snapshot["signal"] == 1:
        st.success(
            "BUY signal detected. Entry, stop-loss, and take-profit "
            "levels are calculated from the configured risk parameters."
        )
    else:
        st.info(
            "NO TRADE signal. The existing strategy does not expose "
            "an active BUY condition, so SL/TP levels are not issued."
        )

    st.caption(
        f"Signal strategy: {snapshot['strategy']} | "
        f"Momentum window: {snapshot['momentum_window']} | "
        f"Trend windows: {snapshot['fast_window']}/"
        f"{snapshot['slow_window']}"
    )

    st.caption(
        f"Quote age: {quote.get('quoteAgeSeconds', 'N/A')} seconds | "
        f"Source: BiQuote | "
        f"Candle timestamp: {snapshot['timestamp']}"
    )


def main() -> None:
    fragment = getattr(st, "fragment", None)

    if fragment is not None:
        @fragment(run_every=f"{DEFAULT_REFRESH_SECONDS}s")
        def live_view() -> None:
            _render_dashboard()

        live_view()
    else:
        _render_dashboard()


if __name__ == "__main__":
    main()
