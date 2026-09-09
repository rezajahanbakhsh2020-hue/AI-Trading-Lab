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
from app_live_proof_dashboard import build_dashboard_chart
from app_live_proof_history import (
    DEFAULT_HISTORY_LIMIT,
    HISTORY_PATH,
    append_history_record,
    build_live_history_record,
    build_history_chart,
    load_history,
    render_history_table,
)
from live_snapshot import save_live_snapshot
from src.evaluation.live_explanation import build_live_explanation


def build_quote_table(
    snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Build the latest live quote table."""

    return pd.DataFrame(
        [
            {
                "Quote": "Bid",
                "Price": snapshot.get("bid"),
            },
            {
                "Quote": "Ask",
                "Price": snapshot.get("ask"),
            },
            {
                "Quote": "Mid",
                "Price": snapshot.get("mid"),
            },
        ]
    )


def build_risk_table(
    snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Build the visual risk-level table."""

    return pd.DataFrame(
        [
            {
                "Level": "Entry",
                "Price": snapshot.get("entry_price"),
            },
            {
                "Level": "Stop Loss",
                "Price": snapshot.get("stop_loss"),
            },
            {
                "Level": "Take Profit",
                "Price": snapshot.get("take_profit"),
            },
        ]
    )


def build_visual_summary(
    snapshot: dict[str, Any],
) -> dict[str, str]:
    """Build compact values for visual dashboard cards."""

    explanation = build_live_explanation(snapshot)

    entry = explanation.get("entry_price")
    stop_loss = explanation.get("stop_loss")
    take_profit = explanation.get("take_profit")

    return {
        "signal": str(
            explanation.get("action", "N/A")
        ),
        "trend": str(
            explanation.get("trend", "N/A")
        ),
        "market": str(
            explanation.get("market_state", "N/A")
        ),
        "entry": (
            f"{float(entry):.2f}"
            if entry is not None
            else "N/A"
        ),
        "stop_loss": (
            f"{float(stop_loss):.2f}"
            if stop_loss is not None
            else "N/A"
        ),
        "take_profit": (
            f"{float(take_profit):.2f}"
            if take_profit is not None
            else "N/A"
        ),
    }


def build_quote_age_label(
    snapshot: dict[str, Any],
) -> str:
    """Return a human-readable quote freshness label."""

    stale = bool(
        snapshot.get("quote_stale", False)
    )
    age = snapshot.get("quote_age_seconds")

    if stale:
        return "STALE"

    if age is None:
        return "UNKNOWN"

    return f"{age}s old"


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | Visual Proof",
        page_icon="📈",
        layout="wide",
    )

    st.title(
        "AI-Trading-Lab — Visual Live Proof"
    )

    st.caption(
        "Real XAU/USD → live candle → signal → trend → "
        "risk levels → visual explanation."
    )

    with st.sidebar:
        st.header("Live Controls")

        interval = st.selectbox(
            "Timeframe",
            SUPPORTED_INTERVALS,
            index=SUPPORTED_INTERVALS.index(
                DEFAULT_INTERVAL
            ),
        )

        limit = st.number_input(
            "Candles",
            min_value=1,
            max_value=MAX_LIMIT,
            value=DEFAULT_LIMIT,
            step=1,
        )

        history_limit = st.number_input(
            "History",
            min_value=1,
            max_value=500,
            value=DEFAULT_HISTORY_LIMIT,
            step=1,
        )

        capture = st.button(
            "Refresh Live Data",
            use_container_width=True,
        )

    if capture:
        try:
            data = fetch_xauusd_ohlc(
                interval=interval,
                limit=int(limit),
            )

            quote = fetch_xauusd_quote()

            snapshot = build_live_history_record(
                data,
                quote,
                interval=interval,
            )

            save_live_snapshot(snapshot)

            history = append_history_record(
                snapshot,
                path=HISTORY_PATH,
                limit=int(history_limit),
            )

            st.session_state[
                "visual_data"
            ] = data

            st.session_state[
                "visual_snapshot"
            ] = snapshot

            st.session_state[
                "visual_history"
            ] = history

            st.success(
                "Live data refreshed successfully."
            )

        except Exception as exc:
            st.error(
                f"Live data refresh failed: {exc}"
            )

    data = st.session_state.get(
        "visual_data"
    )

    snapshot = st.session_state.get(
        "visual_snapshot"
    )

    history = st.session_state.get(
        "visual_history"
    )

    if snapshot is None:
        try:
            history = load_history(
                HISTORY_PATH
            )
        except Exception:
            history = []

        if history:
            snapshot = history[-1]

    if snapshot is None:
        st.info(
            "Press 'Refresh Live Data' to display "
            "the real XAU/USD proof."
        )
        return

    if history is None:
        history = []

    summary = build_visual_summary(
        snapshot
    )

    st.subheader("Live Decision")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Signal",
        summary["signal"],
    )

    col2.metric(
        "Trend",
        summary["trend"],
    )

    col3.metric(
        "Market",
        summary["market"],
    )

    col4.metric(
        "Quote",
        build_quote_age_label(snapshot),
    )

    st.subheader("Price Levels")

    level1, level2, level3 = st.columns(3)

    level1.metric(
        "Entry",
        summary["entry"],
    )

    level2.metric(
        "Stop Loss",
        summary["stop_loss"],
    )

    level3.metric(
        "Take Profit",
        summary["take_profit"],
    )

    st.subheader("Live Explanation")

    explanation = build_live_explanation(
        snapshot
    )

    if summary["signal"] == "BUY":
        st.success(
            explanation["summary"]
        )
    else:
        st.info(
            explanation["summary"]
        )

    for reason in explanation["reasons"]:
        st.markdown(
            f"- {reason}"
        )

    if data is not None and not data.empty:
        st.subheader("XAU/USD Live Chart")

        st.plotly_chart(
            build_dashboard_chart(
                data,
                snapshot,
            ),
            use_container_width=True,
        )

    quote_col, risk_col = st.columns(2)

    with quote_col:
        st.subheader("Latest Quote")

        st.dataframe(
            build_quote_table(snapshot),
            use_container_width=True,
            hide_index=True,
        )

    with risk_col:
        st.subheader("Risk Levels")

        st.dataframe(
            build_risk_table(snapshot),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Latest Candle")

    candle_timestamp = snapshot.get(
        "timestamp",
        "N/A",
    )

    candle_count = snapshot.get(
        "candle_count",
        "N/A",
    )

    st.write(
        f"Timestamp: **{candle_timestamp}**"
    )
    st.write(
        f"Candles loaded: **{candle_count}**"
    )

    if history:
        st.subheader("Live Signal History")

        st.plotly_chart(
            build_history_chart(history),
            use_container_width=True,
        )

        st.dataframe(
            render_history_table(history),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
