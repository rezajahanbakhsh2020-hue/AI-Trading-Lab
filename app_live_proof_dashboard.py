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
from app_live_proof_history import (
    DEFAULT_HISTORY_LIMIT,
    HISTORY_PATH,
    append_history_record,
    build_history_chart,
    build_live_history_record,
    load_history,
    render_history_table,
)
from live_snapshot import save_live_snapshot
from src.evaluation.live_explanation import (
    build_live_explanation,
)


def build_dashboard_chart(
    data: pd.DataFrame,
    snapshot: dict[str, Any],
) -> go.Figure:
    """Build the main visual live-proof chart."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing OHLC columns: {sorted(missing)}"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="XAU/USD",
        )
    )

    entry = snapshot.get("entry_price")
    stop_loss = snapshot.get("stop_loss")
    take_profit = snapshot.get("take_profit")

    if entry is not None:
        figure.add_hline(
            y=float(entry),
            line_dash="dash",
            annotation_text="Entry",
        )

    if stop_loss is not None:
        figure.add_hline(
            y=float(stop_loss),
            line_dash="dash",
            annotation_text="SL",
        )

    if take_profit is not None:
        figure.add_hline(
            y=float(take_profit),
            line_dash="dash",
            annotation_text="TP",
        )

    figure.update_layout(
        title="Live XAU/USD Proof",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
    )

    return figure


def build_status_table(
    snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Build a compact table for the latest live status."""

    return pd.DataFrame(
        [
            {
                "Field": "Signal",
                "Value": snapshot.get(
                    "signal_label",
                    "N/A",
                ),
            },
            {
                "Field": "Trend",
                "Value": snapshot.get(
                    "trend",
                    "N/A",
                ),
            },
            {
                "Field": "Market",
                "Value": snapshot.get(
                    "market_state",
                    "N/A",
                ),
            },
            {
                "Field": "Quote Stale",
                "Value": snapshot.get(
                    "quote_stale",
                    False,
                ),
            },
            {
                "Field": "Quote Age",
                "Value": snapshot.get(
                    "quote_age_seconds",
                    "N/A",
                ),
            },
            {
                "Field": "Timestamp",
                "Value": snapshot.get(
                    "timestamp",
                    "N/A",
                ),
            },
        ]
    )


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | Live Proof Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "AI-Trading-Lab — Live XAU/USD Proof"
    )

    st.caption(
        "Real market data → candle → signal → trend → "
        "risk levels → human-readable explanation."
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
            "Candles",
            min_value=1,
            max_value=MAX_LIMIT,
            value=DEFAULT_LIMIT,
            step=1,
        )

        history_limit = st.number_input(
            "History records",
            min_value=1,
            max_value=500,
            value=DEFAULT_HISTORY_LIMIT,
            step=1,
        )

        capture = st.button(
            "Capture Live Proof",
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
                "live_proof_dashboard_data"
            ] = data

            st.session_state[
                "live_proof_dashboard_snapshot"
            ] = snapshot

            st.session_state[
                "live_proof_dashboard_history"
            ] = history

            st.success(
                "Live XAU/USD proof captured successfully."
            )

        except Exception as exc:
            st.error(
                f"Unable to capture live proof: {exc}"
            )

    data = st.session_state.get(
        "live_proof_dashboard_data"
    )

    snapshot = st.session_state.get(
        "live_proof_dashboard_snapshot"
    )

    history = st.session_state.get(
        "live_proof_dashboard_history"
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
            "Press 'Capture Live Proof' to load real "
            "XAU/USD data into the dashboard."
        )
        return

    explanation = build_live_explanation(
        snapshot
    )

    st.subheader("Current Decision")

    decision_col, trend_col, market_col, price_col = (
        st.columns(4)
    )

    decision_col.metric(
        "Signal",
        explanation["action"],
    )

    trend_col.metric(
        "Trend",
        explanation["trend"],
    )

    market_col.metric(
        "Market",
        explanation["market_state"],
    )

    price_col.metric(
        "Entry",
        (
            f"{float(explanation['entry_price']):.2f}"
            if explanation["entry_price"] is not None
            else "N/A"
        ),
    )

    if explanation["action"] == "BUY":
        st.success(
            explanation["summary"]
        )
    else:
        st.info(
            explanation["summary"]
        )

    st.subheader("Human Explanation")

    st.write(
        explanation["human_text"]
    )

    if explanation["reasons"]:
        for reason in explanation["reasons"]:
            st.markdown(
                f"- {reason}"
            )

    st.subheader("Risk Levels")

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risk_col1.metric(
        "Entry",
        (
            f"{float(explanation['entry_price']):.2f}"
            if explanation["entry_price"] is not None
            else "N/A"
        ),
    )

    risk_col2.metric(
        "Stop Loss",
        (
            f"{float(explanation['stop_loss']):.2f}"
            if explanation["stop_loss"] is not None
            else "N/A"
        ),
    )

    risk_col3.metric(
        "Take Profit",
        (
            f"{float(explanation['take_profit']):.2f}"
            if explanation["take_profit"] is not None
            else "N/A"
        ),
    )

    if data is not None and not data.empty:
        st.subheader("Live Chart")

        st.plotly_chart(
            build_dashboard_chart(
                data,
                snapshot,
            ),
            use_container_width=True,
        )

    st.subheader("Market Status")

    st.dataframe(
        build_status_table(snapshot),
        use_container_width=True,
        hide_index=True,
    )

    if history is None:
        history = []

    if history:
        st.subheader("Signal History")

        st.plotly_chart(
            build_history_chart(history),
            use_container_width=True,
        )

        st.subheader("Recent Live Proof")

        st.dataframe(
            render_history_table(history),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
