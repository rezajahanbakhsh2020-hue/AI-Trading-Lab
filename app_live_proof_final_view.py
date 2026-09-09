from __future__ import annotations

from typing import Any

import pandas as pd
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
    build_history_chart,
    build_live_history_record,
    load_history,
    render_history_table,
)
from app_live_proof_visual import (
    build_quote_age_label,
    build_quote_table,
    build_risk_table,
    build_visual_summary,
)
from live_snapshot import save_live_snapshot
from src.evaluation.live_explanation import (
    build_live_explanation,
)


def build_latest_candle_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Build a table containing the latest OHLC candle."""

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

    latest = data.iloc[-1]

    return pd.DataFrame(
        [
            {
                "Open": latest["open"],
                "High": latest["high"],
                "Low": latest["low"],
                "Close": latest["close"],
            }
        ]
    )


def build_system_status(
    snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Build the final visual system-status table."""

    return pd.DataFrame(
        [
            {
                "Component": "Signal Engine",
                "Status": (
                    "BUY"
                    if snapshot.get("signal") == 1
                    else "NO TRADE"
                ),
            },
            {
                "Component": "Trend Engine",
                "Status": snapshot.get(
                    "trend",
                    "UNKNOWN",
                ),
            },
            {
                "Component": "Market",
                "Status": snapshot.get(
                    "market_state",
                    "UNKNOWN",
                ),
            },
            {
                "Component": "Quote",
                "Status": (
                    "STALE"
                    if snapshot.get(
                        "quote_stale",
                        False,
                    )
                    else "LIVE"
                ),
            },
            {
                "Component": "Candles",
                "Status": str(
                    snapshot.get(
                        "candle_count",
                        "UNKNOWN",
                    )
                ),
            },
        ]
    )


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | Final Live View",
        page_icon="🟢",
        layout="wide",
    )

    st.title(
        "AI-Trading-Lab — Final Live View"
    )

    st.caption(
        "Real XAU/USD data connected to the existing "
        "signal, trend and risk engines."
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
            "History records",
            min_value=1,
            max_value=500,
            value=DEFAULT_HISTORY_LIMIT,
            step=1,
        )

        refresh = st.button(
            "Refresh XAU/USD",
            use_container_width=True,
        )

    if refresh:
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
                "final_view_data"
            ] = data

            st.session_state[
                "final_view_snapshot"
            ] = snapshot

            st.session_state[
                "final_view_history"
            ] = history

            st.success(
                "Real XAU/USD data loaded."
            )

        except Exception as exc:
            st.error(
                f"Unable to load live data: {exc}"
            )

    data = st.session_state.get(
        "final_view_data"
    )

    snapshot = st.session_state.get(
        "final_view_snapshot"
    )

    history = st.session_state.get(
        "final_view_history"
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
            "Press 'Refresh XAU/USD' to start "
            "the live visual proof."
        )
        return

    if history is None:
        history = []

    summary = build_visual_summary(
        snapshot
    )

    explanation = build_live_explanation(
        snapshot
    )

    st.subheader("System Status")

    st.dataframe(
        build_system_status(snapshot),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Live Decision")

    signal_col, trend_col, market_col, quote_col = (
        st.columns(4)
    )

    signal_col.metric(
        "Signal",
        summary["signal"],
    )

    trend_col.metric(
        "Trend",
        summary["trend"],
    )

    market_col.metric(
        "Market",
        summary["market"],
    )

    quote_col.metric(
        "Quote Age",
        build_quote_age_label(snapshot),
    )

    if summary["signal"] == "BUY":
        st.success(
            explanation["summary"]
        )
    else:
        st.info(
            explanation["summary"]
        )

    st.subheader("XAU/USD Price")

    price_col, bid_col, ask_col, mid_col = (
        st.columns(4)
    )

    price_col.metric(
        "Entry",
        summary["entry"],
    )

    bid_col.metric(
        "Bid",
        (
            f"{float(snapshot['bid']):.3f}"
            if snapshot.get("bid") is not None
            else "N/A"
        ),
    )

    ask_col.metric(
        "Ask",
        (
            f"{float(snapshot['ask']):.3f}"
            if snapshot.get("ask") is not None
            else "N/A"
        ),
    )

    mid_col.metric(
        "Mid",
        (
            f"{float(snapshot['mid']):.3f}"
            if snapshot.get("mid") is not None
            else "N/A"
        ),
    )

    st.subheader("Risk Levels")

    st.dataframe(
        build_risk_table(snapshot),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Live Explanation")

    st.write(
        explanation["human_text"]
    )

    for reason in explanation["reasons"]:
        st.markdown(
            f"- {reason}"
        )

    if data is not None and not data.empty:
        st.subheader("Real XAU/USD Chart")

        st.plotly_chart(
            build_dashboard_chart(
                data,
                snapshot,
            ),
            use_container_width=True,
        )

        st.subheader("Latest Candle")

        st.dataframe(
            build_latest_candle_table(data),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Latest Quote")

    st.dataframe(
        build_quote_table(snapshot),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Proof Timestamp")

    st.write(
        f"Latest candle: **{snapshot.get('timestamp', 'N/A')}**"
    )

    st.write(
        f"Loaded candles: **{snapshot.get('candle_count', 'N/A')}**"
    )

    if history:
        st.subheader("Live History")

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
