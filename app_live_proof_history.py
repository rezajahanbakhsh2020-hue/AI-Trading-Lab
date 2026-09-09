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
from live_signal import (
    DEFAULT_MOMENTUM_WINDOW,
    build_live_signal_snapshot,
)
from live_snapshot import (
    REQUIRED_SNAPSHOT_FIELDS,
    save_live_snapshot,
)
from live_trend import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    build_live_trend_snapshot,
)


HISTORY_PATH = "results/live/live_proof_history.json"
DEFAULT_HISTORY_LIMIT = 20


def build_live_history_record(
    data: pd.DataFrame,
    quote: dict[str, Any],
    *,
    interval: str = DEFAULT_INTERVAL,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> dict[str, Any]:
    """Build one human-readable live proof history record."""

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

    return {
        "symbol": "XAUUSD",
        "interval": interval,
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
        "candle_count": int(len(data)),
    }


def validate_history_record(
    record: dict[str, Any],
) -> bool:
    """Validate the core fields required by the live snapshot store."""

    if not isinstance(record, dict):
        return False

    return REQUIRED_SNAPSHOT_FIELDS.issubset(
        record.keys()
    )


def load_history(
    path: str = HISTORY_PATH,
) -> list[dict[str, Any]]:
    """Load saved live proof history."""

    from pathlib import Path
    import json

    history_path = Path(path)

    if not history_path.exists():
        return []

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("history must contain a JSON list.")

    return [
        item
        for item in payload
        if isinstance(item, dict)
    ]


def append_history_record(
    record: dict[str, Any],
    path: str = HISTORY_PATH,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Append a live proof record and keep the latest records."""

    if not validate_history_record(record):
        raise ValueError(
            "record does not contain required snapshot fields."
        )

    if limit < 1:
        raise ValueError("limit must be positive.")

    history = load_history(path)

    history.append(dict(record))

    history = history[-limit:]

    from pathlib import Path
    import json

    target = Path(path)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with target.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            history,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return history


def build_history_chart(
    history: list[dict[str, Any]],
) -> go.Figure:
    """Build a visual signal-history chart."""

    if not history:
        return go.Figure()

    frame = pd.DataFrame(history)

    if "timestamp" not in frame.columns:
        raise ValueError(
            "history records must contain timestamp."
        )

    frame["signal_numeric"] = frame[
        "signal"
    ].astype(float)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["signal_numeric"],
            mode="lines+markers",
            name="Signal",
        )
    )

    figure.update_layout(
        title="Live Signal History",
        xaxis_title="Time",
        yaxis_title="Signal",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["NO TRADE", "BUY"],
        ),
        height=350,
    )

    return figure


def render_history_table(
    history: list[dict[str, Any]],
) -> pd.DataFrame:
    """Prepare history for human-readable dashboard display."""

    if not history:
        return pd.DataFrame()

    columns = [
        "timestamp",
        "signal_label",
        "trend",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "mid",
        "market_state",
        "quote_stale",
    ]

    frame = pd.DataFrame(history)

    available = [
        column
        for column in columns
        if column in frame.columns
    ]

    return frame[available].copy()


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab | Live History",
        page_icon="📈",
        layout="wide",
    )

    st.title(
        "AI-Trading-Lab — Live Proof History"
    )

    st.caption(
        "Real XAU/USD observations collected from "
        "the existing live proof engine."
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

    if st.button("Capture Live Proof"):
        try:
            data = fetch_xauusd_ohlc(
                interval=interval,
                limit=int(limit),
            )

            quote = fetch_xauusd_quote()

            record = build_live_history_record(
                data,
                quote,
                interval=interval,
            )

            save_live_snapshot(record)

            history = append_history_record(
                record,
                limit=int(history_limit),
            )

            st.session_state[
                "live_proof_history"
            ] = history

            st.success(
                "Live proof captured and added to history."
            )

        except Exception as exc:
            st.error(
                f"Unable to capture live proof: {exc}"
            )

    history = st.session_state.get(
        "live_proof_history"
    )

    if history is None:
        try:
            history = load_history()
        except Exception as exc:
            st.error(
                f"Unable to load history: {exc}"
            )
            history = []

    if not history:
        st.info(
            "No live proof history is available yet."
        )
        return

    latest = history[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Signal",
        latest.get("signal_label", "N/A"),
    )

    col2.metric(
        "Trend",
        latest.get("trend", "N/A"),
    )

    col3.metric(
        "Market",
        latest.get("market_state", "N/A"),
    )

    col4.metric(
        "Records",
        len(history),
    )

    st.subheader("Signal History")

    st.plotly_chart(
        build_history_chart(history),
        use_container_width=True,
    )

    st.subheader("Latest Live Proof")

    latest_columns = st.columns(5)

    latest_columns[0].metric(
        "Entry",
        (
            f"{float(latest['entry_price']):.2f}"
            if latest.get("entry_price") is not None
            else "N/A"
        ),
    )

    latest_columns[1].metric(
        "SL",
        (
            f"{float(latest['stop_loss']):.2f}"
            if latest.get("stop_loss") is not None
            else "N/A"
        ),
    )

    latest_columns[2].metric(
        "TP",
        (
            f"{float(latest['take_profit']):.2f}"
            if latest.get("take_profit") is not None
            else "N/A"
        ),
    )

    latest_columns[3].metric(
        "RR",
        (
            f"{float(latest['risk_reward_ratio']):.2f}"
            if latest.get("risk_reward_ratio") is not None
            else "N/A"
        ),
    )

    latest_columns[4].metric(
        "Quote Age",
        (
            f"{float(latest['quote_age_seconds']):.1f}s"
            if latest.get("quote_age_seconds") is not None
            else "N/A"
        ),
    )

    st.subheader("History Table")

    st.dataframe(
        render_history_table(history),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
