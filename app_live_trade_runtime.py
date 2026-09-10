from __future__ import annotations

import pandas as pd
import streamlit as st

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.production_live_bridge import load_production_selection
from src.visualization.live_trade_overlay import build_live_trade_overlay
from app_live_trade_display import _build_chart

SYMBOL = "XAUUSD"
INTERVAL = DEFAULT_INTERVAL
LIMIT = DEFAULT_LIMIT


def _load_runtime_data() -> pd.DataFrame:
    data = fetch_xauusd_ohlc(
        interval=INTERVAL,
        limit=LIMIT,
    )

    required = {
        "openTime",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Missing required live columns: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["openTime"],
        utc=True,
        errors="coerce",
    )

    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    if data.empty:
        raise ValueError("No valid live market data available.")

    return data.reset_index(drop=True)


def _load_stable_selection() -> dict:
    selection = load_production_selection()

    stable_strategy = selection.get("stable_strategy")
    stability_score = selection.get("stability_score")

    if not stable_strategy:
        raise ValueError(
            "Production selection does not contain a stable strategy."
        )

    if stability_score is None:
        raise ValueError(
            "Production selection does not contain a stability score."
        )

    return {
        "stable_strategy": str(stable_strategy),
        "stability_score": float(stability_score),
        "source_path": selection.get("source_path"),
    }


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab Live Runtime",
        layout="wide",
    )

    st.title("AI-Trading-Lab — Live Runtime")

    try:
        data = _load_runtime_data()
        selection = _load_stable_selection()
        quote = fetch_xauusd_quote()

        runtime = build_live_runtime(
            data,
            stable_strategy=selection["stable_strategy"],
            stability_score=selection["stability_score"],
            symbol=SYMBOL,
            interval=INTERVAL,
        )

        overlay = build_live_trade_overlay(
            data,
            runtime.display,
        )

    except Exception as exc:
        st.error(f"Live runtime failed: {exc}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Decision",
            runtime.decision["decision"],
        )

    with col2:
        st.metric(
            "Trend",
            runtime.decision["trend"],
        )

    with col3:
        st.metric(
            "Stability Score",
            f'{runtime.decision["stability_score"]:.3f}',
        )

    with col4:
        st.metric(
            "Signal",
            runtime.decision["signal_label"],
        )

    strategy_col, price_col, market_col = st.columns(3)

    with strategy_col:
        st.metric(
            "Stable Strategy",
            selection["stable_strategy"],
        )

    with price_col:
        st.metric(
            "Live Mid",
            f'{quote["mid"]:.3f}',
        )

    with market_col:
        st.metric(
            "Market",
            str(quote.get("marketState", "unknown")),
        )

    if overlay["decision"] == "BUY":
        levels = overlay["levels"]

        level_columns = st.columns(5)

        labels = (
            ("Entry", levels["entry"]),
            ("SL", levels["stop_loss"]),
            ("TP1", levels["tp1"]),
            ("TP2", levels["tp2"]),
            ("TP3", levels["tp3"]),
        )

        for column, (label, value) in zip(
            level_columns,
            labels,
        ):
            with column:
                st.metric(
                    label,
                    f"{value:.2f}",
                )
    else:
        st.info(
            "NO TRADE — no entry, SL or TP levels are displayed."
        )

    figure = _build_chart(
        data,
        overlay,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    source_path = selection.get("source_path") or "unknown"

    st.caption(
        f'Strategy: {overlay["stable_strategy"]} | '
        f'Symbol: {overlay["symbol"]} | '
        f'Interval: {overlay["interval"]} | '
        f'Candles: {len(data)} | '
        f'Quote stale: {quote.get("stale")} | '
        f'Production source: {source_path}'
    )


if __name__ == "__main__":
    main()
