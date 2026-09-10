from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.live_trade_display import build_live_trade_display
from src.visualization.live_trade_overlay import build_live_trade_overlay


DATA_PATH = "data/raw/xauusd_daily_2025.csv"


def _load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()

    if pd.api.types.is_numeric_dtype(data["timestamp"]):
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            unit="s",
            errors="coerce",
        )
    else:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    if data.empty:
        raise ValueError("No valid market data available.")

    return data.reset_index(drop=True)


def _build_chart(
    data: pd.DataFrame,
    overlay: dict,
) -> go.Figure:
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

    line_definitions = (
        ("Entry", "entry", "dash"),
        ("SL", "stop_loss", "dash"),
        ("TP1", "tp1", "dash"),
        ("TP2", "tp2", "dash"),
        ("TP3", "tp3", "dash"),
    )

    for name, level_key, dash_style in line_definitions:
        price = overlay["levels"][level_key]

        if price is None:
            continue

        figure.add_hline(
            y=price,
            line_dash=dash_style,
            annotation_text=f"{name}: {price:.2f}",
            annotation_position="top right",
        )

    figure.update_layout(
        title="AI-Trading-Lab — Live Trade Display",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
    )

    return figure


def main() -> None:
    st.set_page_config(
        page_title="AI-Trading-Lab Live Trade",
        layout="wide",
    )

    st.title("AI-Trading-Lab — Live Trade Display")

    try:
        data = _load_data()

        display = build_live_trade_display(
            data,
            stable_strategy="momentum",
            stability_score=0.517268,
        )

        overlay = build_live_trade_overlay(
            data,
            display,
        )

    except Exception as exc:
        st.error(f"Live trade display failed: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Decision",
            overlay["decision"],
        )

    with col2:
        st.metric(
            "Trend",
            overlay["trend"],
        )

    with col3:
        st.metric(
            "Stability Score",
            f'{overlay["stability_score"]:.3f}',
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

        figure = _build_chart(
            data,
            overlay,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
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

    st.caption(
        "Strategy: "
        f'{overlay["stable_strategy"]} | '
        f'Timestamp: {overlay["timestamp"]}'
    )


if __name__ == "__main__":
    main()
