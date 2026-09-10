from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def build_live_proof_chart(
    data: pd.DataFrame,
    snapshot: dict[str, Any],
) -> go.Figure:
    """Build a human-readable visual live-proof chart."""

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
            name="XAUUSD",
        )
    )

    fast_window = int(
        snapshot.get("fast_window", 20)
    )

    slow_window = int(
        snapshot.get("slow_window", 50)
    )

    fast_ma = data["close"].rolling(
        fast_window
    ).mean()

    slow_ma = data["close"].rolling(
        slow_window
    ).mean()

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=fast_ma,
            mode="lines",
            name=f"Fast MA ({fast_window})",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=slow_ma,
            mode="lines",
            name=f"Slow MA ({slow_window})",
        )
    )

    entry = snapshot.get("entry_price")

    if entry is not None:
        figure.add_hline(
            y=float(entry),
            line_dash="dash",
            annotation_text="Entry",
        )

    stop_loss = snapshot.get("stop_loss")

    if stop_loss is not None:
        figure.add_hline(
            y=float(stop_loss),
            line_dash="dash",
            annotation_text="SL",
        )

    take_profit = snapshot.get("take_profit")

    if take_profit is not None:
        figure.add_hline(
            y=float(take_profit),
            line_dash="dash",
            annotation_text="TP",
        )

    tp_levels = (
        ("TP1", snapshot.get("tp1")),
        ("TP2", snapshot.get("tp2")),
        ("TP3", snapshot.get("tp3")),
    )

    for label, level in tp_levels:
        if level is None:
            continue

        figure.add_hline(
            y=float(level),
            line_dash="dash",
            annotation_text=label,
        )

    signal = snapshot.get("signal")

    if signal == 1 and entry is not None:
        figure.add_trace(
            go.Scatter(
                x=[data.index[-1]],
                y=[float(entry)],
                mode="markers",
                marker={"size": 12},
                name="BUY Signal",
            )
        )

    symbol = str(
        snapshot.get("symbol", "XAUUSD")
    ).upper()

    interval = str(
        snapshot.get("interval", "")
    )

    signal_label = snapshot.get(
        "signal_label",
        "NO TRADE",
    )

    trend = snapshot.get(
        "trend",
        "UNKNOWN",
    )

    figure.update_layout(
        title=(
            f"{symbol} Live Proof — "
            f"{signal_label} | Trend: {trend}"
            + (f" | {interval}" if interval else "")
        ),
        xaxis_title="Time",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False,
    )

    return figure
