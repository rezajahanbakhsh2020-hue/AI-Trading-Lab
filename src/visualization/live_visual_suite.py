from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from src.visualization.live_decision_board import (
    build_live_decision_board,
)
from src.visualization.live_proof_chart import (
    build_live_proof_chart,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/live/live_visual_suite.html"
)


def build_live_visual_suite(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build the complete visual live-trading suite.

    The suite combines:
    - real market candles
    - moving averages
    - entry
    - stop loss
    - TP1 / TP2 / TP3
    - live signal
    - trend
    - market state
    - quote freshness
    - strategy
    - decision status
    """

    if not isinstance(candles, pd.DataFrame):
        raise ValueError("candles must be a pandas DataFrame.")

    if candles.empty:
        raise ValueError("candles must not be empty.")

    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    chart = build_live_proof_chart(
        candles=candles,
        signal=int(snapshot.get("signal", 0)),
        entry_price=snapshot.get("entry_price"),
        stop_loss=snapshot.get("stop_loss"),
        take_profit=snapshot.get("take_profit"),
    )

    board = build_live_decision_board(snapshot)

    figure = go.Figure()

    for trace in chart.data:
        figure.add_trace(trace)

    figure.update_layout(
        title="AI-Trading-Lab — Complete Live Visual Suite",
        height=850,
        margin={
            "l": 50,
            "r": 50,
            "t": 90,
            "b": 50,
        },
    )

    signal = str(
        snapshot.get(
            "signal_label",
            "NO TRADE",
        )
    )

    trend = str(
        snapshot.get(
            "trend",
            "INSUFFICIENT DATA",
        )
    )

    strategy = str(
        snapshot.get(
            "strategy",
            "N/A",
        )
    )

    market_state = str(
        snapshot.get(
            "market_state",
            "N/A",
        )
    )

    quote_stale = bool(
        snapshot.get(
            "quote_stale",
            False,
        )
    )

    entry = snapshot.get("entry_price")
    stop_loss = snapshot.get("stop_loss")
    tp1 = snapshot.get("tp1")
    tp2 = snapshot.get("tp2")
    tp3 = snapshot.get("tp3")

    if tp1 is None:
        tp1 = snapshot.get("take_profit")

    def fmt(value: Any) -> str:
        if value is None:
            return "N/A"

        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    status = (
        "BUY — ACTIVE SETUP"
        if signal == "BUY"
        else "NO TRADE — WAIT"
    )

    quote_status = (
        "STALE"
        if quote_stale
        else "FRESH"
    )

    summary = (
        f"<b>{status}</b>"
        f"<br>Signal: {signal}"
        f"<br>Trend: {trend}"
        f"<br>Strategy: {strategy}"
        f"<br>Market: {market_state}"
        f"<br>Quote: {quote_status}"
        f"<br>Entry: {fmt(entry)}"
        f"<br>SL: {fmt(stop_loss)}"
        f"<br>TP1: {fmt(tp1)}"
        f"<br>TP2: {fmt(tp2)}"
        f"<br>TP3: {fmt(tp3)}"
    )

    figure.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text=summary,
        showarrow=False,
        align="center",
        font={"size": 15},
    )

    return figure


def build_complete_live_visual_output(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> str:
    """
    Build and save the complete live visual suite as HTML.
    """

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = build_live_visual_suite(
        candles=candles,
        snapshot=snapshot,
    )

    board = build_live_decision_board(snapshot)

    chart_html = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )

    board_html = board.to_html(
        full_html=False,
        include_plotlyjs=False,
    )

    symbol = str(
        snapshot.get(
            "symbol",
            "XAUUSD",
        )
    )

    interval = str(
        snapshot.get(
            "interval",
            "N/A",
        )
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>
AI-Trading-Lab Complete Live Visual Suite
</title>
</head>

<body>

<h1>
AI-Trading-Lab — {symbol} Live Visual Suite
</h1>

<h2>
Market Chart — {interval}
</h2>

{chart_html}

<hr>

<h2>
Complete Decision Board
</h2>

{board_html}

</body>
</html>
"""

    path.write_text(
        html,
        encoding="utf-8",
    )

    return str(path)
