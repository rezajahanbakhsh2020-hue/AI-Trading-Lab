from __future__ import annotations

from typing import Any, Mapping

import plotly.graph_objects as go


VALID_SIGNALS = {"BUY", "NO TRADE"}
VALID_TRENDS = {
    "UP",
    "DOWN",
    "FLAT",
    "INSUFFICIENT DATA",
}


def _display(value: Any, fallback: str = "N/A") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _price(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def build_live_decision_board(
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build a complete visual board for the current live decision.

    The board covers:
    - signal
    - trend
    - market state
    - strategy
    - entry
    - stop loss
    - TP1 / TP2 / TP3
    - quote freshness
    - candle count
    - timestamp
    - human-readable decision state

    The function is intentionally independent from live-data fetching
    so it can be tested deterministically and reused by dashboards.
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    signal = _display(
        snapshot.get("signal_label"),
        "NO TRADE",
    )

    if signal not in VALID_SIGNALS:
        raise ValueError(
            "signal_label must be BUY or NO TRADE."
        )

    trend = _display(
        snapshot.get("trend"),
        "INSUFFICIENT DATA",
    )

    if trend not in VALID_TRENDS:
        raise ValueError(
            "trend must be a supported trend state."
        )

    symbol = _display(
        snapshot.get("symbol"),
        "XAUUSD",
    )
    interval = _display(
        snapshot.get("interval"),
        "N/A",
    )

    strategy = _display(
        snapshot.get("strategy"),
    )

    market_state = _display(
        snapshot.get("market_state"),
    )

    quote_stale = bool(
        snapshot.get("quote_stale", False)
    )

    quote_status = (
        "STALE"
        if quote_stale
        else "FRESH"
    )

    candle_count = _display(
        snapshot.get("candle_count"),
    )

    timestamp = _display(
        snapshot.get("timestamp"),
    )

    entry = _price(
        snapshot.get("entry_price")
    )
    stop_loss = _price(
        snapshot.get("stop_loss")
    )

    tp1 = snapshot.get("tp1")
    tp2 = snapshot.get("tp2")
    tp3 = snapshot.get("tp3")

    if tp1 is None:
        tp1 = snapshot.get("take_profit")

    tp1_display = _price(tp1)
    tp2_display = _price(tp2)
    tp3_display = _price(tp3)

    if signal == "BUY":
        decision_text = "BUY — TRADE SETUP ACTIVE"
    else:
        decision_text = "NO TRADE — WAIT"

    figure = go.Figure()

    figure.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        xref="paper",
        yref="paper",
        line={"width": 1},
        fillcolor="rgba(245,245,245,0.75)",
    )

    figure.add_annotation(
        x=0.5,
        y=0.93,
        xref="paper",
        yref="paper",
        text=f"{symbol} — LIVE DECISION",
        showarrow=False,
        font={"size": 24},
    )

    figure.add_annotation(
        x=0.5,
        y=0.82,
        xref="paper",
        yref="paper",
        text=decision_text,
        showarrow=False,
        font={"size": 22},
    )

    cards = [
        ("SIGNAL", signal, 0.12),
        ("TREND", trend, 0.37),
        ("MARKET", market_state, 0.62),
        ("QUOTE", quote_status, 0.87),
    ]

    for title, value, x in cards:
        figure.add_annotation(
            x=x,
            y=0.67,
            xref="paper",
            yref="paper",
            text=f"<b>{title}</b><br>{value}",
            showarrow=False,
            align="center",
            font={"size": 16},
        )

    risk_items = [
        ("ENTRY", entry, 0.10),
        ("STOP LOSS", stop_loss, 0.28),
        ("TP1", tp1_display, 0.46),
        ("TP2", tp2_display, 0.64),
        ("TP3", tp3_display, 0.82),
    ]

    for title, value, x in risk_items:
        figure.add_annotation(
            x=x,
            y=0.47,
            xref="paper",
            yref="paper",
            text=f"<b>{title}</b><br>{value}",
            showarrow=False,
            align="center",
            font={"size": 14},
        )

    context_text = (
        f"<b>Strategy:</b> {strategy}"
        f"<br><b>Timeframe:</b> {interval}"
        f"<br><b>Candles:</b> {candle_count}"
        f"<br><b>Timestamp:</b> {timestamp}"
    )

    figure.add_annotation(
        x=0.05,
        y=0.25,
        xref="paper",
        yref="paper",
        text=context_text,
        showarrow=False,
        xanchor="left",
        align="left",
        font={"size": 13},
    )

    status_text = (
        "<b>Decision Status</b>"
        f"<br>Signal: {signal}"
        f"<br>Trend: {trend}"
        f"<br>Market: {market_state}"
        f"<br>Quote: {quote_status}"
    )

    figure.add_annotation(
        x=0.55,
        y=0.25,
        xref="paper",
        yref="paper",
        text=status_text,
        showarrow=False,
        xanchor="left",
        align="left",
        font={"size": 13},
    )

    if signal == "NO TRADE":
        explanation = (
            "No active BUY setup. "
            "The system remains in observation mode."
        )
    elif quote_stale:
        explanation = (
            "BUY detected, but quote freshness "
            "requires attention before execution."
        )
    elif tp1 is None:
        explanation = (
            "BUY detected, but TP1 is unavailable."
        )
    else:
        explanation = (
            "BUY setup contains an entry and "
            "available risk/target information."
        )

    figure.add_annotation(
        x=0.5,
        y=0.09,
        xref="paper",
        yref="paper",
        text=explanation,
        showarrow=False,
        align="center",
        font={"size": 13},
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Decision Board",
        height=700,
        xaxis={
            "visible": False,
            "range": [0, 1],
        },
        yaxis={
            "visible": False,
            "range": [0, 1],
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
        showlegend=False,
    )

    return figure
