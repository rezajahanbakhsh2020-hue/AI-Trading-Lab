from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def _text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default

    value = str(value).strip()
    return value if value else default


def _number(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def build_live_status_panel(snapshot: Mapping[str, Any]) -> go.Figure:
    """
    Build a compact visual status panel for the current live decision.

    The panel presents:
    - symbol and interval
    - signal
    - trend
    - strategy
    - market state
    - quote freshness
    - entry
    - stop loss
    - TP1 / TP2 / TP3
    - stability score when available
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    signal = _text(
        snapshot.get("signal_label", snapshot.get("signal")),
        "NO TRADE",
    )
    trend = _text(snapshot.get("trend"), "INSUFFICIENT DATA")
    strategy = _text(snapshot.get("strategy"))
    market_state = _text(snapshot.get("market_state"))

    stale = snapshot.get("quote_stale")
    if stale is True:
        quote_status = "STALE"
    elif stale is False:
        quote_status = "FRESH"
    else:
        quote_status = "UNKNOWN"

    stability = snapshot.get("stability_score")
    if stability is None:
        stability_text = "N/A"
    else:
        try:
            stability_text = f"{float(stability):.4f}"
        except (TypeError, ValueError):
            stability_text = "N/A"

    rows = [
        ("Symbol", _text(snapshot.get("symbol"), "XAU/USD")),
        ("Interval", _text(snapshot.get("interval"))),
        ("Signal", signal),
        ("Trend", trend),
        ("Strategy", strategy),
        ("Market", market_state),
        ("Quote", quote_status),
        ("Entry", _number(snapshot.get("entry_price", snapshot.get("entry")))),
        ("Stop Loss", _number(snapshot.get("stop_loss"))),
        ("TP1", _number(snapshot.get("tp1", snapshot.get("take_profit")))),
        ("TP2", _number(snapshot.get("tp2"))),
        ("TP3", _number(snapshot.get("tp3"))),
        ("Stability", stability_text),
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Table(
            header=dict(
                values=["LIVE STATUS", "VALUE"],
                align="left",
            ),
            cells=dict(
                values=[
                    [row[0] for row in rows],
                    [row[1] for row in rows],
                ],
                align="left",
                height=28,
            ),
        )
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Status",
        height=500,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_status_panel_html(snapshot: Mapping[str, Any]) -> str:
    """Return the live status panel as responsive embeddable HTML."""
    figure = build_live_status_panel(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
