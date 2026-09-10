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


def build_live_decision_summary(snapshot: Mapping[str, Any]) -> go.Figure:
    """
    Build a compact visual summary of the current live trading decision.

    The summary exposes only values already present in the snapshot.
    No missing risk target or decision value is inferred.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    signal = _text(
        snapshot.get("signal_label", snapshot.get("signal")),
        "NO TRADE",
    )
    trend = _text(snapshot.get("trend"), "INSUFFICIENT DATA")
    strategy = _text(snapshot.get("strategy"))

    stale = snapshot.get("quote_stale")
    if stale is True:
        freshness = "STALE"
    elif stale is False:
        freshness = "FRESH"
    else:
        freshness = "UNKNOWN"

    entry = snapshot.get("entry_price", snapshot.get("entry"))
    stop_loss = snapshot.get("stop_loss")
    tp1 = snapshot.get("tp1", snapshot.get("take_profit"))
    tp2 = snapshot.get("tp2")
    tp3 = snapshot.get("tp3")

    labels = [
        "DECISION",
        "TREND",
        "STRATEGY",
        "QUOTE",
        "ENTRY",
        "STOP LOSS",
        "TP1",
        "TP2",
        "TP3",
    ]

    values = [
        signal,
        trend,
        strategy,
        freshness,
        _number(entry),
        _number(stop_loss),
        _number(tp1),
        _number(tp2),
        _number(tp3),
    ]

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["LIVE DECISION", "VALUE"],
                    align="left",
                ),
                cells=dict(
                    values=[labels, values],
                    align="left",
                    height=32,
                ),
            )
        ]
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Decision Summary",
        height=410,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_decision_summary_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the decision summary as responsive embeddable HTML."""
    figure = build_live_decision_summary(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
