from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def _text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def build_live_visual_status(snapshot: Mapping[str, Any]) -> go.Figure:
    """
    Build a compact visual status indicator for the live system.

    The status is derived only from explicit live snapshot values.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    quote_stale = snapshot.get("quote_stale")
    candle_count = snapshot.get("candle_count")

    if candle_count is None:
        candles = snapshot.get("candles")
        try:
            candle_count = len(candles) if candles is not None else 0
        except TypeError:
            candle_count = 0

    signal = _text(
        snapshot.get("signal_label", snapshot.get("signal")),
        "NO TRADE",
    )
    trend = _text(snapshot.get("trend"), "INSUFFICIENT DATA")

    if quote_stale is True:
        quote_status = "STALE"
    elif quote_stale is False:
        quote_status = "FRESH"
    else:
        quote_status = "UNKNOWN"

    if quote_status == "FRESH" and candle_count > 0:
        data_status = "READY"
    else:
        data_status = "CHECK"

    decision_status = (
        "ACTIONABLE"
        if signal == "BUY" and trend == "UP"
        else "MONITOR"
    )

    labels = [
        "DATA",
        "QUOTE",
        "SIGNAL",
        "TREND",
        "DECISION",
    ]

    values = [
        data_status,
        quote_status,
        signal,
        trend,
        decision_status,
    ]

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["LIVE VISUAL STATUS", "STATE"],
                    align="left",
                ),
                cells=dict(
                    values=[labels, values],
                    align="left",
                    height=34,
                ),
            )
        ]
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Visual Status",
        height=285,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_visual_status_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live visual status as responsive HTML."""
    figure = build_live_visual_status(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
