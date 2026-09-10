from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def build_live_visual_snapshot(
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build a visual snapshot of the current live trading state.

    The snapshot exposes only values already supplied by the live pipeline.
    Missing risk targets remain N/A and are never inferred.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    symbol = snapshot.get("symbol") or "N/A"
    interval = snapshot.get("interval") or "N/A"
    signal = (
        snapshot.get("signal_label")
        or snapshot.get("signal")
        or "NO TRADE"
    )
    trend = snapshot.get("trend") or "INSUFFICIENT DATA"
    strategy = snapshot.get("strategy") or "N/A"

    stale = snapshot.get("quote_stale")
    if stale is True:
        quote_status = "STALE"
    elif stale is False:
        quote_status = "FRESH"
    else:
        quote_status = "UNKNOWN"

    values = [
        str(symbol),
        str(interval),
        str(signal),
        str(trend),
        str(strategy),
        quote_status,
    ]

    labels = [
        "SYMBOL",
        "INTERVAL",
        "SIGNAL",
        "TREND",
        "STRATEGY",
        "QUOTE",
    ]

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["LIVE SNAPSHOT", "VALUE"],
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
        title="AI-Trading-Lab — Live Visual Snapshot",
        height=300,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_visual_snapshot_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live visual snapshot as responsive HTML."""
    figure = build_live_visual_snapshot(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
