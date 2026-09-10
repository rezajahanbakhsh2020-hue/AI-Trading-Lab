from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def _text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def build_live_visual_health(snapshot: Mapping[str, Any]) -> go.Figure:
    """
    Build a compact visual health indicator for the live data pipeline.

    Health is based only on explicit snapshot state:
    - quote_stale
    - market_state
    - candle_count
    - trend
    - signal

    No missing value is inferred as healthy.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    quote_stale = snapshot.get("quote_stale")
    market_state = _text(snapshot.get("market_state"), "UNKNOWN")
    trend = _text(snapshot.get("trend"), "INSUFFICIENT DATA")
    signal = _text(
        snapshot.get("signal_label", snapshot.get("signal")),
        "NO TRADE",
    )

    candle_count = snapshot.get("candle_count")
    if candle_count is None:
        candles = snapshot.get("candles")
        try:
            candle_count = len(candles) if candles is not None else 0
        except TypeError:
            candle_count = 0

    checks = [
        ("QUOTE FRESHNESS", quote_stale is False),
        ("MARKET STATE", market_state != "UNKNOWN"),
        ("CANDLE DATA", candle_count > 0),
        ("TREND STATE", trend != "INSUFFICIENT DATA"),
        ("SIGNAL STATE", signal in {"BUY", "NO TRADE"}),
    ]

    labels = [label for label, _ in checks]
    values = ["OK" if valid else "CHECK" for _, valid in checks]

    overall = "HEALTHY" if all(valid for _, valid in checks) else "CHECK"

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["LIVE VISUAL HEALTH", "STATUS"],
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
        title=f"AI-Trading-Lab — Visual Health: {overall}",
        height=270,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_visual_health_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live visual health panel as responsive HTML."""
    figure = build_live_visual_health(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
