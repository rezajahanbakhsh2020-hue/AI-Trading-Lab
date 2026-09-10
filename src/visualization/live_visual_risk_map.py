from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def _number(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_live_visual_risk_map(
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build a visual price-level map for the current live trading decision.

    Only risk levels explicitly available in the snapshot are displayed.
    Missing TP2/TP3 values are not inferred.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    entry = _number(
        snapshot.get("entry_price", snapshot.get("entry"))
    )
    stop_loss = _number(snapshot.get("stop_loss"))
    tp1 = _number(
        snapshot.get("tp1", snapshot.get("take_profit"))
    )
    tp2 = _number(snapshot.get("tp2"))
    tp3 = _number(snapshot.get("tp3"))

    levels = [
        ("Entry", entry),
        ("Stop Loss", stop_loss),
        ("TP1", tp1),
        ("TP2", tp2),
        ("TP3", tp3),
    ]

    available = [
        (label, value)
        for label, value in levels
        if value is not None
    ]

    figure = go.Figure()

    if available:
        x_values = [0, 1]

        for label, value in available:
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=[value, value],
                    mode="lines",
                    name=label,
                    hovertemplate=(
                        f"{label}: %{{y:.4f}}"
                        "<extra></extra>"
                    ),
                )
            )

        prices = [value for _, value in available]
        minimum = min(prices)
        maximum = max(prices)

        if minimum == maximum:
            padding = max(abs(minimum) * 0.001, 1.0)
        else:
            padding = (maximum - minimum) * 0.15

        figure.update_yaxes(
            range=[minimum - padding, maximum + padding]
        )

    figure.update_xaxes(
        visible=False,
        range=[0, 1],
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Visual Risk Map",
        height=520,
        margin=dict(l=70, r=30, t=80, b=40),
        hovermode="y unified",
    )

    return figure


def build_live_visual_risk_map_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live risk map as responsive embeddable HTML."""
    figure = build_live_visual_risk_map(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
