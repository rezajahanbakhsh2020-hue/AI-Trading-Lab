from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def _number(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return default


def build_live_visual_metrics(snapshot: Mapping[str, Any]) -> go.Figure:
    """
    Build a visual metrics table from the current live snapshot.

    Only explicitly supplied numeric metrics are displayed.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    labels = [
        "ENTRY",
        "STOP LOSS",
        "TP1",
        "TP2",
        "TP3",
        "RISK/REWARD",
        "STABILITY SCORE",
    ]

    values = [
        _number(snapshot.get("entry_price", snapshot.get("entry"))),
        _number(snapshot.get("stop_loss")),
        _number(
            snapshot.get(
                "tp1",
                snapshot.get("take_profit"),
            )
        ),
        _number(snapshot.get("tp2")),
        _number(snapshot.get("tp3")),
        _number(snapshot.get("risk_reward_ratio")),
        _number(snapshot.get("stability_score")),
    ]

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["LIVE METRICS", "VALUE"],
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
        title="AI-Trading-Lab — Live Visual Metrics",
        height=335,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_visual_metrics_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live visual metrics as responsive HTML."""
    figure = build_live_visual_metrics(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
