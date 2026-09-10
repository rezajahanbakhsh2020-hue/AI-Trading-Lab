from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go

from src.visualization.live_visual_metrics import (
    build_live_visual_metrics,
)


def build_live_visual_metrics_panel(
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build a titled visual panel containing the live trading metrics.

    The panel reuses the existing metrics component and does not infer
    unavailable values.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    source = build_live_visual_metrics(snapshot)

    figure = go.Figure()

    for trace in source.data:
        figure.add_trace(trace)

    figure.update_layout(
        title="AI-Trading-Lab — Live Metrics Panel",
        height=380,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_visual_metrics_panel_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live metrics panel as responsive HTML."""
    figure = build_live_visual_metrics_panel(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
