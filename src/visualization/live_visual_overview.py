from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from src.visualization.live_visual_health import (
    build_live_visual_health,
)
from src.visualization.live_visual_metrics_panel import (
    build_live_visual_metrics_panel,
)
from src.visualization.live_visual_status import (
    build_live_visual_status,
)


def build_live_visual_overview(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> go.Figure:
    """
    Build a consolidated visual overview of the live trading state.

    The overview combines status, health, and metrics without creating
    unavailable trading values.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if history is not None and not isinstance(history, Sequence):
        raise TypeError("history must be a sequence or None")

    sources = (
        build_live_visual_status(snapshot),
        build_live_visual_health(snapshot),
        build_live_visual_metrics_panel(snapshot),
    )

    figure = go.Figure()

    for source in sources:
        for trace in source.data:
            figure.add_trace(trace)

    figure.update_layout(
        title="AI-Trading-Lab — Live Visual Overview",
        height=920,
        margin=dict(l=20, r=20, t=80, b=30),
    )

    return figure


def build_live_visual_overview_html(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return the consolidated live visual overview as responsive HTML."""
    figure = build_live_visual_overview(
        snapshot,
        history=history,
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
