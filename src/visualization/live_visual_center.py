from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from src.visualization.live_data_quality_panel import (
    build_live_data_quality_panel,
)
from src.visualization.live_decision_summary import (
    build_live_decision_summary,
)
from src.visualization.live_visual_health import (
    build_live_visual_health,
)


def build_live_visual_center(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> go.Figure:
    """
    Build a single visual center containing the current live state.

    The figure combines decision, data quality, and visual health information.
    Historical data is accepted for interface compatibility but no values are
    inferred from it.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if history is not None and not isinstance(history, Sequence):
        raise TypeError("history must be a sequence or None")

    decision = build_live_decision_summary(snapshot)
    quality = build_live_data_quality_panel(snapshot)
    health = build_live_visual_health(snapshot)

    figure = go.Figure()

    for source in (decision, quality, health):
        for trace in source.data:
            figure.add_trace(trace)

    figure.update_layout(
        title="AI-Trading-Lab — Live Visual Center",
        height=980,
        margin=dict(l=20, r=20, t=80, b=30),
    )

    return figure


def build_live_visual_center_html(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return the complete live visual center as responsive HTML."""
    figure = build_live_visual_center(
        snapshot,
        history=history,
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
