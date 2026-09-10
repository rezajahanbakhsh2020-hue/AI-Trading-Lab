from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from src.visualization.live_visual_performance import (
    build_live_visual_performance,
)
from src.visualization.live_visual_signal_chart import (
    build_live_visual_signal_chart,
)
from src.visualization.live_visual_status import (
    build_live_visual_status,
)


def build_live_visual_final_view(
    candles: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> go.Figure:
    """
    Build the final practical visual view of the live trading system.

    The view combines:
    - live price/signal/risk chart,
    - current system status,
    - stability-score history.

    No unavailable trading values are inferred.
    """
    if not isinstance(candles, Sequence) or isinstance(
        candles, (str, bytes)
    ):
        raise TypeError("candles must be a sequence")

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if history is not None and (
        not isinstance(history, Sequence)
        or isinstance(history, (str, bytes))
    ):
        raise TypeError("history must be a sequence or None")

    signal_chart = build_live_visual_signal_chart(
        candles,
        snapshot,
    )
    status_panel = build_live_visual_status(snapshot)
    performance_chart = build_live_visual_performance(
        history if history is not None else []
    )

    figure = go.Figure()

    for trace in signal_chart.data:
        figure.add_trace(trace)

    for trace in performance_chart.data:
        figure.add_trace(trace)

    for trace in status_panel.data:
        figure.add_trace(trace)

    for shape in signal_chart.layout.shapes or []:
        figure.add_shape(shape)

    signal = str(
        snapshot.get("signal_label")
        or snapshot.get("signal")
        or "NO TRADE"
    )
    trend = str(
        snapshot.get("trend")
        or "INSUFFICIENT DATA"
    )
    strategy = str(
        snapshot.get("strategy")
        or "N/A"
    )

    figure.update_layout(
        title=(
            "AI-Trading-Lab — Live Visual Final View"
            f" | {signal} | {trend} | {strategy}"
        ),
        height=1050,
        margin=dict(l=60, r=30, t=100, b=50),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="closest",
    )

    return figure


def build_live_visual_final_view_html(
    candles: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return the final live visual view as responsive HTML."""
    figure = build_live_visual_final_view(
        candles,
        snapshot,
        history=history,
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
