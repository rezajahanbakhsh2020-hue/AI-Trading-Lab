from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go


def _number(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_live_visual_performance(
    history: Sequence[Mapping[str, Any]],
) -> go.Figure:
    """
    Build a visual history of live decision quality.

    The chart uses only stability scores explicitly present in history.
    No performance value is inferred when it is unavailable.
    """
    if not isinstance(history, Sequence) or isinstance(
        history, (str, bytes)
    ):
        raise TypeError("history must be a sequence")

    rows: list[tuple[int, Any, float, str, str]] = []

    for index, snapshot in enumerate(history):
        if not isinstance(snapshot, Mapping):
            raise TypeError("each history item must be a mapping")

        score = _number(snapshot.get("stability_score"))
        if score is None:
            continue

        timestamp = snapshot.get("timestamp", index)
        signal = str(
            snapshot.get("signal_label")
            or snapshot.get("signal")
            or "NO TRADE"
        )
        trend = str(
            snapshot.get("trend")
            or "INSUFFICIENT DATA"
        )

        rows.append(
            (index, timestamp, score, signal, trend)
        )

    figure = go.Figure()

    if rows:
        figure.add_trace(
            go.Scatter(
                x=[row[1] for row in rows],
                y=[row[2] for row in rows],
                mode="lines+markers",
                name="Stability Score",
                text=[
                    f"{row[3]} | {row[4]}"
                    for row in rows
                ],
                customdata=[
                    [row[0], row[3], row[4]]
                    for row in rows
                ],
                hovertemplate=(
                    "Stability: %{y:.4f}"
                    "<br>Signal: %{customdata[1]}"
                    "<br>Trend: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title="AI-Trading-Lab — Live Visual Performance",
        height=560,
        margin=dict(l=60, r=30, t=80, b=50),
        xaxis_title="Time",
        yaxis_title="Stability Score",
        hovermode="x unified",
    )

    return figure


def build_live_visual_performance_html(
    history: Sequence[Mapping[str, Any]],
) -> str:
    """Return the live performance visualization as responsive HTML."""
    figure = build_live_visual_performance(history)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
