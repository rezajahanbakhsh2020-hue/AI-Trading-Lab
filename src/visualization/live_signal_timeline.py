from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go


VALID_SIGNALS = {"BUY", "NO TRADE"}
VALID_TRENDS = {"UP", "DOWN", "FLAT", "INSUFFICIENT DATA"}


def _display(value: Any, default: str) -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _snapshot_signal(snapshot: Mapping[str, Any]) -> str:
    signal = _display(
        snapshot.get("signal_label", snapshot.get("signal")),
        "NO TRADE",
    )

    if signal not in VALID_SIGNALS:
        return "NO TRADE"

    return signal


def _snapshot_trend(snapshot: Mapping[str, Any]) -> str:
    trend = _display(
        snapshot.get("trend"),
        "INSUFFICIENT DATA",
    )

    if trend not in VALID_TRENDS:
        return "INSUFFICIENT DATA"

    return trend


def _snapshot_timestamp(snapshot: Mapping[str, Any], index: int) -> str:
    value = snapshot.get("timestamp")

    if value is None:
        return str(index)

    return str(value)


def build_live_signal_timeline(
    snapshots: Sequence[Mapping[str, Any]],
) -> go.Figure:
    """
    Build a visual timeline of live signal and trend decisions.

    Each snapshot becomes one timeline point. The chart shows:
    - BUY / NO TRADE signal state
    - market trend
    - snapshot timestamp
    - strategy
    - entry / SL / TP1 / TP2 / TP3 when available

    Invalid signal/trend states are normalized to safe visual states
    instead of allowing malformed live data to break the dashboard.
    """
    if not isinstance(snapshots, Sequence) or isinstance(
        snapshots,
        (str, bytes, bytearray),
    ):
        raise TypeError("snapshots must be a sequence")

    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise TypeError("each snapshot must be a mapping")

    figure = go.Figure()

    if not snapshots:
        figure.update_layout(
            title="AI-Trading-Lab — Live Signal Timeline",
            xaxis_title="Snapshot",
            yaxis_title="Signal",
            height=420,
            template="plotly_dark",
        )
        return figure

    x_values = list(range(len(snapshots)))
    signal_values = [
        1 if _snapshot_signal(snapshot) == "BUY" else 0
        for snapshot in snapshots
    ]

    hover_text = []

    for index, snapshot in enumerate(snapshots):
        signal = _snapshot_signal(snapshot)
        trend = _snapshot_trend(snapshot)
        strategy = _display(snapshot.get("strategy"), "N/A")
        timestamp = _snapshot_timestamp(snapshot, index)

        entry = snapshot.get("entry_price", snapshot.get("entry"))
        stop_loss = snapshot.get("stop_loss")
        tp1 = snapshot.get("tp1", snapshot.get("take_profit"))
        tp2 = snapshot.get("tp2")
        tp3 = snapshot.get("tp3")

        hover_text.append(
            "<b>Signal:</b> "
            f"{signal}<br>"
            "<b>Trend:</b> "
            f"{trend}<br>"
            "<b>Strategy:</b> "
            f"{strategy}<br>"
            "<b>Timestamp:</b> "
            f"{timestamp}<br>"
            "<b>Entry:</b> "
            f"{entry if entry is not None else 'N/A'}<br>"
            "<b>SL:</b> "
            f"{stop_loss if stop_loss is not None else 'N/A'}<br>"
            "<b>TP1:</b> "
            f"{tp1 if tp1 is not None else 'N/A'}<br>"
            "<b>TP2:</b> "
            f"{tp2 if tp2 is not None else 'N/A'}<br>"
            "<b>TP3:</b> "
            f"{tp3 if tp3 is not None else 'N/A'}"
        )

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=signal_values,
            mode="lines+markers",
            name="Signal",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    trend_values = [
        {
            "UP": 2,
            "FLAT": 1,
            "DOWN": -1,
            "INSUFFICIENT DATA": 0,
        }[_snapshot_trend(snapshot)]
        for snapshot in snapshots
    ]

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=trend_values,
            mode="lines+markers",
            name="Trend",
            yaxis="y2",
            text=[
                _snapshot_trend(snapshot)
                for snapshot in snapshots
            ],
            hovertemplate="Trend: %{text}<extra></extra>",
        )
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Signal Timeline",
        xaxis=dict(
            title="Snapshot",
            tickmode="linear",
        ),
        yaxis=dict(
            title="Signal",
            tickvals=[0, 1],
            ticktext=["NO TRADE", "BUY"],
            range=[-0.25, 1.25],
        ),
        yaxis2=dict(
            title="Trend",
            overlaying="y",
            side="right",
            tickvals=[-1, 0, 1, 2],
            ticktext=[
                "DOWN",
                "INSUFFICIENT DATA",
                "FLAT / BUY",
                "UP",
            ],
            range=[-1.25, 2.25],
        ),
        hovermode="x unified",
        height=420,
        template="plotly_dark",
        margin=dict(l=70, r=90, t=70, b=60),
    )

    return figure


def build_live_signal_timeline_html(
    snapshots: Sequence[Mapping[str, Any]],
) -> str:
    """Return the signal timeline as embeddable responsive HTML."""
    figure = build_live_signal_timeline(snapshots)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
