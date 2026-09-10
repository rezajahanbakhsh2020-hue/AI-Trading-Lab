from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go


def _value(
    snapshot: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        value = snapshot.get(key)
        if value is not None:
            return value
    return default


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_live_history_frame(
    snapshots: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if not isinstance(snapshots, Sequence):
        raise TypeError(
            "snapshots must be a sequence"
        )

    rows: list[dict[str, Any]] = []

    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise TypeError(
                "each snapshot must be a mapping"
            )

        rows.append(
            {
                "index": index,
                "timestamp": _value(
                    snapshot,
                    "timestamp",
                    "time",
                    "created_at",
                    default=index,
                ),
                "signal": _value(
                    snapshot,
                    "signal_label",
                    "signal",
                    default="NO TRADE",
                ),
                "trend": _value(
                    snapshot,
                    "trend",
                    default="UNKNOWN",
                ),
                "strategy": _value(
                    snapshot,
                    "strategy",
                    default="unknown",
                ),
                "entry": _float_or_none(
                    _value(
                        snapshot,
                        "entry_price",
                        "entry",
                    )
                ),
                "stop_loss": _float_or_none(
                    _value(
                        snapshot,
                        "stop_loss",
                    )
                ),
                "tp1": _float_or_none(
                    _value(
                        snapshot,
                        "tp1",
                        "take_profit",
                    )
                ),
                "tp2": _float_or_none(
                    _value(
                        snapshot,
                        "tp2",
                    )
                ),
                "tp3": _float_or_none(
                    _value(
                        snapshot,
                        "tp3",
                    )
                ),
                "stability_score": _float_or_none(
                    _value(
                        snapshot,
                        "stability_score",
                    )
                ),
                "quote_stale": bool(
                    _value(
                        snapshot,
                        "quote_stale",
                        default=False,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def build_live_history_chart(
    snapshots: Sequence[Mapping[str, Any]],
) -> go.Figure:
    frame = build_live_history_frame(
        snapshots
    )

    figure = go.Figure()

    if frame.empty:
        figure.update_layout(
            title=(
                "AI-Trading-Lab — "
                "Live Decision History"
            ),
            height=650,
        )
        return figure

    x = frame["index"]

    for column, label in (
        ("entry", "Entry"),
        ("stop_loss", "SL"),
        ("tp1", "TP1"),
        ("tp2", "TP2"),
        ("tp3", "TP3"),
    ):
        if frame[column].notna().any():
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=frame[column],
                    mode="lines+markers",
                    name=label,
                    connectgaps=False,
                )
            )

    stability = frame[
        "stability_score"
    ]

    if stability.notna().any():
        figure.add_trace(
            go.Scatter(
                x=x,
                y=stability,
                mode="lines+markers",
                name="Stability Score",
                yaxis="y2",
            )
        )

    signal_text = frame["signal"].astype(str)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=[None] * len(frame),
            mode="markers",
            name="Signal History",
            text=[
                (
                    f"{signal} | "
                    f"Trend: {trend} | "
                    f"Strategy: {strategy}"
                )
                for signal, trend, strategy in zip(
                    frame["signal"],
                    frame["trend"],
                    frame["strategy"],
                )
            ],
            hovertemplate=(
                "%{text}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=(
            "AI-Trading-Lab — "
            "Live Decision History"
        ),
        template="plotly_white",
        height=700,
        margin=dict(
            l=50,
            r=70,
            t=100,
            b=80,
        ),
        xaxis=dict(
            title="Decision Sequence",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(
            title="Price",
        ),
        yaxis2=dict(
            title="Stability Score",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    stale_count = int(
        frame["quote_stale"].sum()
    )

    figure.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Snapshots:</b> {len(frame)} | "
            f"<b>Stale Quotes:</b> {stale_count}"
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
    )

    return figure
