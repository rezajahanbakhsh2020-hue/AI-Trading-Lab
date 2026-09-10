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


def _format(value: Any) -> str:
    number = _number(value)

    if number is None:
        return "N/A"

    return f"{number:.4f}"


def _get(snapshot: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in snapshot and snapshot[key] is not None:
            return snapshot[key]

    return None


def build_live_risk_panel(snapshot: Mapping[str, Any]) -> go.Figure:
    """
    Build a visual risk-level panel for the current live decision.

    The panel displays:
    - signal
    - entry
    - stop loss
    - TP1 / TP2 / TP3
    - risk/reward ratio
    - stop-loss percentage
    - take-profit percentage

    Missing TP2/TP3 values remain N/A. No risk level is invented.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    signal = str(
        snapshot.get(
            "signal_label",
            snapshot.get("signal", "NO TRADE"),
        )
    )

    entry = _get(snapshot, "entry_price", "entry")
    stop_loss = _get(snapshot, "stop_loss")
    tp1 = _get(snapshot, "tp1", "take_profit")
    tp2 = _get(snapshot, "tp2")
    tp3 = _get(snapshot, "tp3")

    risk_reward = _get(
        snapshot,
        "risk_reward_ratio",
        "risk_reward",
    )

    stop_loss_pct = _get(snapshot, "stop_loss_pct")
    take_profit_pct = _get(snapshot, "take_profit_pct")

    labels = [
        "Signal",
        "Entry",
        "Stop Loss",
        "TP1",
        "TP2",
        "TP3",
        "Risk / Reward",
        "Stop Loss %",
        "Take Profit %",
    ]

    values = [
        signal,
        _format(entry),
        _format(stop_loss),
        _format(tp1),
        _format(tp2),
        _format(tp3),
        _format(risk_reward),
        _format(stop_loss_pct),
        _format(take_profit_pct),
    ]

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["RISK LEVEL", "VALUE"],
                    align="left",
                ),
                cells=dict(
                    values=[labels, values],
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Risk Levels",
        height=390,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_risk_panel_html(snapshot: Mapping[str, Any]) -> str:
    """Return the live risk panel as responsive embeddable HTML."""
    figure = build_live_risk_panel(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
