from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import plotly.graph_objects as go


def _text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default

    value = str(value).strip()
    return value if value else default


def _number(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default

    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return default


def build_live_data_quality_panel(
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build a visual panel describing the quality and freshness of live data.

    Only values already present in the snapshot are displayed.
    No quality score is invented when the required inputs are absent.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    symbol = _text(snapshot.get("symbol"))
    interval = _text(snapshot.get("interval"))

    stale = snapshot.get("quote_stale")
    if stale is True:
        quote_status = "STALE"
    elif stale is False:
        quote_status = "FRESH"
    else:
        quote_status = "UNKNOWN"

    market_state = _text(snapshot.get("market_state"))
    quote_age = _number(snapshot.get("quote_age_seconds"))

    candle_count = snapshot.get("candle_count")
    if candle_count is None:
        candles = snapshot.get("candles")
        try:
            candle_count = len(candles) if candles is not None else None
        except TypeError:
            candle_count = None

    labels = [
        "SYMBOL",
        "INTERVAL",
        "MARKET",
        "QUOTE",
        "QUOTE AGE (SEC)",
        "CANDLES",
    ]

    values = [
        symbol,
        interval,
        market_state,
        quote_status,
        quote_age,
        _number(candle_count),
    ]

    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["LIVE DATA QUALITY", "VALUE"],
                    align="left",
                ),
                cells=dict(
                    values=[labels, values],
                    align="left",
                    height=32,
                ),
            )
        ]
    )

    figure.update_layout(
        title="AI-Trading-Lab — Live Data Quality",
        height=300,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return figure


def build_live_data_quality_panel_html(
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live data quality panel as responsive HTML."""
    figure = build_live_data_quality_panel(snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
