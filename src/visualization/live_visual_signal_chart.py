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


def _text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def build_live_visual_signal_chart(
    candles: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build a live price chart with the current signal, trend, and risk levels.

    Every supplied candle must contain valid OHLC data.
    Only risk levels explicitly available in the snapshot are displayed.
    """
    if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
        raise TypeError("candles must be a sequence")

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    rows: list[
        tuple[
            Any,
            float,
            float,
            float,
            float,
            float | None,
            float | None,
        ]
    ] = []

    for candle in candles:
        if not isinstance(candle, Mapping):
            raise TypeError("each candle must be a mapping")

        timestamp = candle.get("timestamp", candle.get("openTime"))
        open_price = _number(candle.get("open"))
        high_price = _number(candle.get("high"))
        low_price = _number(candle.get("low"))
        close_price = _number(candle.get("close"))

        if (
            timestamp is None
            or open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
        ):
            raise TypeError(
                "each candle must contain valid timestamp and OHLC values"
            )

        fast_ma = _number(candle.get("fast_ma"))
        slow_ma = _number(candle.get("slow_ma"))

        rows.append(
            (
                timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                fast_ma,
                slow_ma,
            )
        )

    figure = go.Figure()

    if rows:
        timestamps = [row[0] for row in rows]

        figure.add_trace(
            go.Candlestick(
                x=timestamps,
                open=[row[1] for row in rows],
                high=[row[2] for row in rows],
                low=[row[3] for row in rows],
                close=[row[4] for row in rows],
                name="XAU/USD",
            )
        )

        if any(row[5] is not None for row in rows):
            figure.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[row[5] for row in rows],
                    mode="lines",
                    name="Fast MA",
                )
            )

        if any(row[6] is not None for row in rows):
            figure.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[row[6] for row in rows],
                    mode="lines",
                    name="Slow MA",
                )
            )

    levels = [
        (
            "Entry",
            _number(snapshot.get("entry_price", snapshot.get("entry"))),
        ),
        ("Stop Loss", _number(snapshot.get("stop_loss"))),
        (
            "TP1",
            _number(snapshot.get("tp1", snapshot.get("take_profit"))),
        ),
        ("TP2", _number(snapshot.get("tp2"))),
        ("TP3", _number(snapshot.get("tp3"))),
    ]

    for label, value in levels:
        if value is None:
            continue

        figure.add_hline(
            y=value,
            line_dash="dash",
            annotation_text=f"{label}: {value:.4f}",
            annotation_position="top left",
        )

    signal = _text(
        snapshot.get("signal_label", snapshot.get("signal")),
        "NO TRADE",
    )
    trend = _text(snapshot.get("trend"), "INSUFFICIENT DATA")
    strategy = _text(snapshot.get("strategy"))

    figure.update_layout(
        title=(
            "AI-Trading-Lab — Live Signal Chart"
            f" | {signal} | {trend} | {strategy}"
        ),
        height=700,
        margin=dict(l=60, r=30, t=90, b=50),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )

    return figure


def build_live_visual_signal_chart_html(
    candles: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> str:
    """Return the live signal chart as responsive embeddable HTML."""
    figure = build_live_visual_signal_chart(candles, snapshot)

    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )
