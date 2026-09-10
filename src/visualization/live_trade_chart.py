from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go


REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
)


LEVELS = (
    ("entry_price", "Entry"),
    ("stop_loss", "SL"),
    ("tp1", "TP1"),
    ("tp2", "TP2"),
    ("tp3", "TP3"),
)


def _validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Market data is missing required columns: "
            + ", ".join(missing)
        )

    result = data.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains invalid datetime values."
        )

    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if result[list(("open", "high", "low", "close"))].isna().any().any():
        raise ValueError(
            "OHLC columns contain invalid numeric values."
        )

    return (
        result.sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _numeric_level(
    trade_display: Mapping[str, Any],
    key: str,
) -> float | None:
    value = trade_display.get(key)

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must be numeric or None."
        ) from exc

    if not pd.notna(numeric_value):
        return None

    return numeric_value


def _add_level(
    figure: go.Figure,
    *,
    timestamp_start: pd.Timestamp,
    timestamp_end: pd.Timestamp,
    value: float,
    label: str,
    dash: str = "dash",
) -> None:
    figure.add_shape(
        type="line",
        x0=timestamp_start,
        x1=timestamp_end,
        y0=value,
        y1=value,
        line={
            "dash": dash,
            "width": 2,
        },
    )

    figure.add_annotation(
        x=timestamp_end,
        y=value,
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
    )


def build_live_trade_chart(
    data: pd.DataFrame,
    trade_display: Mapping[str, Any],
    *,
    title: str = "XAU/USD Live Trade Setup",
    show_volume: bool = False,
) -> go.Figure:
    """
    Build a live trade chart from market candles and the existing
    live_trade_display output.

    The function is visualization-only.

    It does not:
    - calculate signals;
    - select a strategy;
    - calculate risk;
    - modify trade levels;
    - execute orders.

    Expected trade_display fields may include:
    entry_price, stop_loss, tp1, tp2, tp3,
    decision, stable_strategy, stability_score,
    signal_label and trend.
    """

    frame = _validate_market_data(data)

    if not isinstance(trade_display, Mapping):
        raise TypeError(
            "trade_display must be a mapping."
        )

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="XAU/USD",
        )
    )

    start = frame["timestamp"].iloc[0]
    end = frame["timestamp"].iloc[-1]

    for key, label in LEVELS:
        value = _numeric_level(
            trade_display,
            key,
        )

        if value is None:
            continue

        _add_level(
            figure,
            timestamp_start=start,
            timestamp_end=end,
            value=value,
            label=label,
        )

    decision = str(
        trade_display.get(
            "decision",
            "NO TRADE",
        )
    )

    strategy = str(
        trade_display.get(
            "stable_strategy",
            "N/A",
        )
    )

    stability = trade_display.get(
        "stability_score"
    )

    if stability is None:
        stability_text = "N/A"
    else:
        try:
            stability_text = f"{float(stability):.2f}"
        except (TypeError, ValueError):
            stability_text = str(stability)

    signal_label = str(
        trade_display.get(
            "signal_label",
            "N/A",
        )
    )

    trend = str(
        trade_display.get(
            "trend",
            "N/A",
        )
    )

    figure.update_layout(
        title=(
            f"{title} | "
            f"{decision} | "
            f"{strategy} | "
            f"Stability {stability_text}"
        ),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    if show_volume and "volume" in frame.columns:
        figure.add_trace(
            go.Bar(
                x=frame["timestamp"],
                y=pd.to_numeric(
                    frame["volume"],
                    errors="coerce",
                ),
                name="Volume",
                yaxis="y2",
                opacity=0.25,
            )
        )

        figure.update_layout(
            yaxis2={
                "title": "Volume",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            }
        )

    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        text=(
            f"Signal: {signal_label} | "
            f"Trend: {trend} | "
            f"Strategy: {strategy} | "
            f"Stability: {stability_text}"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
    )

    return figure


def validate_live_trade_chart(
    figure: go.Figure,
) -> bool:
    """
    Validate the basic structure of a generated live trade chart.
    """

    if not isinstance(figure, go.Figure):
        raise TypeError(
            "figure must be a plotly.graph_objects.Figure."
        )

    if len(figure.data) == 0:
        raise ValueError(
            "live trade chart must contain at least one trace."
        )

    return True
