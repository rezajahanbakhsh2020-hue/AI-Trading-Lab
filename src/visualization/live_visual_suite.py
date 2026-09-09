from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from src.visualization.live_decision_board import (
    build_live_decision_board,
)
from src.visualization.live_proof_chart import (
    build_live_proof_chart,
)


DEFAULT_OUTPUT_PATH = Path(
    "results/live/live_visual_suite.html"
)


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


def _format_price(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(snapshot)

    if result.get("tp1") is None:
        result["tp1"] = result.get("take_profit")

    if result.get("take_profit") is None:
        result["take_profit"] = result.get("tp1")

    return result


def _decision_text(
    snapshot: Mapping[str, Any],
) -> str:
    signal_label = _value(
        snapshot,
        "signal_label",
        default="NO TRADE",
    )

    if signal_label == "BUY":
        return "BUY — TRADE SETUP ACTIVE"

    return "NO TRADE — WAIT"


def build_live_visual_suite(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
) -> go.Figure:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError(
            "candles must be a pandas DataFrame"
        )

    if candles.empty:
        raise ValueError(
            "candles must not be empty"
        )

    if not isinstance(snapshot, Mapping):
        raise TypeError(
            "snapshot must be a mapping"
        )

    normalized = _normalize_snapshot(snapshot)

    proof_chart = build_live_proof_chart(
        candles,
        normalized,
    )

    decision_board = build_live_decision_board(
        normalized,
    )

    figure = go.Figure()

    for trace in proof_chart.data:
        figure.add_trace(trace)

    for trace in decision_board.data:
        figure.add_trace(trace)

    symbol = _value(
        normalized,
        "symbol",
        default="XAUUSD",
    )

    interval = _value(
        normalized,
        "interval",
        default="5m",
    )

    signal_label = _value(
        normalized,
        "signal_label",
        default="NO TRADE",
    )

    trend = _value(
        normalized,
        "trend",
        default="INSUFFICIENT DATA",
    )

    strategy = _value(
        normalized,
        "strategy",
        default="unknown",
    )

    market_state = _value(
        normalized,
        "market_state",
        default="UNKNOWN",
    )

    entry = _format_price(
        _value(
            normalized,
            "entry_price",
            "entry",
        )
    )

    stop_loss = _format_price(
        _value(
            normalized,
            "stop_loss",
        )
    )

    tp1 = _format_price(
        _value(
            normalized,
            "tp1",
            "take_profit",
        )
    )

    tp2 = _format_price(
        _value(
            normalized,
            "tp2",
        )
    )

    tp3 = _format_price(
        _value(
            normalized,
            "tp3",
        )
    )

    decision = _decision_text(normalized)

    figure.update_layout(
        title=(
            "AI-Trading-Lab — Live Visual Suite"
            f" | {symbol} | {interval}"
        ),
        template="plotly_white",
        height=900,
        margin=dict(
            l=40,
            r=40,
            t=190,
            b=80,
        ),
        annotations=[
            dict(
                x=0.5,
                y=1.13,
                xref="paper",
                yref="paper",
                text=(
                    f"<b>{symbol}</b> | "
                    f"{interval} | "
                    f"Signal: <b>{signal_label}</b> | "
                    f"Trend: <b>{trend}</b>"
                ),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=16),
            ),
            dict(
                x=0.5,
                y=1.075,
                xref="paper",
                yref="paper",
                text=(
                    f"<b>{decision}</b> | "
                    f"Strategy: {strategy} | "
                    f"Market: {market_state}"
                ),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=14),
            ),
            dict(
                x=0.5,
                y=1.025,
                xref="paper",
                yref="paper",
                text=(
                    f"Entry: <b>{entry}</b> | "
                    f"SL: <b>{stop_loss}</b> | "
                    f"TP1: <b>{tp1}</b> | "
                    f"TP2: <b>{tp2}</b> | "
                    f"TP3: <b>{tp3}</b>"
                ),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=13),
            ),
        ],
    )

    return figure


def build_complete_live_visual_output(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> str:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError(
            "candles must be a pandas DataFrame"
        )

    if not isinstance(snapshot, Mapping):
        raise TypeError(
            "snapshot must be a mapping"
        )

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = build_live_visual_suite(
        candles,
        snapshot,
    )

    chart_html = to_html(
        figure,
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
        },
    )

    symbol = _value(
        snapshot,
        "symbol",
        default="XAUUSD",
    )

    interval = _value(
        snapshot,
        "interval",
        default="5m",
    )

    signal_label = _value(
        snapshot,
        "signal_label",
        default="NO TRADE",
    )

    trend = _value(
        snapshot,
        "trend",
        default="INSUFFICIENT DATA",
    )

    decision = _decision_text(snapshot)

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '    <meta charset="utf-8">\n'
        '    <meta name="viewport" '
        'content="width=device-width, initial-scale=1">\n'
        "    <title>"
        "AI-Trading-Lab — Live Visual Suite"
        "</title>\n"
        "</head>\n"
        "<body>\n"
        "    <main>\n"
        "        <h1>"
        "AI-Trading-Lab — Live Visual Suite"
        "</h1>\n"
        "        <h2>"
        "Complete Decision Board"
        "</h2>\n"
        "        <p>\n"
        "            <strong>Symbol:</strong> "
        f"{symbol}"
        "            | "
        "            <strong>Interval:</strong> "
        f"{interval}"
        "        </p>\n"
        "        <p>\n"
        "            <strong>Signal:</strong> "
        f"{signal_label}"
        "            | "
        "            <strong>Trend:</strong> "
        f"{trend}"
        "            | "
        "            <strong>Decision:</strong> "
        f"{decision}"
        "        </p>\n"
        f"        {chart_html}\n"
        "    </main>\n"
        "</body>\n"
        "</html>\n"
    )

    output.write_text(
        html,
        encoding="utf-8",
    )

    return str(output)
