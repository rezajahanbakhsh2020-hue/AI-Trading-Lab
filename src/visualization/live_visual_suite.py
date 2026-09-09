from pathlib import Path
from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from src.visualization.live_decision_board import build_live_decision_board
from src.visualization.live_proof_chart import build_live_proof_chart


DEFAULT_OUTPUT_PATH = Path(
    "results/live/live_visual_suite.html"
)


def _value(snapshot, key, default=None):
    if isinstance(snapshot, Mapping):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def _with_tp_levels(snapshot):
    if not isinstance(snapshot, Mapping):
        return snapshot

    result = dict(snapshot)

    take_profit = result.get("take_profit")

    if result.get("tp1") is None:
        result["tp1"] = take_profit

    if result.get("tp2") is None:
        result["tp2"] = None

    if result.get("tp3") is None:
        result["tp3"] = None

    return result


def _summary_text(snapshot):
    symbol = _value(snapshot, "symbol", "XAUUSD")
    interval = _value(snapshot, "interval", "5m")
    signal = _value(snapshot, "signal_label", _value(snapshot, "signal", "UNKNOWN"))
    trend = _value(snapshot, "trend", "UNKNOWN")
    strategy = _value(snapshot, "strategy", "UNKNOWN")
    market_state = _value(snapshot, "market_state", "UNKNOWN")
    quote_stale = _value(snapshot, "quote_stale", False)

    entry = _value(snapshot, "entry_price")
    stop_loss = _value(snapshot, "stop_loss")
    tp1 = _value(snapshot, "tp1", _value(snapshot, "take_profit"))
    tp2 = _value(snapshot, "tp2")
    tp3 = _value(snapshot, "tp3")

    freshness = "STALE" if quote_stale else "LIVE"

    return (
        f"{symbol} | {interval} | "
        f"Signal: {signal} | "
        f"Trend: {trend} | "
        f"Strategy: {strategy} | "
        f"Market: {market_state} | "
        f"Quote: {freshness} | "
        f"Entry: {entry} | "
        f"SL: {stop_loss} | "
        f"TP1: {tp1} | "
        f"TP2: {tp2} | "
        f"TP3: {tp3}"
    )


def build_live_visual_suite(
    candles: pd.DataFrame,
    snapshot,
) -> go.Figure:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles must be a pandas DataFrame")

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    normalized_snapshot = _with_tp_levels(snapshot)

    chart = build_live_proof_chart(
        candles,
        normalized_snapshot,
    )

    decision_board = build_live_decision_board(
        normalized_snapshot
    )

    figure = go.Figure()

    for trace in chart.data:
        figure.add_trace(trace)

    for trace in decision_board.data:
        figure.add_trace(trace)

    figure.update_layout(
        title="AI-Trading-Lab — Complete Live Visual Suite",
        template="plotly_white",
        height=850,
        margin=dict(
            l=40,
            r=40,
            t=100,
            b=40,
        ),
    )

    figure.add_annotation(
        text=_summary_text(normalized_snapshot),
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
        align="left",
        font=dict(size=12),
    )

    return figure


def build_complete_live_visual_output(
    candles: pd.DataFrame,
    snapshot,
    output_path=DEFAULT_OUTPUT_PATH,
) -> str:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles must be a pandas DataFrame")

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

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
        config={"responsive": True},
    )

    symbol = _value(
        snapshot,
        "symbol",
        "XAUUSD",
    )

    interval = _value(
        snapshot,
        "interval",
        "5m",
    )

    html = f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    />
    <title>
        AI-Trading-Lab Complete Live Visual Suite
    </title>
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: Arial, sans-serif;
            background: #ffffff;
        }}

        .header {{
            padding: 14px 18px;
            font-size: 20px;
            font-weight: 700;
        }}

        .meta {{
            padding: 0 18px 12px 18px;
            font-size: 13px;
        }}

        .chart {{
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="header">
        AI-Trading-Lab — Complete Live Visual Suite
    </div>

    <div class="meta">
        Symbol: {symbol} |
        Interval: {interval}
    </div>

    <div class="chart">
        {chart_html}
    </div>
</body>
</html>
"""

    output.write_text(
        html,
        encoding="utf-8",
    )

    return str(output)
