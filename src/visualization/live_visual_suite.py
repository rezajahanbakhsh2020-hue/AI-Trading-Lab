from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go

from src.visualization.live_decision_board import build_live_decision_board
from src.visualization.live_proof_chart import build_live_proof_chart


DEFAULT_OUTPUT_PATH = Path("results/live/live_visual_suite.html")


def _build_chart_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build the snapshot format expected by the live proof chart."""
    return {
        "symbol": snapshot.get("symbol", "XAUUSD"),
        "interval": snapshot.get("interval", "5m"),
        "signal": int(snapshot.get("signal", 0)),
        "signal_label": snapshot.get("signal_label", "NO TRADE"),
        "trend": snapshot.get("trend", "INSUFFICIENT DATA"),
        "strategy": snapshot.get("strategy"),
        "entry_price": snapshot.get("entry_price"),
        "stop_loss": snapshot.get("stop_loss"),
        "take_profit": snapshot.get("take_profit"),
        "fast_window": snapshot.get("fast_window", 20),
        "slow_window": snapshot.get("slow_window", 50),
        "timestamp": snapshot.get("timestamp"),
    }


def _build_decision_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build the snapshot format expected by the decision board."""
    result = dict(snapshot)

    if "tp1" not in result:
        result["tp1"] = result.get("take_profit")

    if "tp2" not in result:
        result["tp2"] = result.get("take_profit_2")

    if "tp3" not in result:
        result["tp3"] = result.get("take_profit_3")

    return result


def build_live_visual_suite(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
) -> go.Figure:
    """
    Build the complete visual live-trading suite.

    The suite combines:
    - real market candles
    - moving averages
    - entry
    - stop loss
    - TP1 / TP2 / TP3
    - live signal
    - trend
    - market state
    - quote freshness
    - strategy
    - decision status
    """

    if not isinstance(candles, pd.DataFrame):
        raise ValueError("candles must be a pandas DataFrame.")

    if candles.empty:
        raise ValueError("candles must not be empty.")

    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    chart_snapshot = _build_chart_snapshot(snapshot)
    decision_snapshot = _build_decision_snapshot(snapshot)

    chart = build_live_proof_chart(
        candles,
        chart_snapshot,
    )

    decision_board = build_live_decision_board(
        decision_snapshot,
    )

    figure = go.Figure()

    for trace in chart.data:
        figure.add_trace(trace)

    for trace in decision_board.data:
        figure.add_trace(trace)

    signal_label = snapshot.get(
        "signal_label",
        "BUY" if int(snapshot.get("signal", 0)) == 1 else "NO TRADE",
    )
    trend = snapshot.get("trend", "INSUFFICIENT DATA")
    strategy = snapshot.get("strategy", "N/A")
    market_state = snapshot.get("market_state", "UNKNOWN")

    quote_stale = snapshot.get("quote_stale")
    quote_status = (
        "STALE"
        if quote_stale is True
        else "FRESH"
        if quote_stale is False
        else "UNKNOWN"
    )

    entry = snapshot.get("entry_price")
    stop_loss = snapshot.get("stop_loss")
    tp1 = decision_snapshot.get("tp1")
    tp2 = decision_snapshot.get("tp2")
    tp3 = decision_snapshot.get("tp3")

    summary = (
        f"Signal: {signal_label} | "
        f"Trend: {trend} | "
        f"Strategy: {strategy} | "
        f"Market: {market_state} | "
        f"Quote: {quote_status} | "
        f"Entry: {entry} | "
        f"SL: {stop_loss} | "
        f"TP1: {tp1} | "
        f"TP2: {tp2} | "
        f"TP3: {tp3}"
    )

    figure.add_annotation(
        text=summary,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.08,
        showarrow=False,
        align="center",
    )

    figure.update_layout(
        title="XAU/USD Complete Live Visual Suite",
        template="plotly_white",
        height=850,
        margin=dict(t=130, l=40, r=40, b=40),
    )

    return figure


def build_complete_live_visual_output(
    candles: pd.DataFrame,
    snapshot: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """
    Build the complete live visual suite and persist it as HTML.
    """

    figure = build_live_visual_suite(
        candles,
        snapshot,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True)

    return path
