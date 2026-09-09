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


def _build_decision_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the snapshot format expected by the decision board."""
    result = dict(snapshot)

    if "tp1" not in result:
        result["tp1"] = result.get("take_profit
