from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.visualization.live_trade_chart import build_live_trade_chart


def build_live_trade_dashboard(
    market_data: pd.DataFrame,
    live_result: Mapping[str, Any],
) -> go.Figure:
    """
    Build the final visual dashboard from the existing live pipeline.

    This module only connects existing live output to visualization.
    It does not calculate signals, strategies, risk, or trade levels.
    """

    if not isinstance(live_result, Mapping):
        raise TypeError("live_result must be a mapping.")

    trade_display = live_result.get("trade_display")

    if trade_display is None:
        trade_display = live_result

    if not isinstance(trade_display, Mapping):
        raise TypeError("trade_display must be a mapping.")

    figure = build_live_trade_chart(
        market_data,
        trade_display,
        title="AI-Trading-Lab | Live XAU/USD",
    )

    decision = trade_display.get("decision", "NO TRADE")
    strategy = trade_display.get("stable_strategy", "N/A")
    stability = trade_display.get("stability_score")

    if stability is None:
        stability_text = "N/A"
    else:
        try:
            stability_text = f"{float(stability):.2f}"
        except (TypeError, ValueError):
            stability_text = str(stability)

    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.01,
        text=(
            f"Decision: {decision} | "
            f"Strategy: {strategy} | "
            f"Stability: {stability_text}"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
    )

    return figure
