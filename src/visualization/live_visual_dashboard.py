from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.visualization.live_history_chart import build_live_history_chart
from src.visualization.live_visual_suite import build_live_visual_suite


def build_live_visual_dashboard(
    candles: pd.DataFrame,
    snapshot: Mapping,
    snapshots: Sequence[Mapping],
) -> str:
    """
    Build the complete visual dashboard for live XAU/USD decisions.

    The dashboard combines:
    - the current live visual suite,
    - the historical decision chart.

    Returns a self-contained responsive HTML document.
    """
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles must be a pandas DataFrame")

    if candles.empty:
        raise ValueError("candles must not be empty")

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if not isinstance(snapshots, Sequence) or isinstance(
        snapshots, (str, bytes, bytearray)
    ):
        raise TypeError("snapshots must be a sequence")

    current_figure = build_live_visual_suite(candles, snapshot)
    history_figure = build_live_history_chart(snapshots)

    current_html = current_figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
    )

    history_html = history_figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
    )

    symbol = str(snapshot.get("symbol", "XAU/USD"))
    interval = str(snapshot.get("interval", "N/A"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-Trading-Lab — Live Visual Dashboard</title>
<style>
html, body {{
    margin: 0;
    padding: 0;
    background: #111827;
    color: #f3f4f6;
    font-family: Arial, sans-serif;
}}

body {{
    min-height: 100vh;
}}

.dashboard {{
    width: 100%;
    max-width: 1600px;
    margin: 0 auto;
    padding: 16px;
    box-sizing: border-box;
}}

.header {{
    padding: 12px 8px 18px;
}}

.header h1 {{
    margin: 0 0 6px;
    font-size: 24px;
}}

.header p {{
    margin: 0;
    color: #9ca3af;
    font-size: 14px;
}}

.panel {{
    background: #1f2937;
    border-radius: 10px;
    margin-bottom: 16px;
    overflow: hidden;
}}

.panel-title {{
    padding: 12px 16px;
    font-size: 16px;
    font-weight: 600;
    border-bottom: 1px solid #374151;
}}

.chart {{
    width: 100%;
}}

@media (max-width: 700px) {{
    .dashboard {{
        padding: 8px;
    }}

    .header h1 {{
        font-size: 20px;
    }}
}}
</style>
</head>
<body>
<div class="dashboard">
    <div class="header">
        <h1>AI-Trading-Lab — Live Visual Dashboard</h1>
        <p>{symbol} · {interval} · Current Decision + Decision History</p>
    </div>

    <section class="panel">
        <div class="panel-title">Current Live Decision</div>
        <div class="chart">{current_html}</div>
    </section>

    <section class="panel">
        <div class="panel-title">Live Decision History</div>
        <div class="chart">{history_html}</div>
    </section>
</div>
</body>
</html>
"""


def save_live_visual_dashboard(
    candles: pd.DataFrame,
    snapshot: Mapping,
    snapshots: Sequence[Mapping],
    output_path: str,
) -> str:
    """Build and save the complete live visual dashboard."""
    html = build_live_visual_dashboard(candles, snapshot, snapshots)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    return output_path
