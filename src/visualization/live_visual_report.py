from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.visualization.live_dashboard_export import (
    build_live_dashboard_export,
)


def build_live_visual_report(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """
    Build a complete human-readable HTML visual report for the live state.

    The report reuses the existing live dashboard export and adds a compact
    report header with the current symbol, interval, signal, and trend.
    No unavailable trading value is inferred.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if history is not None and not isinstance(history, Sequence):
        raise TypeError("history must be a sequence or None")

    symbol = str(snapshot.get("symbol") or "N/A")
    interval = str(snapshot.get("interval") or "N/A")
    signal = str(
        snapshot.get("signal_label")
        or snapshot.get("signal")
        or "NO TRADE"
    )
    trend = str(snapshot.get("trend") or "INSUFFICIENT DATA")

    dashboard_html = build_live_dashboard_export(
        snapshot,
        history=history,
    )

    marker = "<main>"
    replacement = f"""
<main>
    <header>
        <h1>AI-Trading-Lab — Live Visual Report</h1>
        <p>
            Symbol: {symbol}
            | Interval: {interval}
            | Signal: {signal}
            | Trend: {trend}
        </p>
    </header>
""".strip()

    if marker not in dashboard_html:
        raise ValueError("dashboard export does not contain main container")

    return dashboard_html.replace(marker, replacement, 1)
