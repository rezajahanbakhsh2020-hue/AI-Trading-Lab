from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.visualization.live_visual_report import (
    build_live_visual_report,
)


def build_live_visual_report_index(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """
    Build a lightweight HTML index around the complete live visual report.

    The function provides a clear human-facing entry point for the visual
    report without creating or inferring any new trading data.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if history is not None and not isinstance(history, Sequence):
        raise TypeError("history must be a sequence or None")

    report = build_live_visual_report(
        snapshot,
        history=history,
    )

    symbol = str(snapshot.get("symbol") or "N/A")
    interval = str(snapshot.get("interval") or "N/A")

    marker = "<body>"
    replacement = f"""<body>
<header>
    <h1>AI-Trading-Lab — Live Visual Center</h1>
    <p>Symbol: {symbol} | Interval: {interval}</p>
</header>
<nav aria-label="Visual report navigation">
    <a href="#live-report">Live Report</a>
</nav>
<section id="live-report" aria-label="Complete live visual report">
    {report}
</section>
"""

    if marker not in report:
        raise ValueError("live visual report does not contain body container")

    return report.replace(marker, replacement, 1)


def build_live_visual_report_index_html(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return the live visual center as complete HTML."""
    return build_live_visual_report_index(
        snapshot,
        history=history,
    )
