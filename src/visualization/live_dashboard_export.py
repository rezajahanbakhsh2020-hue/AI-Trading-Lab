from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.visualization.live_data_quality_panel import (
    build_live_data_quality_panel_html,
)
from src.visualization.live_decision_summary import (
    build_live_decision_summary_html,
)


def build_live_dashboard_export(
    snapshot: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """
    Build a self-contained HTML export of the core live visual state.

    The export reuses existing visual components and does not infer
    unavailable trading or risk values.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    if history is not None and not isinstance(history, Sequence):
        raise TypeError("history must be a sequence or None")

    decision_html = build_live_decision_summary_html(snapshot)
    quality_html = build_live_data_quality_panel_html(snapshot)

    history_count = len(history) if history is not None else 0

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1">
    <title>AI-Trading-Lab — Live Dashboard Export</title>
</head>
<body>
    <main>
        <h1>AI-Trading-Lab — Live Dashboard Export</h1>
        <p>History snapshots: {history_count}</p>

        <section aria-label="Live decision summary">
            {decision_html}
        </section>

        <section aria-label="Live data quality">
            {quality_html}
        </section>
    </main>
</body>
</html>
""".strip()
