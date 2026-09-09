"""Build a human-readable visual report from the persisted live proof snapshot."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

from live_snapshot import (
    DEFAULT_SNAPSHOT_PATH,
    load_live_snapshot,
)


DEFAULT_REPORT_PATH = Path("results/live/live_proof.html")


def _display(value: Any) -> str:
    """Return a safe human-readable representation."""
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def _escape(value: Any) -> str:
    """Escape a value before inserting it into HTML."""
    return html.escape(_display(value))


def build_live_proof_html(snapshot: Mapping[str, Any]) -> str:
    """Build an HTML visual report from a persisted live snapshot."""

    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    required_fields = (
        "symbol",
        "interval",
        "signal_label",
        "trend",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "market_state",
        "quote_stale",
        "quote_age_seconds",
        "timestamp",
        "candle_count",
    )

    missing = [
        field
        for field in required_fields
        if field not in snapshot
    ]

    if missing:
        raise ValueError(
            f"Missing required snapshot fields: {missing}"
        )

    signal = str(snapshot["signal_label"])
    trend = str(snapshot["trend"])
    stale = bool(snapshot["quote_stale"])

    signal_class = (
        "positive"
        if signal == "BUY"
        else "neutral"
    )

    trend_class = (
        "positive"
        if trend == "UP"
        else "negative"
        if trend == "DOWN"
        else "neutral"
    )

    stale_class = "negative" if stale else "positive"
    stale_text = "STALE" if stale else "LIVE"

    title = (
        f"{_escape(snapshot['symbol'])} "
        f"Live Proof"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 24px;
    background: #f4f6f8;
    color: #1f2937;
}}

.container {{
    max-width: 1000px;
    margin: 0 auto;
}}

h1 {{
    margin-bottom: 6px;
}}

.subtitle {{
    color: #6b7280;
    margin-bottom: 24px;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}}

.card {{
    background: white;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}}

.label {{
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 7px;
}}

.value {{
    font-size: 24px;
    font-weight: 700;
}}

.positive {{
    color: #15803d;
}}

.negative {{
    color: #b91c1c;
}}

.neutral {{
    color: #374151;
}}

.section {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}}

.row {{
    display: flex;
    justify-content: space-between;
    gap: 20px;
    padding: 9px 0;
    border-bottom: 1px solid #e5e7eb;
}}

.row:last-child {{
    border-bottom: none;
}}

.key {{
    color: #6b7280;
}}

.value-small {{
    font-weight: 600;
    text-align: right;
}}

.footer {{
    color: #6b7280;
    font-size: 12px;
    margin-top: 20px;
}}
</style>
</head>

<body>
<div class="container">

<h1>{title}</h1>
<div class="subtitle">
    Persisted end-to-end live system proof
</div>

<div class="grid">

    <div class="card">
        <div class="label">Signal</div>
        <div class="value {signal_class}">
            {_escape(signal)}
        </div>
    </div>

    <div class="card">
        <div class="label">Trend</div>
        <div class="value {trend_class}">
            {_escape(trend)}
        </div>
    </div>

    <div class="card">
        <div class="label">Entry Price</div>
        <div class="value">
            {_escape(snapshot["entry_price"])}
        </div>
    </div>

    <div class="card">
        <div class="label">Risk / Reward</div>
        <div class="value">
            {_escape(snapshot["risk_reward_ratio"])}
        </div>
    </div>

</div>

<div class="section">
    <h2>Risk Levels</h2>

    <div class="row">
        <span class="key">Entry</span>
        <span class="value-small">
            {_escape(snapshot["entry_price"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Stop Loss</span>
        <span class="value-small">
            {_escape(snapshot["stop_loss"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Take Profit</span>
        <span class="value-small">
            {_escape(snapshot["take_profit"])}
        </span>
    </div>

</div>

<div class="section">
    <h2>Market Proof</h2>

    <div class="row">
        <span class="key">Symbol</span>
        <span class="value-small">
            {_escape(snapshot["symbol"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Interval</span>
        <span class="value-small">
            {_escape(snapshot["interval"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Market State</span>
        <span class="value-small">
            {_escape(snapshot["market_state"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Quote Status</span>
        <span class="value-small {stale_class}">
            {_escape(stale_text)}
        </span>
    </div>

    <div class="row">
        <span class="key">Quote Age (seconds)</span>
        <span class="value-small">
            {_escape(snapshot["quote_age_seconds"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Candle Count</span>
        <span class="value-small">
            {_escape(snapshot["candle_count"])}
        </span>
    </div>

    <div class="row">
        <span class="key">Last Candle Timestamp</span>
        <span class="value-small">
            {_escape(snapshot["timestamp"])}
        </span>
    </div>

</div>

<div class="section">
    <h2>Engine Output</h2>

    <div class="row">
        <span class="key">Strategy</span>
        <span class="value-small">
            {_escape(snapshot.get("strategy"))}
        </span>
    </div>

    <div class="row">
        <span class="key">Momentum</span>
        <span class="value-small">
            {_escape(snapshot.get("momentum"))}
        </span>
    </div>

    <div class="row">
        <span class="key">Fast Window</span>
        <span class="value-small">
            {_escape(snapshot.get("fast_window"))}
        </span>
    </div>

    <div class="row">
        <span class="key">Slow Window</span>
        <span class="value-small">
            {_escape(snapshot.get("slow_window"))}
        </span>
    </div>

    <div class="row">
        <span class="key">Momentum Window</span>
        <span class="value-small">
            {_escape(snapshot.get("momentum_window"))}
        </span>
    </div>

</div>

<div class="footer">
    Generated from the persisted live proof snapshot.
    This report does not recalculate trading signals or risk levels.
</div>

</div>
</body>
</html>
"""


def save_live_proof_report(
    snapshot: Mapping[str, Any],
    path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Build and save the visual live proof report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = build_live_proof_html(snapshot)

    output_path.write_text(
        content,
        encoding="utf-8",
    )

    return output_path


def build_report_from_snapshot(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Load the persisted snapshot and create its visual report."""

    snapshot = load_live_snapshot(snapshot_path)

    return save_live_proof_report(
        snapshot,
        report_path,
    )


def main() -> None:
    report_path = build_report_from_snapshot()

    print("=== XAU/USD LIVE PROOF REPORT ===")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
