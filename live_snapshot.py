"""Create a persistent, serializable snapshot of the live trading system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SNAPSHOT_PATH = Path("results/live/live_snapshot.json")


def build_live_snapshot(
    system_output: Mapping[str, Any],
    *,
    symbol: str = "XAUUSD",
    interval: str = "5m",
) -> dict[str, Any]:
    """Build a normalized snapshot from the live system output."""

    if not isinstance(system_output, Mapping):
        raise ValueError("system_output must be a mapping.")

    if not symbol:
        raise ValueError("symbol must not be empty.")

    if not interval:
        raise ValueError("interval must not be empty.")

    required_fields = (
        "signal",
        "signal_label",
        "trend",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "timestamp",
    )

    missing = [
        field
        for field in required_fields
        if field not in system_output
    ]

    if missing:
        raise ValueError(
            f"Missing required system output fields: {missing}"
        )

    snapshot = {
        "symbol": str(symbol),
        "interval": str(interval),
        "signal": int(system_output["signal"]),
        "signal_label": str(system_output["signal_label"]),
        "trend": str(system_output["trend"]),
        "strategy": str(system_output.get("strategy", "")),
        "momentum": system_output.get("momentum"),
        "entry_price": system_output["entry_price"],
        "stop_loss": system_output["stop_loss"],
        "take_profit": system_output["take_profit"],
        "risk_reward_ratio": system_output["risk_reward_ratio"],
        "stop_loss_pct": system_output.get("stop_loss_pct"),
        "take_profit_pct": system_output.get("take_profit_pct"),
        "momentum_window": system_output.get("momentum_window"),
        "fast_window": system_output.get("fast_window"),
        "slow_window": system_output.get("slow_window"),
        "timestamp": str(system_output["timestamp"]),
    }

    return snapshot


def save_live_snapshot(
    snapshot: Mapping[str, Any],
    path: str | Path = DEFAULT_SNAPSHOT_PATH,
) -> Path:
    """Save a live snapshot as formatted JSON and return its path."""

    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serialized = json.dumps(
            dict(snapshot),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"snapshot is not JSON serializable: {exc}"
        ) from exc

    output_path.write_text(
        serialized + "\n",
        encoding="utf-8",
    )

    return output_path


def load_live_snapshot(
    path: str | Path = DEFAULT_SNAPSHOT_PATH,
) -> dict[str, Any]:
    """Load a previously saved live snapshot."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {input_path}"
        )

    try:
        payload = json.loads(
            input_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid snapshot JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Snapshot JSON must contain an object.")

    return payload
