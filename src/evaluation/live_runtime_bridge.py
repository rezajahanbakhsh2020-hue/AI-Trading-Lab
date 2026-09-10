from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.live_system_bridge import (
    run_live_system_bridge,
)


def build_live_runtime_snapshot(
    live_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Normalize an already-generated live signal/trend/risk result
    into the snapshot consumed by the evaluation bridge.

    No trading value is recalculated.
    """
    if not isinstance(live_result, Mapping):
        raise TypeError("live_result must be a mapping")

    required_fields = (
        "symbol",
        "interval",
        "signal",
        "signal_label",
        "trend",
        "strategy",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "timestamp",
    )

    missing = [
        field
        for field in required_fields
        if field not in live_result
    ]

    if missing:
        raise ValueError(
            "live_result is missing required fields: "
            + ", ".join(missing)
        )

    return dict(live_result)


def run_live_runtime_bridge(
    live_result: Mapping[str, Any],
    history: list[Mapping[str, Any]] | None = None,
    store_path: str | None = None,
) -> dict[str, Any]:
    """
    Connect an existing live signal/trend/risk result to the
    complete decision and end-to-end evaluation chain.
    """
    snapshot = build_live_runtime_snapshot(
        live_result
    )

    bridge_result = run_live_system_bridge(
        snapshot=snapshot,
        history=history,
        store_path=store_path,
    )

    return {
        "live_result": dict(live_result),
        "snapshot": snapshot,
        "bridge": bridge_result,
        "passed": bridge_result["passed"],
    }


def validate_live_runtime_bridge(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live runtime bridge result.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "live_result",
        "snapshot",
        "bridge",
        "passed",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "live runtime bridge is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["live_result"], Mapping):
        raise ValueError("live_result must be a mapping")

    if not isinstance(result["snapshot"], Mapping):
        raise ValueError("snapshot must be a mapping")

    if not isinstance(result["bridge"], Mapping):
        raise ValueError("bridge must be a mapping")

    if not isinstance(result["passed"], bool):
        raise ValueError("passed must be boolean")

    return True
