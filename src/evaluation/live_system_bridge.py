from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.live_decision_pipeline import (
    process_live_decision,
)
from src.evaluation.live_end_to_end import (
    run_live_end_to_end,
)


def run_live_system_bridge(
    snapshot: Mapping[str, Any],
    history: list[Mapping[str, Any]] | None = None,
    store_path: str | None = None,
) -> dict[str, Any]:
    """
    Connect one live market snapshot to the complete decision chain.

    Flow:
        live snapshot
        -> decision record
        -> decision pipeline
        -> final decision
        -> end-to-end proof

    This function does not fetch market data, recalculate trading
    signals, recalculate risk levels, or execute orders.
    """
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")

    pipeline_result = process_live_decision(
        snapshot=snapshot,
        history=history,
        store_path=store_path,
    )

    end_to_end_result = run_live_end_to_end(
        pipeline_result
    )

    return {
        "snapshot": dict(snapshot),
        "pipeline": pipeline_result,
        "end_to_end": end_to_end_result,
        "passed": end_to_end_result[
            "end_to_end_passed"
        ],
    }


def validate_live_system_bridge(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live system bridge result.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "snapshot",
        "pipeline",
        "end_to_end",
        "passed",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "live system bridge is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["snapshot"], Mapping):
        raise ValueError("snapshot must be a mapping")

    if not isinstance(result["pipeline"], Mapping):
        raise ValueError("pipeline must be a mapping")

    if not isinstance(result["end_to_end"], Mapping):
        raise ValueError("end_to_end must be a mapping")

    if not isinstance(result["passed"], bool):
        raise ValueError("passed must be boolean")

    return True
