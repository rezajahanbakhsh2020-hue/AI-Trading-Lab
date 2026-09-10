from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.live_decision_handoff import (
    build_live_decision_handoff,
)
from src.evaluation.live_decision_readiness import (
    evaluate_live_decision_readiness,
)


def finalize_live_decision(
    pipeline_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build the final live decision package.

    The function connects the decision pipeline to the handoff and
    readiness layers. It does not recalculate signals, risk levels,
    stability, performance, or execute orders.
    """
    if not isinstance(pipeline_result, Mapping):
        raise TypeError("pipeline_result must be a mapping")

    handoff = build_live_decision_handoff(
        pipeline_result
    )

    readiness = evaluate_live_decision_readiness(
        handoff
    )

    return {
        "handoff": handoff,
        "readiness": readiness,
        "ready": readiness["ready"],
        "status": readiness["status"],
        "actionable": handoff["actionable"],
        "symbol": handoff["symbol"],
        "signal_label": handoff["signal_label"],
        "trend": handoff["trend"],
        "strategy": handoff["strategy"],
        "entry_price": handoff["entry_price"],
        "stop_loss": handoff["stop_loss"],
        "take_profit": handoff["take_profit"],
        "risk_reward_ratio": handoff["risk_reward_ratio"],
        "stability_score": handoff["stability_score"],
    }


def validate_final_live_decision(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of the final live decision package.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "handoff",
        "readiness",
        "ready",
        "status",
        "actionable",
        "symbol",
        "signal_label",
        "trend",
        "strategy",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "stability_score",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "final live decision is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["handoff"], Mapping):
        raise ValueError("handoff must be a mapping")

    if not isinstance(result["readiness"], Mapping):
        raise ValueError("readiness must be a mapping")

    if not isinstance(result["ready"], bool):
        raise ValueError("ready must be boolean")

    if not isinstance(result["actionable"], bool):
        raise ValueError("actionable must be boolean")

    if not isinstance(result["status"], str):
        raise ValueError("status must be a string")

    return True
