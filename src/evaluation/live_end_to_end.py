from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.live_decision_final import (
    finalize_live_decision,
)


def run_live_end_to_end(
    pipeline_result: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Run the final evaluation-side end-to-end proof.

    The function connects the existing live decision pipeline to the
    final decision package. It does not fetch market data, recalculate
    signals, recalculate risk levels, or execute orders.
    """
    if not isinstance(pipeline_result, Mapping):
        raise TypeError("pipeline_result must be a mapping")

    final_decision = finalize_live_decision(
        pipeline_result
    )

    handoff = final_decision["handoff"]
    readiness = final_decision["readiness"]

    proof = {
        "symbol": final_decision["symbol"],
        "signal_label": final_decision["signal_label"],
        "trend": final_decision["trend"],
        "strategy": final_decision["strategy"],
        "entry_price": final_decision["entry_price"],
        "stop_loss": final_decision["stop_loss"],
        "take_profit": final_decision["take_profit"],
        "risk_reward_ratio": final_decision[
            "risk_reward_ratio"
        ],
        "stability_score": final_decision[
            "stability_score"
        ],
        "actionable": final_decision["actionable"],
        "ready": final_decision["ready"],
        "status": final_decision["status"],
        "quote_stale": handoff["quote_stale"],
        "market_state": handoff["market_state"],
        "healthy": readiness["checks"]["healthy"],
        "audit_passed": readiness["checks"][
            "audit_passed"
        ],
    }

    return {
        "final_decision": final_decision,
        "proof": proof,
        "end_to_end_passed": (
            final_decision["ready"]
            and final_decision["actionable"]
            and readiness["checks"]["healthy"]
            and readiness["checks"]["audit_passed"]
        ),
    }


def validate_live_end_to_end(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of an end-to-end proof result.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "final_decision",
        "proof",
        "end_to_end_passed",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "end-to-end result is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(
        result["final_decision"],
        Mapping,
    ):
        raise ValueError(
            "final_decision must be a mapping"
        )

    if not isinstance(result["proof"], Mapping):
        raise ValueError("proof must be a mapping")

    if not isinstance(
        result["end_to_end_passed"],
        bool,
    ):
        raise ValueError(
            "end_to_end_passed must be boolean"
        )

    return True
