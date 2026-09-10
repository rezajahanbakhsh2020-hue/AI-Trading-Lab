from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.live_decision_health import (
    calculate_live_decision_health,
)
from src.evaluation.live_decision_summary import (
    summarize_live_decisions,
)


def evaluate_live_decision_history(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a final evaluation of recorded live decisions.

    This function evaluates recorded decision history only. It does not
    recalculate signals, risk levels, stability scores, or performance.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    summary = summarize_live_decisions(records)
    health = calculate_live_decision_health(records)

    return {
        "record_count": summary["record_count"],
        "signal_counts": dict(summary["signal_counts"]),
        "trend_counts": dict(summary["trend_counts"]),
        "strategy_counts": dict(summary["strategy_counts"]),
        "latest": (
            dict(summary["latest"])
            if summary["latest"] is not None
            else None
        ),
        "health_status": health["status"],
        "healthy": health["healthy"],
        "audit_passed": health["audit_passed"],
        "invalid_structure_count": health[
            "invalid_structure_count"
        ],
        "timeline_issue_count": health[
            "timeline_issue_count"
        ],
        "changed_transition_count": health[
            "changed_transition_count"
        ],
    }


def validate_live_decision_evaluation(
    evaluation: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live decision evaluation.
    """
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping")

    required_fields = (
        "record_count",
        "signal_counts",
        "trend_counts",
        "strategy_counts",
        "latest",
        "health_status",
        "healthy",
        "audit_passed",
        "invalid_structure_count",
        "timeline_issue_count",
        "changed_transition_count",
    )

    missing = [
        field
        for field in required_fields
        if field not in evaluation
    ]

    if missing:
        raise ValueError(
            "evaluation is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(evaluation["record_count"], int):
        raise ValueError("record_count must be an integer")

    for field in (
        "signal_counts",
        "trend_counts",
        "strategy_counts",
    ):
        if not isinstance(evaluation[field], Mapping):
            raise ValueError(
                f"{field} must be a mapping"
            )

    if evaluation["latest"] is not None and not isinstance(
        evaluation["latest"], Mapping
    ):
        raise ValueError("latest must be a mapping or None")

    if not isinstance(evaluation["health_status"], str):
        raise ValueError("health_status must be a string")

    for field in (
        "healthy",
        "audit_passed",
    ):
        if not isinstance(evaluation[field], bool):
            raise ValueError(
                f"{field} must be boolean"
            )

    for field in (
        "invalid_structure_count",
        "timeline_issue_count",
        "changed_transition_count",
    ):
        if not isinstance(evaluation[field], int):
            raise ValueError(
                f"{field} must be an integer"
            )

    return True
