from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.live_decision_audit_report import (
    build_live_decision_audit_report,
)


def calculate_live_decision_health(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Calculate a compact health status for live decision history.

    This function evaluates recorded decision integrity only. It does not
    recalculate signals, risk levels, stability, or performance.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    audit = build_live_decision_audit_report(records)

    record_count = audit["record_count"]
    invalid_structure = (
        audit["structural_audit"]["invalid_count"]
    )
    timeline_issues = (
        audit["consistency_audit"]["issue_count"]
    )

    if record_count == 0:
        status = "EMPTY"
    elif not audit["audit_passed"]:
        status = "UNHEALTHY"
    elif record_count == 1:
        status = "INITIALIZING"
    elif timeline_issues == 0 and invalid_structure == 0:
        status = "HEALTHY"
    else:
        status = "DEGRADED"

    return {
        "status": status,
        "healthy": status in {"HEALTHY", "INITIALIZING"},
        "record_count": record_count,
        "invalid_structure_count": invalid_structure,
        "timeline_issue_count": timeline_issues,
        "changed_transition_count": audit[
            "changed_transition_count"
        ],
        "audit_passed": audit["audit_passed"],
    }


def validate_live_decision_health(
    health: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live decision health result.
    """
    if not isinstance(health, Mapping):
        raise TypeError("health must be a mapping")

    required_fields = (
        "status",
        "healthy",
        "record_count",
        "invalid_structure_count",
        "timeline_issue_count",
        "changed_transition_count",
        "audit_passed",
    )

    missing = [
        field
        for field in required_fields
        if field not in health
    ]

    if missing:
        raise ValueError(
            "health result is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(health["status"], str):
        raise ValueError("status must be a string")

    if not isinstance(health["healthy"], bool):
        raise ValueError("healthy must be boolean")

    if not isinstance(health["audit_passed"], bool):
        raise ValueError("audit_passed must be boolean")

    for field in (
        "record_count",
        "invalid_structure_count",
        "timeline_issue_count",
        "changed_transition_count",
    ):
        if not isinstance(health[field], int):
            raise ValueError(
                f"{field} must be an integer"
            )

    return True
