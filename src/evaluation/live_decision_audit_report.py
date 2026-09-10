from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.live_decision_audit import (
    audit_live_decision_history,
)
from src.evaluation.live_decision_change import (
    compare_live_decisions,
)
from src.evaluation.live_decision_consistency import (
    check_live_decision_consistency,
)


def build_live_decision_audit_report(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build one operational audit report for live decision history.

    The report combines structural audit, temporal consistency, and
    consecutive-decision changes. It does not recalculate trading values.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    structural = audit_live_decision_history(records)
    consistency = check_live_decision_consistency(records)

    changes: list[dict[str, Any]] = []

    for index in range(1, len(records)):
        comparison = compare_live_decisions(
            records[index - 1],
            records[index],
        )

        changes.append(
            {
                "from_index": index - 1,
                "to_index": index,
                **comparison,
            }
        )

    changed_transition_count = sum(
        1 for item in changes
        if item["changed"]
    )

    return {
        "record_count": len(records),
        "valid_structure": structural["valid"],
        "structural_audit": structural,
        "consistent_timeline": consistency["consistent"],
        "consistency_audit": consistency,
        "transition_count": len(changes),
        "changed_transition_count": changed_transition_count,
        "changes": changes,
        "audit_passed": (
            structural["valid"]
            and consistency["consistent"]
        ),
    }


def validate_live_decision_audit_report(
    report: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of an operational live-decision audit report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")

    required_fields = (
        "record_count",
        "valid_structure",
        "structural_audit",
        "consistent_timeline",
        "consistency_audit",
        "transition_count",
        "changed_transition_count",
        "changes",
        "audit_passed",
    )

    missing = [
        field
        for field in required_fields
        if field not in report
    ]

    if missing:
        raise ValueError(
            "audit report is missing required fields: "
            + ", ".join(missing)
        )

    for field in (
        "record_count",
        "transition_count",
        "changed_transition_count",
    ):
        if not isinstance(report[field], int):
            raise ValueError(
                f"{field} must be an integer"
            )

    for field in (
        "valid_structure",
        "consistent_timeline",
        "audit_passed",
    ):
        if not isinstance(report[field], bool):
            raise ValueError(
                f"{field} must be boolean"
            )

    if not isinstance(report["structural_audit"], Mapping):
        raise ValueError(
            "structural_audit must be a mapping"
        )

    if not isinstance(report["consistency_audit"], Mapping):
        raise ValueError(
            "consistency_audit must be a mapping"
        )

    if not isinstance(report["changes"], list):
        raise ValueError("changes must be a list")

    if report["transition_count"] != len(report["changes"]):
        raise ValueError(
            "transition_count must match changes"
        )

    if report["changed_transition_count"] != sum(
        1 for change in report["changes"]
        if isinstance(change, Mapping)
        and change.get("changed") is True
    ):
        raise ValueError(
            "changed_transition_count does not match changes"
        )

    return True
