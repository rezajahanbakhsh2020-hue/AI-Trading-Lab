from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.live_decision_record import (
    DECISION_RECORD_FIELDS,
)


def audit_live_decision(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Audit one live decision record without recalculating trading values.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    required_fields = (
        "timestamp",
        "symbol",
        "interval",
        "signal",
        "signal_label",
        "trend",
        "strategy",
    )

    missing_fields = [
        field for field in required_fields
        if field not in record
    ]

    empty_fields = [
        field
        for field in required_fields
        if field in record
        and record.get(field) is None
    ]

    unexpected_fields = [
        field
        for field in record
        if field not in DECISION_RECORD_FIELDS
    ]

    return {
        "valid": not missing_fields and not empty_fields,
        "missing_fields": missing_fields,
        "empty_fields": empty_fields,
        "unexpected_fields": unexpected_fields,
        "field_count": len(record),
    }


def audit_live_decision_history(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Audit a collection of live decision records.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records = list(history)

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each history item must be a mapping")

    audits = [
        audit_live_decision(record)
        for record in records
    ]

    invalid_count = sum(
        1 for audit in audits
        if not audit["valid"]
    )

    return {
        "record_count": len(records),
        "valid_count": len(records) - invalid_count,
        "invalid_count": invalid_count,
        "valid": invalid_count == 0,
        "audits": audits,
    }


def validate_live_decision_audit(
    audit: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of an audit result.
    """
    if not isinstance(audit, Mapping):
        raise TypeError("audit must be a mapping")

    required_fields = (
        "valid",
        "missing_fields",
        "empty_fields",
        "unexpected_fields",
        "field_count",
    )

    missing = [
        field for field in required_fields
        if field not in audit
    ]

    if missing:
        raise ValueError(
            "audit is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(audit["valid"], bool):
        raise ValueError("audit 'valid' must be boolean")

    if not isinstance(audit["missing_fields"], list):
        raise ValueError("audit 'missing_fields' must be a list")

    if not isinstance(audit["empty_fields"], list):
        raise ValueError("audit 'empty_fields' must be a list")

    if not isinstance(audit["unexpected_fields"], list):
        raise ValueError(
            "audit 'unexpected_fields' must be a list"
        )

    if not isinstance(audit["field_count"], int):
        raise ValueError("audit 'field_count' must be an integer")

    return True
