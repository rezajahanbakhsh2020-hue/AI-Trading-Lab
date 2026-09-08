from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def validate_report(
    report: Mapping[str, Any],
    required_fields: Iterable[str] | None = None,
) -> bool:
    """
    Validate the basic structure of an evaluation report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if required_fields is None:
        return True

    if isinstance(required_fields, str):
        raise TypeError(
            "required_fields must be an iterable of field names."
        )

    for field in required_fields:
        if not isinstance(field, str):
            raise TypeError(
                "required field names must be strings."
            )

        if field not in report:
            raise ValueError(
                f"Required report field is missing: {field}"
            )

    return True


def validate_numeric_fields(
    report: Mapping[str, Any],
    fields: Iterable[str],
) -> bool:
    """
    Validate that selected report fields contain finite numeric values.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if isinstance(fields, str):
        raise TypeError(
            "fields must be an iterable of field names."
        )

    for field in fields:
        if not isinstance(field, str):
            raise TypeError(
                "field names must be strings."
            )

        if field not in report:
            raise ValueError(
                f"Report field is missing: {field}"
            )

        value = report[field]

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"Report field must be numeric: {field}"
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"Report field must be finite: {field}"
            )

    return True


def validate_report_fields(
    report: Mapping[str, Any],
    required_fields: Iterable[str] | None = None,
    numeric_fields: Iterable[str] | None = None,
) -> bool:
    """
    Validate required and numeric fields in an evaluation report.
    """
    validate_report(
        report,
        required_fields=required_fields,
    )

    if numeric_fields is not None:
        validate_numeric_fields(
            report,
            numeric_fields,
        )

    return True
