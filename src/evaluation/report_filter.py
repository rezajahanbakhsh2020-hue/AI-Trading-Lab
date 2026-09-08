from __future__ import annotations

from typing import Any, Mapping


def filter_reports(
    reports: list[Mapping[str, Any]],
    metric: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[Mapping[str, Any]]:
    """
    Filter evaluation reports by a numeric metric range.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    if minimum is not None and (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
    ):
        raise TypeError("minimum must be numeric or None.")

    if maximum is not None and (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
    ):
        raise TypeError("maximum must be numeric or None.")

    if minimum is not None and maximum is not None:
        if minimum > maximum:
            raise ValueError(
                "minimum must not be greater than maximum."
            )

    filtered: list[Mapping[str, Any]] = []

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

        value = report.get(metric)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"Report must contain a numeric value for: {metric}"
            )

        if minimum is not None and value < minimum:
            continue

        if maximum is not None and value > maximum:
            continue

        filtered.append(report)

    return filtered


def filter_reports_by_status(
    reports: list[Mapping[str, Any]],
    status: str,
) -> list[Mapping[str, Any]]:
    """
    Filter evaluation reports by their status field.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(status, str):
        raise TypeError("status must be a string.")

    return [
        report
        for report in reports
        if isinstance(report, Mapping)
        and report.get("status") == status
    ]


def filter_acceptable_reports(
    reports: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """
    Return reports marked as acceptable.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    return [
        report
        for report in reports
        if isinstance(report, Mapping)
        and report.get("acceptable") is True
    ]
