from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


def group_reports_by_field(
    reports: list[Mapping[str, Any]],
    field: str,
) -> dict[Any, list[Mapping[str, Any]]]:
    """
    Group evaluation reports by the value of a selected field.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(field, str):
        raise TypeError("field must be a string.")

    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

        if field not in report:
            raise ValueError(
                f"Report field is missing: {field}"
            )

        value = report[field]

        try:
            groups[value].append(report)
        except TypeError as exc:
            raise ValueError(
                f"Report field value must be hashable: {field}"
            ) from exc

    return dict(groups)


def group_reports_by_status(
    reports: list[Mapping[str, Any]],
) -> dict[Any, list[Mapping[str, Any]]]:
    """
    Group evaluation reports by their status field.
    """
    return group_reports_by_field(
        reports,
        "status",
    )


def count_reports_by_field(
    reports: list[Mapping[str, Any]],
    field: str,
) -> dict[Any, int]:
    """
    Count evaluation reports for each value of a selected field.
    """
    groups = group_reports_by_field(
        reports,
        field,
    )

    return {
        key: len(group)
        for key, group in groups.items()
    }


def count_reports_by_status(
    reports: list[Mapping[str, Any]],
) -> dict[Any, int]:
    """
    Count evaluation reports by status.
    """
    return count_reports_by_field(
        reports,
        "status",
    )
