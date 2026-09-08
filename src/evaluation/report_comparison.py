from __future__ import annotations

from typing import Any, Mapping


def compare_reports(
    first_report: Mapping[str, Any],
    second_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Compare common metrics between two evaluation reports.

    Parameters
    ----------
    first_report:
        First evaluation report.
    second_report:
        Second evaluation report.

    Returns
    -------
    dict[str, dict[str, Any]]
        Comparison results for common fields. Each field contains
        the values from both reports and their difference.

    Raises
    ------
    TypeError
        If either report is not a mapping.
    """
    if not isinstance(first_report, Mapping):
        raise TypeError("first_report must be a mapping.")

    if not isinstance(second_report, Mapping):
        raise TypeError("second_report must be a mapping.")

    common_fields = [
        field
        for field in first_report
        if field in second_report
    ]

    comparison: dict[str, dict[str, Any]] = {}

    for field in common_fields:
        first_value = first_report[field]
        second_value = second_report[field]

        result: dict[str, Any] = {
            "first": first_value,
            "second": second_value,
        }

        if (
            isinstance(first_value, (int, float))
            and not isinstance(first_value, bool)
            and isinstance(second_value, (int, float))
            and not isinstance(second_value, bool)
        ):
            result["difference"] = second_value - first_value

        comparison[field] = result

    return comparison


def calculate_report_differences(
    first_report: Mapping[str, Any],
    second_report: Mapping[str, Any],
) -> dict[str, float]:
    """
    Calculate numeric differences between two evaluation reports.

    The difference is calculated as:

        second_report - first_report

    Parameters
    ----------
    first_report:
        First evaluation report.
    second_report:
        Second evaluation report.

    Returns
    -------
    dict[str, float]
        Numeric differences for common numeric fields.

    Raises
    ------
    TypeError
        If either report is not a mapping.
    """
    if not isinstance(first_report, Mapping):
        raise TypeError("first_report must be a mapping.")

    if not isinstance(second_report, Mapping):
        raise TypeError("second_report must be a mapping.")

    differences: dict[str, float] = {}

    for field in first_report:
        if field not in second_report:
            continue

        first_value = first_report[field]
        second_value = second_report[field]

        if (
            isinstance(first_value, (int, float))
            and not isinstance(first_value, bool)
            and isinstance(second_value, (int, float))
            and not isinstance(second_value, bool)
        ):
            differences[field] = float(second_value - first_value)

    return differences


def find_improved_metrics(
    first_report: Mapping[str, Any],
    second_report: Mapping[str, Any],
) -> list[str]:
    """
    Return common numeric metrics whose value increased.

    Parameters
    ----------
    first_report:
        First evaluation report.
    second_report:
        Second evaluation report.

    Returns
    -------
    list[str]
        Names of metrics with higher values in the second report.
    """
    differences = calculate_report_differences(
        first_report,
        second_report,
    )

    return [
        field
        for field, difference in differences.items()
        if difference > 0
    ]


def find_degraded_metrics(
    first_report: Mapping[str, Any],
    second_report: Mapping[str, Any],
) -> list[str]:
    """
    Return common numeric metrics whose value decreased.

    Parameters
    ----------
    first_report:
        First evaluation report.
    second_report:
        Second evaluation report.

    Returns
    -------
    list[str]
        Names of metrics with lower values in the second report.
    """
    differences = calculate_report_differences(
        first_report,
        second_report,
    )

    return [
        field
        for field, difference in differences.items()
        if difference < 0
    ]
