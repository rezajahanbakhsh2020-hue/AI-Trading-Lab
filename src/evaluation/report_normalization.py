from __future__ import annotations

from typing import Any, Mapping


def normalize_report_metrics(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Normalize the main numeric evaluation metrics.

    Percentage-style metrics such as total_return, max_drawdown,
    and win_rate are kept in their original numeric representation.
    Numeric values are converted to float, while non-numeric values
    are preserved.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    normalized: dict[str, Any] = {}

    for field, value in report.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            normalized[field] = float(value)
        else:
            normalized[field] = value

    return normalized


def normalize_report_metric(
    report: Mapping[str, Any],
    metric: str,
) -> dict[str, Any]:
    """
    Normalize one selected numeric metric in an evaluation report.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping.")

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    if metric not in report:
        raise ValueError(
            f"Report field is missing: {metric}"
        )

    value = report[metric]

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        normalized = dict(report)
        normalized[metric] = float(value)
        return normalized

    raise ValueError(
        f"Report field must be numeric: {metric}"
    )


def normalize_reports(
    reports: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Normalize numeric values across multiple evaluation reports.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    return [
        normalize_report_metrics(report)
        for report in reports
    ]
