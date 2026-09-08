from __future__ import annotations

from typing import Any, Mapping


def find_common_report_metrics(
    reports: list[Mapping[str, Any]],
) -> list[str]:
    """
    Return metric names that are present in every report.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not reports:
        return []

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

    common_metrics = set(reports[0].keys())

    for report in reports[1:]:
        common_metrics.intersection_update(report.keys())

    return [
        field
        for field in reports[0]
        if field in common_metrics
    ]


def find_inconsistent_report_metrics(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> list[str]:
    """
    Return metrics whose presence differs across reports.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

    for metric in metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "metric names must be strings."
            )

    inconsistent: list[str] = []

    for metric in metrics:
        presence = [
            metric in report
            for report in reports
        ]

        if presence and any(presence) and not all(presence):
            inconsistent.append(metric)

    return inconsistent


def calculate_report_metric_coverage(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, float]:
    """
    Calculate the percentage of reports containing each metric.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

    for metric in metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "metric names must be strings."
            )

    if not reports:
        return {
            metric: 0.0
            for metric in metrics
        }

    return {
        metric: round(
            sum(metric in report for report in reports)
            / len(reports),
            10,
        )
        for metric in metrics
    }


def build_report_consistency_summary(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, Any]:
    """
    Build a consistency summary for a collection of reports.
    """
    coverage = calculate_report_metric_coverage(
        reports,
        metrics,
    )

    inconsistent = find_inconsistent_report_metrics(
        reports,
        metrics,
    )

    common = find_common_report_metrics(reports)

    return {
        "report_count": len(reports),
        "common_metrics": common,
        "inconsistent_metrics": inconsistent,
        "coverage": coverage,
        "consistent": not inconsistent,
    }
