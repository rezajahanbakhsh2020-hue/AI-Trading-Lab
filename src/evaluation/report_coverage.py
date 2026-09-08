from __future__ import annotations

from typing import Any, Mapping


def calculate_report_coverage(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, float]:
    """
    Calculate the percentage of reports containing each metric
    with a numeric value.
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

    coverage: dict[str, float] = {}

    for metric in metrics:
        valid_count = 0

        for report in reports:
            value = report.get(metric)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                valid_count += 1

        coverage[metric] = round(
            valid_count / len(reports),
            10,
        )

    return coverage


def find_low_coverage_metrics(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
    minimum_coverage: float = 1.0,
) -> list[str]:
    """
    Return metrics whose numeric coverage is below the threshold.
    """
    if (
        isinstance(minimum_coverage, bool)
        or not isinstance(minimum_coverage, (int, float))
    ):
        raise TypeError(
            "minimum_coverage must be numeric."
        )

    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError(
            "minimum_coverage must be between 0 and 1."
        )

    coverage = calculate_report_coverage(
        reports,
        metrics,
    )

    return [
        metric
        for metric in metrics
        if coverage[metric] < minimum_coverage
    ]


def is_report_set_complete(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> bool:
    """
    Return True when every report contains a numeric value
    for every requested metric.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not reports:
        return False

    coverage = calculate_report_coverage(
        reports,
        metrics,
    )

    return all(
        value == 1.0
        for value in coverage.values()
    )


def build_report_coverage_summary(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
    minimum_coverage: float = 1.0,
) -> dict[str, Any]:
    """
    Build a coverage summary for a collection of reports.
    """
    coverage = calculate_report_coverage(
        reports,
        metrics,
    )

    low_coverage = find_low_coverage_metrics(
        reports,
        metrics,
        minimum_coverage=minimum_coverage,
    )

    return {
        "report_count": len(reports),
        "coverage": coverage,
        "low_coverage_metrics": low_coverage,
        "complete": is_report_set_complete(
            reports,
            metrics,
        ),
    }
