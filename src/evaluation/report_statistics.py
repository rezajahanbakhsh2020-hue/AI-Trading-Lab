from __future__ import annotations

from typing import Any, Mapping


def calculate_metric_statistics(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    """
    Calculate basic statistics for a metric across multiple reports.

    Parameters
    ----------
    reports:
        Collection of evaluation reports.
    metric:
        Metric name to analyze.

    Returns
    -------
    dict[str, float]
        Count, minimum, maximum, mean, and range.

    Raises
    ------
    TypeError
        If reports or metric has an invalid type.
    ValueError
        If no valid numeric values are available.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    values: list[float] = []

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

        value = report.get(metric)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            values.append(float(value))

    if not values:
        raise ValueError(
            f"No numeric values found for metric: {metric}"
        )

    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)

    return {
        "count": float(len(values)),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "range": maximum - minimum,
    }


def calculate_report_statistics(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """
    Calculate statistics for multiple metrics across reports.

    Parameters
    ----------
    reports:
        Collection of evaluation reports.
    metrics:
        Metric names to analyze.

    Returns
    -------
    dict[str, dict[str, float]]
        Statistics keyed by metric name.

    Raises
    ------
    TypeError
        If reports or metrics has an invalid type.
    ValueError
        If a requested metric has no numeric values.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    result: dict[str, dict[str, float]] = {}

    for metric in metrics:
        result[metric] = calculate_metric_statistics(
            reports,
            metric,
        )

    return result


def find_best_report(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> Mapping[str, Any]:
    """
    Return the report with the highest numeric value for a metric.

    Parameters
    ----------
    reports:
        Collection of evaluation reports.
    metric:
        Metric used for ranking.

    Returns
    -------
    Mapping[str, Any]
        Report with the highest metric value.

    Raises
    ------
    TypeError
        If reports or metric has an invalid type.
    ValueError
        If no report contains a numeric value for the metric.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    valid_reports = [
        report
        for report in reports
        if isinstance(report, Mapping)
        and isinstance(report.get(metric), (int, float))
        and not isinstance(report.get(metric), bool)
    ]

    if not valid_reports:
        raise ValueError(
            f"No numeric values found for metric: {metric}"
        )

    return max(
        valid_reports,
        key=lambda report: float(report[metric]),
    )
