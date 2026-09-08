from __future__ import annotations

from typing import Any, Mapping


def aggregate_report_metric(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    """
    Aggregate a numeric metric across evaluation reports.
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
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"Report must contain a numeric value for: {metric}"
            )

        values.append(float(value))

    if not values:
        raise ValueError(
            f"No reports available for metric: {metric}"
        )

    total = sum(values)
    mean = total / len(values)

    return {
        "count": float(len(values)),
        "sum": total,
        "mean": mean,
    }


def aggregate_report_metrics(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """
    Aggregate multiple numeric metrics across reports.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    result: dict[str, dict[str, float]] = {}

    for metric in metrics:
        result[metric] = aggregate_report_metric(
            reports,
            metric,
        )

    return result


def calculate_weighted_report_metric(
    reports: list[Mapping[str, Any]],
    metric: str,
    weight_field: str,
) -> float:
    """
    Calculate a weighted average of a report metric.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    if not isinstance(weight_field, str):
        raise TypeError("weight_field must be a string.")

    weighted_sum = 0.0
    total_weight = 0.0

    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("Each report must be a mapping.")

        value = report.get(metric)
        weight = report.get(weight_field)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"Report must contain a numeric value for: {metric}"
            )

        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
        ):
            raise ValueError(
                f"Report must contain a numeric value for: {weight_field}"
            )

        if weight < 0:
            raise ValueError(
                f"Weight must not be negative: {weight_field}"
            )

        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)

    if total_weight == 0:
        raise ValueError("Total weight must be greater than zero.")

    return weighted_sum / total_weight
