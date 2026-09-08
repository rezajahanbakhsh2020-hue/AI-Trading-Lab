from __future__ import annotations

from typing import Any, Mapping


def calculate_metric_trend(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> dict[str, float | str]:
    """
    Calculate the trend of a numeric metric across ordered reports.

    The reports are assumed to be ordered from oldest to newest.
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

    first = values[0]
    last = values[-1]
    change = round(last - first, 10)

    if change > 0:
        direction = "increasing"
    elif change < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "first": first,
        "last": last,
        "change": change,
        "direction": direction,
    }


def calculate_metric_changes(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> list[float]:
    """
    Calculate consecutive changes for a numeric metric.
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

    return [
        round(values[index] - values[index - 1], 10)
        for index in range(1, len(values))
    ]


def find_improving_trends(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> list[str]:
    """
    Return metrics whose final value is greater than their initial value.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    improving: list[str] = []

    for metric in metrics:
        trend = calculate_metric_trend(
            reports,
            metric,
        )

        if trend["direction"] == "increasing":
            improving.append(metric)

    return improving
