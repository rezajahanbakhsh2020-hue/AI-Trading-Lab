from __future__ import annotations

from typing import Any, Mapping


def calculate_metric_drift(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> float:
    """
    Calculate the absolute drift between the first and last
    numeric values of a metric.
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

    return round(abs(values[-1] - values[0]), 10)


def calculate_relative_metric_drift(
    reports: list[Mapping[str, Any]],
    metric: str,
) -> float:
    """
    Calculate relative drift between the first and last values.

    The calculation uses the absolute first value as the denominator.
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

    if first == 0:
        if last == 0:
            return 0.0
        raise ValueError(
            "Relative drift cannot be calculated from a zero baseline."
        )

    return round(
        abs(last - first) / abs(first),
        10,
    )


def find_high_drift_metrics(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
    maximum_drift: float,
) -> list[str]:
    """
    Return metrics whose absolute drift exceeds the threshold.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    if (
        isinstance(maximum_drift, bool)
        or not isinstance(maximum_drift, (int, float))
    ):
        raise TypeError(
            "maximum_drift must be numeric."
        )

    if maximum_drift < 0:
        raise ValueError(
            "maximum_drift must not be negative."
        )

    for metric in metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "metric names must be strings."
            )

    return [
        metric
        for metric in metrics
        if calculate_metric_drift(
            reports,
            metric,
        ) > maximum_drift
    ]


def build_report_drift_summary(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
    maximum_drift: float,
) -> dict[str, Any]:
    """
    Build a compact drift summary for multiple report metrics.
    """
    drift = {
        metric: calculate_metric_drift(
            reports,
            metric,
        )
        for metric in metrics
    }

    high_drift = find_high_drift_metrics(
        reports,
        metrics,
        maximum_drift,
    )

    return {
        "report_count": len(reports),
        "drift": drift,
        "high_drift_metrics": high_drift,
        "stable": not high_drift,
    }
