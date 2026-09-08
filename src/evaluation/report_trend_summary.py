from __future__ import annotations

from typing import Any, Mapping

from src.evaluation.report_trend import calculate_metric_trend


def summarize_report_trends(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, float | str]]:
    """
    Calculate trends for multiple metrics across ordered reports.
    """
    if not isinstance(reports, list):
        raise TypeError("reports must be a list.")

    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list.")

    for metric in metrics:
        if not isinstance(metric, str):
            raise TypeError(
                "metric names must be strings."
            )

    return {
        metric: calculate_metric_trend(
            reports,
            metric,
        )
        for metric in metrics
    }


def count_trend_directions(
    trends: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """
    Count increasing, decreasing, and stable metric trends.
    """
    if not isinstance(trends, Mapping):
        raise TypeError("trends must be a mapping.")

    counts = {
        "increasing": 0,
        "decreasing": 0,
        "stable": 0,
    }

    for metric, trend in trends.items():
        if not isinstance(metric, str):
            raise TypeError(
                "trend metric names must be strings."
            )

        if not isinstance(trend, Mapping):
            raise TypeError(
                "Each trend must be a mapping."
            )

        direction = trend.get("direction")

        if direction not in counts:
            raise ValueError(
                f"Invalid trend direction: {direction}"
            )

        counts[direction] += 1

    return counts


def build_report_trend_summary(
    reports: list[Mapping[str, Any]],
    metrics: list[str],
) -> dict[str, Any]:
    """
    Build a complete multi-metric trend summary.
    """
    trends = summarize_report_trends(
        reports,
        metrics,
    )

    counts = count_trend_directions(
        trends,
    )

    return {
        "report_count": len(reports),
        "metric_count": len(metrics),
        "trends": trends,
        "direction_counts": counts,
    }
