from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build extrema summaries from per-metric report-summary details."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metric_details = report["metric_details"]

    changes = {
        item["metric"]: item["change"]
        for item in metric_details
    }

    ranges = {
        item["metric"]: item["range"]
        for item in metric_details
    }

    if changes:
        largest_change_metric = max(
            changes,
            key=lambda metric: abs(changes[metric]),
        )
        smallest_change_metric = min(
            changes,
            key=lambda metric: abs(changes[metric]),
        )
        maximum_change_metric = max(
            changes,
            key=changes.get,
        )
        minimum_change_metric = min(
            changes,
            key=changes.get,
        )
    else:
        largest_change_metric = None
        smallest_change_metric = None
        maximum_change_metric = None
        minimum_change_metric = None

    if ranges:
        widest_metric = max(
            ranges,
            key=ranges.get,
        )
        narrowest_metric = min(
            ranges,
            key=ranges.get,
        )
    else:
        widest_metric = None
        narrowest_metric = None

    change_values = tuple(changes.values())
    range_values = tuple(ranges.values())

    if change_values:
        change_range = max(change_values) - min(change_values)
    else:
        change_range = 0.0

    if range_values:
        range_range = max(range_values) - min(range_values)
    else:
        range_range = 0.0

    return {
        "snapshot_count": report["snapshot_count"],
        "lower_bound": report["lower_bound"],
        "upper_bound": report["upper_bound"],
        "quality_trend": report["quality_trend"],
        "stability_trend": report["stability_trend"],
        "consistency_trend": report["consistency_trend"],
        "quality_change": report["quality_change"],
        "stability_change": report["stability_change"],
        "consistency_change": report["consistency_change"],
        "largest_change_metric": report["largest_change_metric"],
        "largest_change": report["largest_change"],
        "overall_change_direction": report["overall_change_direction"],
        "widest_metric": report["widest_metric"],
        "widest_range": report["widest_range"],
        "improving_metrics": report["improving_metrics"],
        "declining_metrics": report["declining_metrics"],
        "flat_metrics": report["flat_metrics"],
        "positive_change_metrics": report["positive_change_metrics"],
        "negative_change_metrics": report["negative_change_metrics"],
        "improving_metric_count": report["improving_metric_count"],
        "declining_metric_count": report["declining_metric_count"],
        "flat_metric_count": report["flat_metric_count"],
        "positive_change_metric_count": report["positive_change_metric_count"],
        "negative_change_metric_count": report["negative_change_metric_count"],
        "metric_count": report["metric_count"],
        "status_counts": report["status_counts"],
        "changes": changes,
        "ranges": ranges,
        "largest_change_abs_metric": largest_change_metric,
        "largest_change_abs": (
            abs(changes[largest_change_metric])
            if largest_change_metric is not None
            else 0.0
        ),
        "smallest_change_abs_metric": smallest_change_metric,
        "smallest_change_abs": (
            abs(changes[smallest_change_metric])
            if smallest_change_metric is not None
            else 0.0
        ),
        "maximum_change_metric": maximum_change_metric,
        "maximum_change": (
            changes[maximum_change_metric]
            if maximum_change_metric is not None
            else 0.0
        ),
        "minimum_change_metric": minimum_change_metric,
        "minimum_change": (
            changes[minimum_change_metric]
            if minimum_change_metric is not None
            else 0.0
        ),
        "change_range": change_range,
        "widest_metric": widest_metric,
        "widest_range": (
            ranges[widest_metric]
            if widest_metric is not None
            else 0.0
        ),
        "narrowest_metric": narrowest_metric,
        "narrowest_range": (
            ranges[narrowest_metric]
            if narrowest_metric is not None
            else 0.0
        ),
        "range_range": range_range,
        "metric_details": metric_details,
        "details": report["details"],
    }
