from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema import (
    build_quality_score_trend_detail_summary_extrema,
)


def build_quality_score_trend_detail_summary_extrema_comparison(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a comparison-focused extrema report for quality-score trends."""

    report = build_quality_score_trend_detail_summary_extrema(
        history,
        lower_bound,
        upper_bound,
    )

    quality_change = report["quality_change"]
    stability_change = report["stability_change"]
    consistency_change = report["consistency_change"]

    changes = {
        "quality": quality_change,
        "stability": stability_change,
        "consistency": consistency_change,
    }

    largest_change_metric = max(
        changes,
        key=lambda metric: abs(changes[metric]),
        default=None,
    )

    if largest_change_metric is None:
        largest_change = 0.0
    else:
        largest_change = changes[largest_change_metric]

    positive_changes = {
        metric: value
        for metric, value in changes.items()
        if value > 0
    }

    negative_changes = {
        metric: value
        for metric, value in changes.items()
        if value < 0
    }

    return {
        "snapshot_count": report["snapshot_count"],
        "lower_bound": report["lower_bound"],
        "upper_bound": report["upper_bound"],
        "quality_trend": report["quality_trend"],
        "stability_trend": report["stability_trend"],
        "consistency_trend": report["consistency_trend"],
        "quality_change": quality_change,
        "stability_change": stability_change,
        "consistency_change": consistency_change,
        "quality_range": report["quality_range"],
        "stability_range": report["stability_range"],
        "consistency_range": report["consistency_range"],
        "quality_peak_snapshot": report["quality_peak_snapshot"],
        "quality_floor_snapshot": report["quality_floor_snapshot"],
        "stability_peak_snapshot": report["stability_peak_snapshot"],
        "stability_floor_snapshot": report["stability_floor_snapshot"],
        "consistency_peak_snapshot": report["consistency_peak_snapshot"],
        "consistency_floor_snapshot": report["consistency_floor_snapshot"],
        "largest_change_metric": largest_change_metric,
        "largest_change": largest_change,
        "positive_change_metrics": tuple(positive_changes),
        "negative_change_metrics": tuple(negative_changes),
        "details": report["details"],
    }
