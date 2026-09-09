from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison import (
    build_quality_score_trend_detail_summary_extrema_comparison,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build detailed per-metric comparison information for quality-score trends."""

    report = build_quality_score_trend_detail_summary_extrema_comparison(
        history,
        lower_bound,
        upper_bound,
    )

    changes = {
        "quality": report["quality_change"],
        "stability": report["stability_change"],
        "consistency": report["consistency_change"],
    }

    metric_details = [
        {
            "metric": metric,
            "change": change,
            "direction": (
                "up"
                if change > 0
                else "down"
                if change < 0
                else "flat"
            ),
            "is_largest_change": metric == report["largest_change_metric"],
            "is_positive": change > 0,
            "is_negative": change < 0,
        }
        for metric, change in changes.items()
    ]

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
        "quality_range": report["quality_range"],
        "stability_range": report["stability_range"],
        "consistency_range": report["consistency_range"],
        "quality_peak_snapshot": report["quality_peak_snapshot"],
        "quality_floor_snapshot": report["quality_floor_snapshot"],
        "stability_peak_snapshot": report["stability_peak_snapshot"],
        "stability_floor_snapshot": report["stability_floor_snapshot"],
        "consistency_peak_snapshot": report["consistency_peak_snapshot"],
        "consistency_floor_snapshot": report["consistency_floor_snapshot"],
        "largest_change_metric": report["largest_change_metric"],
        "largest_change": report["largest_change"],
        "positive_change_metrics": report["positive_change_metrics"],
        "negative_change_metrics": report["negative_change_metrics"],
        "metric_details": metric_details,
        "details": report["details"],
    }
