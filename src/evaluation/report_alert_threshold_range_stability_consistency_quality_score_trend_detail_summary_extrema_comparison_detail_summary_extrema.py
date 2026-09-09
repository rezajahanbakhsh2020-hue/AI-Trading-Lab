from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build extrema-focused details from the comparison summary."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metric_extrema = {
        "quality": {
            "range": report["quality_range"],
            "peak_snapshot": report["quality_peak_snapshot"],
            "floor_snapshot": report["quality_floor_snapshot"],
        },
        "stability": {
            "range": report["stability_range"],
            "peak_snapshot": report["stability_peak_snapshot"],
            "floor_snapshot": report["stability_floor_snapshot"],
        },
        "consistency": {
            "range": report["consistency_range"],
            "peak_snapshot": report["consistency_peak_snapshot"],
            "floor_snapshot": report["consistency_floor_snapshot"],
        },
    }

    widest_metric = min(
        metric_extrema,
        key=lambda metric: (
            -metric_extrema[metric]["range"],
            ("quality", "stability", "consistency").index(metric),
        ),
    )

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
        "positive_change_metrics": report["positive_change_metrics"],
        "negative_change_metrics": report["negative_change_metrics"],
        "up_metrics": report["up_metrics"],
        "down_metrics": report["down_metrics"],
        "flat_metrics": report["flat_metrics"],
        "metric_extrema": metric_extrema,
        "widest_metric": widest_metric,
        "widest_range": metric_extrema[widest_metric]["range"],
        "details": report["details"],
        "metric_details": report["metric_details"],
    }
