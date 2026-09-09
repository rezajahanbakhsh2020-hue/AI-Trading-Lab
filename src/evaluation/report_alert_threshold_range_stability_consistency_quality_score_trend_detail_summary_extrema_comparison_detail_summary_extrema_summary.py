from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a compact extrema summary for quality-score trend metrics."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metric_extrema = report["metric_extrema"]

    widest_metric = report["widest_metric"]
    widest_range = report["widest_range"]

    range_order = sorted(
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
        "widest_metric": widest_metric,
        "widest_range": widest_range,
        "range_order": tuple(range_order),
        "range_rank": {
            metric: rank
            for rank, metric in enumerate(range_order, start=1)
        },
        "range_spread": {
            metric: metric_extrema[metric]["range"]
            for metric in metric_extrema
        },
        "peak_snapshots": {
            metric: metric_extrema[metric]["peak_snapshot"]
            for metric in metric_extrema
        },
        "floor_snapshots": {
            metric: metric_extrema[metric]["floor_snapshot"]
            for metric in metric_extrema
        },
        "metric_extrema": metric_extrema,
        "details": report["details"],
        "metric_details": report["metric_details"],
    }
