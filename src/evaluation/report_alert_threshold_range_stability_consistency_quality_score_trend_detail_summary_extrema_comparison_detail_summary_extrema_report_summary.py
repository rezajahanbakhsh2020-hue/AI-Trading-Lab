from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a compact summary from the final extrema report."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metrics = (
        "quality",
        "stability",
        "consistency",
    )

    improving_metrics = tuple(
        metric
        for metric in metrics
        if report[f"{metric}_change"] > 0
    )

    declining_metrics = tuple(
        metric
        for metric in metrics
        if report[f"{metric}_change"] < 0
    )

    flat_metrics = tuple(
        metric
        for metric in metrics
        if report[f"{metric}_change"] == 0
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
        "widest_metric": report["widest_metric"],
        "widest_range": report["widest_range"],
        "improving_metrics": improving_metrics,
        "declining_metrics": declining_metrics,
        "flat_metrics": flat_metrics,
        "improving_metric_count": len(improving_metrics),
        "declining_metric_count": len(declining_metrics),
        "flat_metric_count": len(flat_metrics),
        "metric_status": report["metric_status"],
        "metric_extrema": report["metric_extrema"],
        "details": report["details"],
        "metric_details": report["metric_details"],
    }
