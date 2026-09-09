from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary import (
    build_quality_score_trend_detail_summary,
)


def build_quality_score_trend_detail_summary_extrema(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build an extrema-focused quality-score trend summary."""

    report = build_quality_score_trend_detail_summary(
        history,
        lower_bound,
        upper_bound,
    )

    details = report["details"]

    if not details:
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
            "quality_range": 0.0,
            "stability_range": 0.0,
            "consistency_range": 0.0,
            "quality_peak_snapshot": None,
            "quality_floor_snapshot": None,
            "stability_peak_snapshot": None,
            "stability_floor_snapshot": None,
            "consistency_peak_snapshot": None,
            "consistency_floor_snapshot": None,
            "details": [],
        }

    quality_peak = max(
        details,
        key=lambda item: item["quality_score"],
    )
    quality_floor = min(
        details,
        key=lambda item: item["quality_score"],
    )

    stability_peak = max(
        details,
        key=lambda item: item["stability_score"],
    )
    stability_floor = min(
        details,
        key=lambda item: item["stability_score"],
    )

    consistency_peak = max(
        details,
        key=lambda item: item["consistency_score"],
    )
    consistency_floor = min(
        details,
        key=lambda item: item["consistency_score"],
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
        "quality_range": (
            quality_peak["quality_score"]
            - quality_floor["quality_score"]
        ),
        "stability_range": (
            stability_peak["stability_score"]
            - stability_floor["stability_score"]
        ),
        "consistency_range": (
            consistency_peak["consistency_score"]
            - consistency_floor["consistency_score"]
        ),
        "quality_peak_snapshot": quality_peak["snapshot_index"],
        "quality_floor_snapshot": quality_floor["snapshot_index"],
        "stability_peak_snapshot": stability_peak["snapshot_index"],
        "stability_floor_snapshot": stability_floor["snapshot_index"],
        "consistency_peak_snapshot": consistency_peak["snapshot_index"],
        "consistency_floor_snapshot": consistency_floor["snapshot_index"],
        "details": details,
    }
