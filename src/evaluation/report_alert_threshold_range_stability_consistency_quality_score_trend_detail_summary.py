from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail import (
    build_quality_score_trend_detail,
)


def build_quality_score_trend_detail_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a summarized quality-score trend detail report."""

    report = build_quality_score_trend_detail(
        history,
        lower_bound,
        upper_bound,
    )

    details = report["details"]

    if details:
        best_quality = max(
            details,
            key=lambda item: item["quality_score"],
        )
        worst_quality = min(
            details,
            key=lambda item: item["quality_score"],
        )

        best_stability = max(
            details,
            key=lambda item: item["stability_score"],
        )
        worst_stability = min(
            details,
            key=lambda item: item["stability_score"],
        )

        best_consistency = max(
            details,
            key=lambda item: item["consistency_score"],
        )
        worst_consistency = min(
            details,
            key=lambda item: item["consistency_score"],
        )
    else:
        best_quality = None
        worst_quality = None
        best_stability = None
        worst_stability = None
        best_consistency = None
        worst_consistency = None

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
        "best_quality_snapshot": (
            None
            if best_quality is None
            else best_quality["snapshot_index"]
        ),
        "best_quality_score": (
            None
            if best_quality is None
            else best_quality["quality_score"]
        ),
        "worst_quality_snapshot": (
            None
            if worst_quality is None
            else worst_quality["snapshot_index"]
        ),
        "worst_quality_score": (
            None
            if worst_quality is None
            else worst_quality["quality_score"]
        ),
        "best_stability_snapshot": (
            None
            if best_stability is None
            else best_stability["snapshot_index"]
        ),
        "best_stability_score": (
            None
            if best_stability is None
            else best_stability["stability_score"]
        ),
        "worst_stability_snapshot": (
            None
            if worst_stability is None
            else worst_stability["snapshot_index"]
        ),
        "worst_stability_score": (
            None
            if worst_stability is None
            else worst_stability["stability_score"]
        ),
        "best_consistency_snapshot": (
            None
            if best_consistency is None
            else best_consistency["snapshot_index"]
        ),
        "best_consistency_score": (
            None
            if best_consistency is None
            else best_consistency["consistency_score"]
        ),
        "worst_consistency_snapshot": (
            None
            if worst_consistency is None
            else worst_consistency["snapshot_index"]
        ),
        "worst_consistency_score": (
            None
            if worst_consistency is None
            else worst_consistency["consistency_score"]
        ),
        "details": details,
    }
