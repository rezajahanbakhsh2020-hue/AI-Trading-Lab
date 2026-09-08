from __future__ import annotations

from typing import Any, Mapping


def _validate_history(
    history: list[Mapping[str, Any]],
) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Each history item must be a mapping."
            )

        value = item.get("alert_count")

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise ValueError(
                "Each history item must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError(
                "alert_count must not be negative."
            )


def calculate_alert_clusters(
    history: list[Mapping[str, Any]],
    threshold: int = 1,
) -> int:
    """
    Count clusters of consecutive snapshots containing alerts.

    A snapshot belongs to an alert cluster when its alert_count
    is greater than or equal to threshold.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
    ):
        raise ValueError(
            "threshold must be an integer."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative."
        )

    values = [
        item["alert_count"]
        for item in history
    ]

    clusters = 0
    in_cluster = False

    for value in values:
        active = value >= threshold

        if active and not in_cluster:
            clusters += 1

        in_cluster = active

    return clusters


def calculate_clustered_snapshots(
    history: list[Mapping[str, Any]],
    threshold: int = 1,
) -> int:
    """
    Count snapshots belonging to alert clusters.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
    ):
        raise ValueError(
            "threshold must be an integer."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative."
        )

    values = [
        item["alert_count"]
        for item in history
    ]

    return sum(
        value >= threshold
        for value in values
    )


def build_alert_clustering_summary(
    history: list[Mapping[str, Any]],
    threshold: int = 1,
) -> dict[str, Any]:
    """
    Build a complete alert-clustering summary.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return {
        "snapshot_count": len(history),
        "alert_clusters": calculate_alert_clusters(
            history,
            threshold,
        ),
        "clustered_snapshots": calculate_clustered_snapshots(
            history,
            threshold,
        ),
        "threshold": threshold,
    }
