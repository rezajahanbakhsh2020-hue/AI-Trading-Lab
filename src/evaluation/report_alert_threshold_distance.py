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


def _validate_threshold(threshold: int) -> None:
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
    ):
        raise TypeError(
            "threshold must be an integer."
        )

    if threshold < 0:
        raise ValueError(
            "threshold must not be negative."
        )


def calculate_distance(
    alert_count: int,
    threshold: int,
) -> int:
    """Calculate the absolute distance from an alert count to a threshold."""
    if (
        not isinstance(alert_count, int)
        or isinstance(alert_count, bool)
    ):
        raise TypeError(
            "alert_count must be an integer."
        )

    if alert_count < 0:
        raise ValueError(
            "alert_count must not be negative."
        )

    _validate_threshold(threshold)

    return abs(alert_count - threshold)


def calculate_total_distance(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """Calculate the total absolute distance across all snapshots."""
    _validate_history(history)
    _validate_threshold(threshold)

    return sum(
        calculate_distance(
            item["alert_count"],
            threshold,
        )
        for item in history
    )


def calculate_average_distance(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """Calculate the average absolute distance across all snapshots."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return 0.0

    return calculate_total_distance(
        history,
        threshold,
    ) / len(history)


def find_closest_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> list[int]:
    """Return positions with the minimum distance to the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return []

    distances = [
        calculate_distance(
            item["alert_count"],
            threshold,
        )
        for item in history
    ]

    minimum_distance = min(distances)

    return [
        index
        for index, distance in enumerate(distances)
        if distance == minimum_distance
    ]


def find_farthest_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> list[int]:
    """Return positions with the maximum distance from the threshold."""
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        return []

    distances = [
        calculate_distance(
            item["alert_count"],
            threshold,
        )
        for item in history
    ]

    maximum_distance = max(distances)

    return [
        index
        for index, distance in enumerate(distances)
        if distance == maximum_distance
    ]


def build_threshold_distance_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> dict[str, Any]:
    """Build a complete threshold-distance summary."""
    _validate_history(history)
    _validate_threshold(threshold)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "total_distance": calculate_total_distance(
            history,
            threshold,
        ),
        "average_distance": calculate_average_distance(
            history,
            threshold,
        ),
        "closest_positions": find_closest_positions(
            history,
            threshold,
        ),
        "farthest_positions": find_farthest_positions(
            history,
            threshold,
        ),
    }
