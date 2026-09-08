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


def _validate_tolerance(tolerance: int) -> None:
    if (
        not isinstance(tolerance, int)
        or isinstance(tolerance, bool)
    ):
        raise TypeError(
            "tolerance must be an integer."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance must not be negative."
        )


def calculate_distance(
    alert_count: int,
    threshold: int,
) -> int:
    """Calculate absolute distance from the threshold."""
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


def is_within_tolerance(
    alert_count: int,
    threshold: int,
    tolerance: int,
) -> bool:
    """Return whether an alert count is within the allowed tolerance."""
    _validate_tolerance(tolerance)

    return calculate_distance(
        alert_count,
        threshold,
    ) <= tolerance


def count_within_tolerance(
    history: list[Mapping[str, Any]],
    threshold: int,
    tolerance: int,
) -> int:
    """Count snapshots whose alert count is within tolerance."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_tolerance(tolerance)

    return sum(
        is_within_tolerance(
            item["alert_count"],
            threshold,
            tolerance,
        )
        for item in history
    )


def calculate_proximity_ratio(
    history: list[Mapping[str, Any]],
    threshold: int,
    tolerance: int,
) -> float:
    """Calculate the proportion of snapshots within tolerance."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_tolerance(tolerance)

    if not history:
        return 0.0

    return count_within_tolerance(
        history,
        threshold,
        tolerance,
    ) / len(history)


def find_within_tolerance_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    tolerance: int,
) -> list[int]:
    """Return positions whose alert counts are within tolerance."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_tolerance(tolerance)

    return [
        index
        for index, item in enumerate(history)
        if is_within_tolerance(
            item["alert_count"],
            threshold,
            tolerance,
        )
    ]


def find_outside_tolerance_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    tolerance: int,
) -> list[int]:
    """Return positions whose alert counts exceed tolerance."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_tolerance(tolerance)

    return [
        index
        for index, item in enumerate(history)
        if not is_within_tolerance(
            item["alert_count"],
            threshold,
            tolerance,
        )
    ]


def build_threshold_proximity_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
    tolerance: int,
) -> dict[str, Any]:
    """Build a complete threshold-proximity summary."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_tolerance(tolerance)

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "tolerance": tolerance,
        "within_tolerance_count": count_within_tolerance(
            history,
            threshold,
            tolerance,
        ),
        "proximity_ratio": calculate_proximity_ratio(
            history,
            threshold,
            tolerance,
        ),
        "within_tolerance_positions": find_within_tolerance_positions(
            history,
            threshold,
            tolerance,
        ),
        "outside_tolerance_positions": find_outside_tolerance_positions(
            history,
            threshold,
            tolerance,
        ),
    }
