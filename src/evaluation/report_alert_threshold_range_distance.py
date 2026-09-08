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


def _validate_bounds(
    lower_bound: int,
    upper_bound: int,
) -> None:
    for name, value in (
        ("lower_bound", lower_bound),
        ("upper_bound", upper_bound),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name} must not be negative."
            )

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not be greater than upper_bound."
        )


def calculate_range_distance(
    alert_count: int,
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Calculate distance from an inclusive alert-count range."""
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

    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    if alert_count < lower_bound:
        return lower_bound - alert_count

    if alert_count > upper_bound:
        return alert_count - upper_bound

    return 0


def calculate_total_range_distance(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Calculate total distance of all snapshots from the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return sum(
        calculate_range_distance(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    )


def calculate_mean_range_distance(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate mean distance of snapshots from the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    if not history:
        return 0.0

    total_distance = calculate_total_range_distance(
        history,
        lower_bound,
        upper_bound,
    )

    return total_distance / len(history)


def find_farthest_range_positions(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> list[int]:
    """Return positions with the greatest distance from the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    if not history:
        return []

    distances = [
        calculate_range_distance(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    ]

    maximum_distance = max(distances)

    return [
        index
        for index, distance in enumerate(distances)
        if distance == maximum_distance
    ]


def build_range_distance_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete range-distance summary."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    distances = [
        calculate_range_distance(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    ]

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "total_distance": sum(distances),
        "mean_distance": (
            sum(distances) / len(history)
            if history
            else 0.0
        ),
        "maximum_distance": (
            max(distances)
            if distances
            else 0
        ),
        "farthest_positions": find_farthest_range_positions(
            history,
            lower_bound,
            upper_bound,
        ),
    }
