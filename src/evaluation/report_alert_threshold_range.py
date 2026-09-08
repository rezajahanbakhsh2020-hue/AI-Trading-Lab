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


def is_alert_count_in_range(
    alert_count: int,
    lower_bound: int,
    upper_bound: int,
) -> bool:
    """Return whether an alert count is inside an inclusive range."""
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

    return lower_bound <= alert_count <= upper_bound


def calculate_range_width(
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Calculate the inclusive width of an alert-count range."""
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return upper_bound - lower_bound


def count_alerts_in_range(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count snapshots whose alert count is inside the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return sum(
        is_alert_count_in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
        for item in history
    )


def find_alert_range_positions(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> list[int]:
    """Return positions whose alert count is inside the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return [
        index
        for index, item in enumerate(history)
        if is_alert_count_in_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )
    ]


def calculate_range_coverage(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of snapshots inside the alert range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    if not history:
        return 0.0

    count = count_alerts_in_range(
        history,
        lower_bound,
        upper_bound,
    )

    return count / len(history)


def build_threshold_range_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete alert threshold-range summary."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    count = count_alerts_in_range(
        history,
        lower_bound,
        upper_bound,
    )

    positions = find_alert_range_positions(
        history,
        lower_bound,
        upper_bound,
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "range_width": calculate_range_width(
            lower_bound,
            upper_bound,
        ),
        "in_range_count": count,
        "out_of_range_count": len(history) - count,
        "coverage": calculate_range_coverage(
            history,
            lower_bound,
            upper_bound,
        ),
        "in_range_positions": positions,
    }
