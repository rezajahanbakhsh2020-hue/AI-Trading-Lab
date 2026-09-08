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


def calculate_consistent_snapshot_count(
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
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )


def calculate_inconsistent_snapshot_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count snapshots whose alert count is outside the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    consistent_count = calculate_consistent_snapshot_count(
        history,
        lower_bound,
        upper_bound,
    )

    return len(history) - consistent_count


def calculate_consistency_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of snapshots inside the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    if not history:
        return 0.0

    return (
        calculate_consistent_snapshot_count(
            history,
            lower_bound,
            upper_bound,
        )
        / len(history)
    )


def calculate_inconsistency_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of snapshots outside the range."""
    return 1.0 - calculate_consistency_ratio(
        history,
        lower_bound,
        upper_bound,
    )


def find_inconsistent_positions(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> list[int]:
    """Return positions of snapshots outside the range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return [
        index
        for index, item in enumerate(history)
        if (
            item["alert_count"] < lower_bound
            or item["alert_count"] > upper_bound
        )
    ]


def build_consistency_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete alert-range consistency summary."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    consistent_count = calculate_consistent_snapshot_count(
        history,
        lower_bound,
        upper_bound,
    )
    inconsistent_count = len(history) - consistent_count

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "consistent_snapshot_count": consistent_count,
        "inconsistent_snapshot_count": inconsistent_count,
        "consistency_ratio": (
            consistent_count / len(history)
            if history
            else 0.0
        ),
        "inconsistency_ratio": (
            inconsistent_count / len(history)
            if history
            else 0.0
        ),
        "inconsistent_positions": find_inconsistent_positions(
            history,
            lower_bound,
            upper_bound,
        ),
    }
