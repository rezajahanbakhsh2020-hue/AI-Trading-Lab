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


def calculate_range_stability(
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

    inside_count = sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )

    return inside_count / len(history)


def calculate_range_instability(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of snapshots outside the alert range."""
    return 1.0 - calculate_range_stability(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_stable_snapshot_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count snapshots inside the alert range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )


def calculate_unstable_snapshot_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count snapshots outside the alert range."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    return len(history) - calculate_stable_snapshot_count(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_stability_score(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Return a stability score from 0.0 to 1.0."""
    return calculate_range_stability(
        history,
        lower_bound,
        upper_bound,
    )


def build_range_stability_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete alert-range stability summary."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    stable_count = calculate_stable_snapshot_count(
        history,
        lower_bound,
        upper_bound,
    )
    unstable_count = len(history) - stable_count

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "stable_snapshot_count": stable_count,
        "unstable_snapshot_count": unstable_count,
        "stability": (
            stable_count / len(history)
            if history
            else 0.0
        ),
        "instability": (
            unstable_count / len(history)
            if history
            else 0.0
        ),
        "stability_score": (
            stable_count / len(history)
            if history
            else 0.0
        ),
    }
