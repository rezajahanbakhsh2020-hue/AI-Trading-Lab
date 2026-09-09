from __future__ import annotations

from typing import Any, Mapping


def _validate_history(
    history: list[Mapping[str, Any]],
) -> None:
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("Each history item must be a mapping.")

        value = item.get("alert_count")

        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                "Each history item must contain an integer alert_count."
            )

        if value < 0:
            raise ValueError("alert_count must not be negative.")


def _validate_bounds(
    lower_bound: int,
    upper_bound: int,
) -> None:
    for name, value in (
        ("lower_bound", lower_bound),
        ("upper_bound", upper_bound),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} must not be negative.")

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound must not be greater than upper_bound."
        )


def calculate_consistent_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count observations inside the inclusive target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    return sum(
        lower_bound <= item["alert_count"] <= upper_bound
        for item in history
    )


def calculate_inconsistent_count(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count observations outside the inclusive target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    return len(history) - calculate_consistent_count(
        history,
        lower_bound,
        upper_bound,
    )


def calculate_consistency_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations inside the target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    return calculate_consistent_count(
        history,
        lower_bound,
        upper_bound,
    ) / len(history)


def calculate_inconsistency_ratio(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the fraction of observations outside the target range."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    if not history:
        return 0.0

    return calculate_inconsistent_count(
        history,
        lower_bound,
        upper_bound,
    ) / len(history)


def build_consistency_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete range-consistency summary."""
    _validate_history(history)
    _validate_bounds(lower_bound, upper_bound)

    consistent_count = calculate_consistent_count(
        history,
        lower_bound,
        upper_bound,
    )
    inconsistent_count = len(history) - consistent_count

    consistency_ratio = (
        consistent_count / len(history)
        if history
        else 0.0
    )

    inconsistency_ratio = (
        inconsistent_count / len(history)
        if history
        else 0.0
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "consistent_count": consistent_count,
        "inconsistent_count": inconsistent_count,
        "consistency_ratio": consistency_ratio,
        "inconsistency_ratio": inconsistency_ratio,
        "consistency_percentage": consistency_ratio * 100.0,
        "inconsistency_percentage": inconsistency_ratio * 100.0,
    }
