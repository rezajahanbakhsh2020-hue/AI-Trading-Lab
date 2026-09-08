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


def is_outside_range(
    alert_count: int,
    lower_bound: int,
    upper_bound: int,
) -> bool:
    """Return whether an alert count is outside an inclusive range."""
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

    return (
        alert_count < lower_bound
        or alert_count > upper_bound
    )


def find_outside_range_streaks(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> list[int]:
    """Return lengths of consecutive outside-range streaks."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    streaks: list[int] = []
    current_streak = 0

    for item in history:
        outside = is_outside_range(
            item["alert_count"],
            lower_bound,
            upper_bound,
        )

        if outside:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = 0

    if current_streak > 0:
        streaks.append(current_streak)

    return streaks


def count_outside_range_streaks(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Count consecutive outside-range streaks."""
    return len(
        find_outside_range_streaks(
            history,
            lower_bound,
            upper_bound,
        )
    )


def calculate_total_outside_range_duration(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Calculate total snapshots spent outside the range."""
    streaks = find_outside_range_streaks(
        history,
        lower_bound,
        upper_bound,
    )

    return sum(streaks)


def calculate_mean_outside_range_streak(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> float:
    """Calculate the mean length of outside-range streaks."""
    streaks = find_outside_range_streaks(
        history,
        lower_bound,
        upper_bound,
    )

    if not streaks:
        return 0.0

    return sum(streaks) / len(streaks)


def calculate_longest_outside_range_streak(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> int:
    """Calculate the longest consecutive outside-range streak."""
    streaks = find_outside_range_streaks(
        history,
        lower_bound,
        upper_bound,
    )

    if not streaks:
        return 0

    return max(streaks)


def build_outside_range_streak_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete outside-range streak summary."""
    _validate_history(history)
    _validate_bounds(
        lower_bound,
        upper_bound,
    )

    streaks = find_outside_range_streaks(
        history,
        lower_bound,
        upper_bound,
    )

    return {
        "snapshot_count": len(history),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "streak_count": len(streaks),
        "streaks": streaks,
        "total_outside_duration": sum(streaks),
        "mean_streak": (
            sum(streaks) / len(streaks)
            if streaks
            else 0.0
        ),
        "longest_streak": (
            max(streaks)
            if streaks
            else 0
        ),
    }
