from __future__ import annotations

from typing import Any, Mapping


def calculate_alert_streaks(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Calculate lengths of consecutive alert streaks.

    A snapshot with alert_count > 0 is considered active.
    Each returned value represents one consecutive active-alert streak.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    streaks: list[int] = []
    current_streak = 0

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

        if value > 0:
            current_streak += 1
        elif current_streak > 0:
            streaks.append(current_streak)
            current_streak = 0

    if current_streak > 0:
        streaks.append(current_streak)

    return streaks


def calculate_longest_alert_streak(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Return the longest consecutive alert streak.
    """
    streaks = calculate_alert_streaks(history)

    if not streaks:
        return 0

    return max(streaks)


def calculate_average_alert_streak(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average length of alert streaks.

    Returns zero when no alert streak exists.
    """
    streaks = calculate_alert_streaks(history)

    if not streaks:
        return 0.0

    return sum(streaks) / len(streaks)


def count_alert_streaks(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count the number of consecutive alert streaks.
    """
    return len(
        calculate_alert_streaks(history)
    )


def calculate_single_snapshot_streaks(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Count alert streaks that lasted exactly one snapshot.
    """
    streaks = calculate_alert_streaks(history)

    return sum(
        1
        for streak in streaks
        if streak == 1
    )


def build_alert_streak_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete alert-streak summary.
    """
    streaks = calculate_alert_streaks(history)

    total_alert_snapshots = sum(streaks)
    streak_count = len(streaks)

    average_streak = (
        total_alert_snapshots / streak_count
        if streak_count
        else 0.0
    )

    longest_streak = (
        max(streaks)
        if streaks
        else 0
    )

    single_snapshot_streaks = sum(
        1
        for streak in streaks
        if streak == 1
    )

    return {
        "snapshot_count": len(history),
        "streak_count": streak_count,
        "streaks": streaks,
        "total_alert_snapshots": total_alert_snapshots,
        "longest_streak": longest_streak,
        "average_streak": average_streak,
        "single_snapshot_streaks": single_snapshot_streaks,
    }
