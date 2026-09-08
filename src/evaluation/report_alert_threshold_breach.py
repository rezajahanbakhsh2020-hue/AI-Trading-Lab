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


def count_threshold_breaches(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """
    Count observations whose alert count is above the threshold.
    """
    _validate_history(history)
    _validate_threshold(threshold)

    return sum(
        1
        for item in history
        if item["alert_count"] > threshold
    )


def find_threshold_breach_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> list[int]:
    """
    Return zero-based positions where the alert count exceeds the threshold.
    """
    _validate_history(history)
    _validate_threshold(threshold)

    return [
        index
        for index, item in enumerate(history)
        if item["alert_count"] > threshold
    ]


def calculate_threshold_breach_ratio(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> float:
    """
    Calculate the proportion of observations above the threshold.
    """
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return count_threshold_breaches(
        history,
        threshold,
    ) / len(history)


def calculate_longest_breach_streak(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> int:
    """
    Calculate the longest consecutive streak above the threshold.
    """
    _validate_history(history)
    _validate_threshold(threshold)

    longest = 0
    current = 0

    for item in history:
        if item["alert_count"] > threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def build_alert_threshold_breach_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
) -> dict[str, Any]:
    """
    Build a complete threshold-breach summary.
    """
    _validate_history(history)
    _validate_threshold(threshold)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    breach_count = count_threshold_breaches(
        history,
        threshold,
    )
    ratio = calculate_threshold_breach_ratio(
        history,
        threshold,
    )

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "breach_count": breach_count,
        "breach_positions": find_threshold_breach_positions(
            history,
            threshold,
        ),
        "breach_ratio": ratio,
        "breach_percentage": ratio * 100.0,
        "longest_breach_streak": calculate_longest_breach_streak(
            history,
            threshold,
        ),
    }
