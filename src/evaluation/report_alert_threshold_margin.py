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


def _validate_margin(margin: int) -> None:
    if (
        not isinstance(margin, int)
        or isinstance(margin, bool)
    ):
        raise TypeError(
            "margin must be an integer."
        )

    if margin < 0:
        raise ValueError(
            "margin must not be negative."
        )


def calculate_margin(
    alert_count: int,
    threshold: int,
) -> int:
    """Calculate the signed margin relative to the threshold."""
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

    return alert_count - threshold


def is_above_margin(
    alert_count: int,
    threshold: int,
    margin: int,
) -> bool:
    """Return whether alert count is at least margin above threshold."""
    _validate_margin(margin)

    return calculate_margin(
        alert_count,
        threshold,
    ) >= margin


def is_below_margin(
    alert_count: int,
    threshold: int,
    margin: int,
) -> bool:
    """Return whether alert count is at least margin below threshold."""
    _validate_margin(margin)

    return calculate_margin(
        alert_count,
        threshold,
    ) <= -margin


def count_above_margin(
    history: list[Mapping[str, Any]],
    threshold: int,
    margin: int,
) -> int:
    """Count snapshots at or above the upper margin."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_margin(margin)

    return sum(
        is_above_margin(
            item["alert_count"],
            threshold,
            margin,
        )
        for item in history
    )


def count_below_margin(
    history: list[Mapping[str, Any]],
    threshold: int,
    margin: int,
) -> int:
    """Count snapshots at or below the lower margin."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_margin(margin)

    return sum(
        is_below_margin(
            item["alert_count"],
            threshold,
            margin,
        )
        for item in history
    )


def find_above_margin_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    margin: int,
) -> list[int]:
    """Return positions at or above the upper margin."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_margin(margin)

    return [
        index
        for index, item in enumerate(history)
        if is_above_margin(
            item["alert_count"],
            threshold,
            margin,
        )
    ]


def find_below_margin_positions(
    history: list[Mapping[str, Any]],
    threshold: int,
    margin: int,
) -> list[int]:
    """Return positions at or below the lower margin."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_margin(margin)

    return [
        index
        for index, item in enumerate(history)
        if is_below_margin(
            item["alert_count"],
            threshold,
            margin,
        )
    ]


def build_threshold_margin_summary(
    history: list[Mapping[str, Any]],
    threshold: int,
    margin: int,
) -> dict[str, Any]:
    """Build a complete threshold-margin summary."""
    _validate_history(history)
    _validate_threshold(threshold)
    _validate_margin(margin)

    above_count = count_above_margin(
        history,
        threshold,
        margin,
    )
    below_count = count_below_margin(
        history,
        threshold,
        margin,
    )

    return {
        "snapshot_count": len(history),
        "threshold": threshold,
        "margin": margin,
        "above_margin_count": above_count,
        "below_margin_count": below_count,
        "above_margin_positions": find_above_margin_positions(
            history,
            threshold,
            margin,
        ),
        "below_margin_positions": find_below_margin_positions(
            history,
            threshold,
            margin,
        ),
    }
