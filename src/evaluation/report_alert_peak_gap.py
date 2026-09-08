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


def calculate_peak_alert_count(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the maximum alert count observed.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    return max(
        item["alert_count"]
        for item in history
    )


def find_peak_positions(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Return zero-based positions where the peak alert count occurs.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)

    return [
        index
        for index, item in enumerate(history)
        if item["alert_count"] == peak
    ]


def calculate_peak_gaps(
    history: list[Mapping[str, Any]],
) -> list[int]:
    """
    Calculate the number of snapshots between consecutive peak
    alert observations.
    """
    positions = find_peak_positions(history)

    if len(positions) < 2:
        return []

    return [
        current - previous - 1
        for previous, current in zip(
            positions,
            positions[1:],
        )
    ]


def calculate_longest_peak_gap(
    history: list[Mapping[str, Any]],
) -> int:
    """
    Calculate the longest gap between consecutive peak observations.
    """
    gaps = calculate_peak_gaps(history)

    if not gaps:
        return 0

    return max(gaps)


def calculate_average_peak_gap(
    history: list[Mapping[str, Any]],
) -> float:
    """
    Calculate the average gap between consecutive peak observations.
    """
    gaps = calculate_peak_gaps(history)

    if not gaps:
        return 0.0

    return sum(gaps) / len(gaps)


def build_alert_peak_gap_summary(
    history: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete summary of gaps between peak alert observations.
    """
    _validate_history(history)

    if not history:
        raise ValueError(
            "No alert history available."
        )

    peak = calculate_peak_alert_count(history)
    positions = find_peak_positions(history)
    gaps = calculate_peak_gaps(history)

    return {
        "snapshot_count": len(history),
        "peak_alert_count": peak,
        "peak_occurrence_count": len(positions),
        "peak_positions": positions,
        "peak_gap_count": len(gaps),
        "peak_gaps": gaps,
        "average_peak_gap": calculate_average_peak_gap(history),
        "longest_peak_gap": calculate_longest_peak_gap(history),
    }
