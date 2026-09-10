from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.live_decision_record import (
    build_live_decision_record,
    validate_live_decision_record,
)


def append_live_decision(
    history: Iterable[Mapping[str, Any]] | None,
    snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Append one live decision snapshot to decision history.

    Existing history entries are copied. The supplied snapshot is converted
    into a normalized decision record and validated before being appended.
    """
    if history is None:
        records: list[dict[str, Any]] = []
    else:
        if isinstance(history, (str, bytes)) or not isinstance(
            history, Iterable
        ):
            raise TypeError("history must be iterable or None")

        records = []
        for item in history:
            if not isinstance(item, Mapping):
                raise TypeError("each history item must be a mapping")

            record = dict(item)
            validate_live_decision_record(record)
            records.append(record)

    record = build_live_decision_record(snapshot)
    validate_live_decision_record(record)
    records.append(record)

    return records


def build_live_decision_history(
    snapshots: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert multiple live snapshots into normalized decision history.
    """
    if isinstance(snapshots, (str, bytes)) or not isinstance(
        snapshots, Iterable
    ):
        raise TypeError("snapshots must be iterable")

    history: list[dict[str, Any]] = []

    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise TypeError("each snapshot must be a mapping")

        record = build_live_decision_record(snapshot)
        validate_live_decision_record(record)
        history.append(record)

    return history


def get_latest_live_decision(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """
    Return a copy of the latest decision record, or None for empty history.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    latest: dict[str, Any] | None = None

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each history item must be a mapping")

        record = dict(item)
        validate_live_decision_record(record)
        latest = record

    return latest
