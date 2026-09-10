from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.live_decision_record import (
    validate_live_decision_record,
)


def save_live_decision_history(
    history: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """
    Persist validated live decision history as JSON.

    Recorded trading values are stored exactly as supplied and are not
    recalculated.
    """
    if isinstance(history, (str, bytes)) or not isinstance(
        history, Iterable
    ):
        raise TypeError("history must be iterable")

    records: list[dict[str, Any]] = []

    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each history item must be a mapping")

        record = dict(item)
        validate_live_decision_record(record)
        records.append(record)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return target


def load_live_decision_history(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load and validate persisted live decision history from JSON.
    """
    target = Path(path)

    if not target.exists():
        raise FileNotFoundError(
            f"decision history file not found: {target}"
        )

    raw = json.loads(
        target.read_text(encoding="utf-8")
    )

    if not isinstance(raw, list):
        raise ValueError("stored decision history must be a list")

    records: list[dict[str, Any]] = []

    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(
                "each stored decision record must be an object"
            )

        record = dict(item)
        validate_live_decision_record(record)
        records.append(record)

    return records


def append_live_decision_to_store(
    record: Mapping[str, Any],
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Append one validated decision record to persistent history.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    new_record = dict(record)
    validate_live_decision_record(new_record)

    target = Path(path)

    if target.exists():
        history = load_live_decision_history(target)
    else:
        history = []

    history.append(new_record)
    save_live_decision_history(history, target)

    return history
