"""Persist and load an append-only history of live trading snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_HISTORY_PATH = Path("results/live/live_snapshot_history.jsonl")


def append_live_snapshot(
    snapshot: Mapping[str, Any],
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> Path:
    """Append one JSON-serializable live snapshot to a JSONL history file."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serialized = json.dumps(
            dict(snapshot),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"snapshot is not JSON serializable: {exc}"
        ) from exc

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")

    return output_path


def load_live_snapshot_history(
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> list[dict[str, Any]]:
    """Load all valid snapshot records from a JSONL history file."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Snapshot history file not found: {input_path}"
        )

    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid snapshot history JSON at line {line_number}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Each snapshot history record must contain an object."
            )

        records.append(payload)

    return records


def latest_live_snapshot(
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> dict[str, Any]:
    """Return the most recently appended live snapshot."""
    records = load_live_snapshot_history(path)

    if not records:
        raise ValueError("Snapshot history contains no records.")

    return records[-1]
