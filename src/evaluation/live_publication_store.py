from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class PublicationIntegrityError(ValueError):
    """Raised when a conflicting publication replay or payload mismatch occurs."""


def save_publication_history(
    history: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Persist publication history list as JSON."""
    if isinstance(history, (str, bytes)) or not isinstance(history, Iterable):
        raise TypeError("history must be iterable")

    records: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, Mapping):
            raise TypeError("each publication item must be a mapping")
        records.append(dict(item))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_publication_history(path: str | Path) -> list[dict[str, Any]]:
    """Load publication history from JSON file."""
    target = Path(path)
    if not target.exists():
        return []

    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("stored publication history must be a list")

    return [dict(item) for item in raw if isinstance(item, Mapping)]


def append_publication_record(
    record: Mapping[str, Any],
    path: str | Path,
    enforce_idempotency: bool = True,
) -> list[dict[str, Any]]:
    """Append publication record to store with deterministic replay protection."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    new_record = dict(record)
    pub_id = new_record.get("publication_id") or new_record.get("event_id")
    if not pub_id or not str(pub_id).strip():
        raise ValueError("publication record missing publication_id")

    target = Path(path)
    history = load_publication_history(target)

    if enforce_idempotency and pub_id:
        for existing in history:
            ext_pub_id = existing.get("publication_id") or existing.get("event_id")
            if ext_pub_id == pub_id:
                # Compare canonical publication content
                payload_keys = ("publication_id", "signal_id", "decision_id", "symbol", "timeframe", "decision", "entry", "stop_loss", "tp1")
                match_all = True
                for k in payload_keys:
                    if existing.get(k) != new_record.get(k):
                        match_all = False
                        break

                if match_all:
                    # Identical replay: safe no-op
                    return history
                else:
                    # Conflicting replay for same identity -> fail closed
                    raise PublicationIntegrityError(
                        f"Conflicting publication replay detected for publication_id '{pub_id}'. "
                        "Existing publication record differs from new record."
                    )

    history.append(new_record)
    save_publication_history(history, target)
    return history
