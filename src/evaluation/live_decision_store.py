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
    enforce_idempotency: bool = True,
) -> list[dict[str, Any]]:
    """
    Append one validated decision record to persistent history.

    If enforce_idempotency is True, checks existing records for matching decision_id or signal_id:
      - If existing record has identical content -> idempotent no-op (or return existing history).
      - If existing record has conflicting content -> fail closed raising ValueError.
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

    if enforce_idempotency:
        new_dec_id = new_record.get("decision_id")
        new_sig_id = new_record.get("signal_id")

        for existing in history:
            ext_dec_id = existing.get("decision_id")
            ext_sig_id = existing.get("signal_id")

            id_match = (new_dec_id and new_dec_id == ext_dec_id) or (new_sig_id and new_sig_id == ext_sig_id)
            if id_match:
                # Compare authorization fingerprint explicitly for conflict detection
                new_raw_auth = new_record.get("runtime_authorization_fingerprint")
                ext_raw_auth = existing.get("runtime_authorization_fingerprint")

                new_auth_fp = str(new_raw_auth).strip() if new_raw_auth is not None and str(new_raw_auth).strip() != "" else None
                ext_auth_fp = str(ext_raw_auth).strip() if ext_raw_auth is not None and str(ext_raw_auth).strip() != "" else None

                existing_has_auth = ext_auth_fp is not None
                new_has_auth = new_auth_fp is not None

                if existing_has_auth != new_has_auth:
                    raise ValueError(
                        f"Conflicting replay detected for decision/signal identity '{new_dec_id or new_sig_id}': "
                        "authorization lineage presence mismatch."
                    )

                if existing_has_auth and new_has_auth and ext_auth_fp != new_auth_fp:
                    raise ValueError(
                        f"Conflicting replay detected for decision/signal identity '{new_dec_id or new_sig_id}': "
                        f"runtime_authorization_fingerprint mismatch ('{ext_auth_fp}' vs '{new_auth_fp}')."
                    )

                # Compare canonical live decision fingerprint explicitly for conflict detection
                new_cld_fp = new_record.get("canonical_live_decision_fingerprint")
                ext_cld_fp = existing.get("canonical_live_decision_fingerprint")

                new_cld_str = str(new_cld_fp).strip() if new_cld_fp is not None and str(new_cld_fp).strip() != "" else None
                ext_cld_str = str(ext_cld_fp).strip() if ext_cld_fp is not None and str(ext_cld_fp).strip() != "" else None

                if (ext_cld_str is not None) != (new_cld_str is not None):
                    raise ValueError(
                        f"Conflicting replay detected for decision/signal identity '{new_dec_id or new_sig_id}': "
                        "canonical_live_decision_fingerprint presence mismatch."
                    )

                if ext_cld_str is not None and new_cld_str is not None and ext_cld_str != new_cld_str:
                    raise ValueError(
                        f"Conflicting replay detected for decision/signal identity '{new_dec_id or new_sig_id}': "
                        f"canonical_live_decision_fingerprint mismatch ('{ext_cld_str}' vs '{new_cld_str}')."
                    )

                # Compare canonical content
                # Exclude runtime volatile timestamps if present
                keys_to_compare = [k for k in new_record if k not in ("recorded_at", "created_at")]
                match_all = True
                for k in keys_to_compare:
                    if existing.get(k) != new_record.get(k):
                        match_all = False
                        break

                if match_all:
                    # Idempotent replay: record already exists identically
                    return history
                else:
                    # Conflicting replay for same identity -> fail closed
                    raise ValueError(
                        f"Conflicting replay detected for decision/signal identity '{new_dec_id or new_sig_id}'. "
                        "Existing record differs from new record."
                    )

    history.append(new_record)
    save_live_decision_history(history, target)

    return history
