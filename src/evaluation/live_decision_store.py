from __future__ import annotations

import json
import os
import tempfile
import fcntl
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleError,
    LiveDecisionLifecycleState,
    transition_live_decision,
    validate_live_decision_lifecycle,
)
from src.evaluation.live_decision_record import (
    build_live_decision_record,
    validate_live_decision_record,
)
from src.evaluation.live_production_decision import Direction

DEFAULT_STORE_PATH = Path("results/live/decision_history.json")


@contextmanager
def _file_lock(lock_path: Path):
    """
    Inter-process file lock using fcntl.flock for Linux/Unix environments.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _atomic_write_json(records: list[dict[str, Any]], target: Path) -> None:
    """
    Crash-safe atomic JSON write to target path.

    Writes to a temporary file in the same directory, flushes and fsyncs,
    and replaces the target atomically using os.replace(). Cleans up the
    temporary file on error.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=target.parent,
        prefix=f"{target.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_path_str)
    try:
        content = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


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
    _atomic_write_json(records, target)

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
    lock_path = target.parent / f"{target.name}.lock"

    with _file_lock(lock_path):
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
        _atomic_write_json(history, target)

        return history


def persist_canonical_live_decision(
    canonical_decision: CanonicalLiveDecision,
    path: str | Path,
    enforce_idempotency: bool = True,
    actor: str = "live_decision_store",
    timestamp_utc: Optional[str] = None,
) -> CanonicalLiveDecision:
    """Enforced persistence lifecycle boundary for a CanonicalLiveDecision.

    Requires current state PRESENTABLE (or PERSISTED for identical replay).
    Transitions artifact to PERSISTED, validates, appends to store, and returns the canonical PERSISTED artifact.
    """
    if not isinstance(canonical_decision, CanonicalLiveDecision):
        raise TypeError(f"canonical_decision must be a CanonicalLiveDecision, got {type(canonical_decision).__name__}")

    validate_live_decision_lifecycle(canonical_decision)

    if canonical_decision.current_state == LiveDecisionLifecycleState.PERSISTED:
        target = Path(path)
        if target.exists():
            history = load_live_decision_history(target)
            for existing in history:
                if existing.get("canonical_live_decision_fingerprint") == canonical_decision.canonical_live_decision_fingerprint:
                    return canonical_decision
                if existing.get("decision_id") == canonical_decision.decision_id:
                    raise ValueError(
                        f"Conflicting replay detected for decision_id '{canonical_decision.decision_id}': "
                        f"canonical_live_decision_fingerprint mismatch."
                    )
        raise LiveDecisionLifecycleError("Canonical decision is marked PERSISTED but missing from persistence store.")

    if canonical_decision.current_state != LiveDecisionLifecycleState.PRESENTABLE:
        raise LiveDecisionLifecycleError(
            f"Cannot persist canonical live decision in state '{canonical_decision.current_state.value}': expected state 'PRESENTABLE'."
        )

    # Execute transition PRESENTABLE -> PERSISTED
    persisted_decision = transition_live_decision(
        canonical_decision,
        LiveDecisionLifecycleState.PERSISTED,
        actor=actor,
        timestamp_utc=timestamp_utc,
        reason="canonical_persistence_boundary",
    )

    validate_live_decision_lifecycle(persisted_decision)

    receipt = persisted_decision.authorization_receipt
    dec = persisted_decision.decision
    signal = persisted_decision.signal
    risk = persisted_decision.risk_levels

    record = build_live_decision_record({
        "timestamp": dec.market_timestamp,
        "symbol": receipt.symbol,
        "interval": receipt.timeframe,
        "signal": 1 if dec.direction == Direction.BUY else 0,
        "signal_label": dec.direction.value,
        "trend": "UP" if dec.direction == Direction.BUY else "NEUTRAL",
        "strategy": receipt.strategy_name,
        "entry_price": risk.entry_price,
        "stop_loss": risk.stop_loss,
        "take_profit": risk.tp2 if risk.tp2 is not None else risk.tp1,
        "risk_reward_ratio": risk.risk_reward_ratio,
        "stability_score": dec.confidence,
        "decision_id": dec.decision_id,
        "signal_id": signal.signal_id,
        "canonical_live_decision_fingerprint": persisted_decision.canonical_live_decision_fingerprint,
        "current_lifecycle_state": persisted_decision.current_state.value,
        "transition_history": [tr.as_dict() for tr in persisted_decision.transition_history],
        "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
        "authorization_policy_version": receipt.authorization_policy_version,
        "authorized_at_utc": receipt.authorized_at_utc,
        "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
        "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
        "candidate_id": receipt.candidate_id,
        "strategy_name": receipt.strategy_name,
        "strategy_version": receipt.strategy_version,
    })

    append_live_decision_to_store(record, path, enforce_idempotency=enforce_idempotency)

    return persisted_decision
