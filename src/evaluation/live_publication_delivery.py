"""Durable Live Publication Delivery Receipt & Recovery Ledger for Project 2 Outbound Integration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

DEFAULT_DELIVERY_STORE_PATH = Path("results/live/delivery_history.json")


class DeliveryIntegrityError(ValueError):
    """Raised when a delivery receipt conflict, lineage mismatch, or invalid state transition occurs."""


class DeliveryStatus(str, Enum):
    """Explicit lifecycle status for live publication delivery attempts."""

    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    SKIPPED_DISABLED = "SKIPPED_DISABLED"
    SKIPPED_NO_TRADE = "SKIPPED_NO_TRADE"
    SKIPPED_STALE = "SKIPPED_STALE"
    DELIVERED = "DELIVERED"
    REJECTED = "REJECTED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_PERMANENT = "FAILED_PERMANENT"

    @property
    def is_delivered(self) -> bool:
        return self == DeliveryStatus.DELIVERED

    @property
    def is_skipped(self) -> bool:
        return self in (
            DeliveryStatus.SKIPPED_DISABLED,
            DeliveryStatus.SKIPPED_NO_TRADE,
            DeliveryStatus.SKIPPED_STALE,
        )

    @property
    def is_retryable(self) -> bool:
        return self == DeliveryStatus.FAILED_RETRYABLE


def _redact_secrets_from_text(text: Optional[str]) -> Optional[str]:
    """Ensure no secrets or authorization headers appear in log texts or error messages."""
    if not text:
        return text
    result = str(text)
    for kw in ("Bearer ", "api_key=", "X-API-Key="):
        if kw in result:
            lines = []
            for line in result.splitlines():
                if kw in line:
                    lines.append("[REDACTED_AUTHORIZATION_SECRET]")
                else:
                    lines.append(line)
            result = "\n".join(lines)
    return result


@dataclass(frozen=True)
class PublicationDeliveryReceipt:
    """Immutable, fingerprinted delivery evidence binding publication to decision lineage."""

    publication_id: str
    decision_id: str
    signal_id: str
    canonical_live_decision_fingerprint: str
    runtime_authorization_fingerprint: str
    candidate_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: str
    delivery_status: DeliveryStatus
    attempt_count: int
    first_attempt_at_utc: str
    last_attempt_at_utc: str
    http_status: Optional[int] = None
    remote_event_id: Optional[str] = None
    error: Optional[str] = None
    publisher_version: Optional[str] = "1.0"
    promoted_artifact_fingerprint: Optional[str] = None
    governance_decision_fingerprint: Optional[str] = None
    campaign_selection_decision_fingerprint: Optional[str] = None
    delivery_receipt_fingerprint: str = field(default="", init=False)

    def __post_init__(self) -> None:
        mandatory_str_fields = {
            "publication_id": self.publication_id,
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "canonical_live_decision_fingerprint": self.canonical_live_decision_fingerprint,
            "runtime_authorization_fingerprint": self.runtime_authorization_fingerprint,
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "first_attempt_at_utc": self.first_attempt_at_utc,
            "last_attempt_at_utc": self.last_attempt_at_utc,
        }
        for field_name, val in mandatory_str_fields.items():
            if not val or not str(val).strip():
                raise ValueError(f"PublicationDeliveryReceipt missing mandatory non-empty field '{field_name}'")

        if not isinstance(self.delivery_status, DeliveryStatus):
            try:
                status_enum = DeliveryStatus(str(self.delivery_status))
                object.__setattr__(self, "delivery_status", status_enum)
            except Exception:
                raise ValueError(f"Invalid delivery_status '{self.delivery_status}'")

        if self.attempt_count < 1:
            raise ValueError(f"attempt_count must be >= 1, got {self.attempt_count}")

        # Redact any accidental secret in error message
        if self.error:
            object.__setattr__(self, "error", _redact_secrets_from_text(self.error))

        # Compute deterministic delivery_receipt_fingerprint
        fp = compute_delivery_receipt_fingerprint(
            publication_id=self.publication_id,
            decision_id=self.decision_id,
            signal_id=self.signal_id,
            canonical_live_decision_fingerprint=self.canonical_live_decision_fingerprint,
            runtime_authorization_fingerprint=self.runtime_authorization_fingerprint,
            candidate_id=self.candidate_id,
            strategy_name=self.strategy_name,
            strategy_version=self.strategy_version,
            symbol=self.symbol,
            timeframe=self.timeframe,
            delivery_status=self.delivery_status.value,
            attempt_count=self.attempt_count,
            first_attempt_at_utc=self.first_attempt_at_utc,
            last_attempt_at_utc=self.last_attempt_at_utc,
            http_status=self.http_status,
            remote_event_id=self.remote_event_id,
            error=self.error,
            publisher_version=self.publisher_version,
            promoted_artifact_fingerprint=self.promoted_artifact_fingerprint,
            governance_decision_fingerprint=self.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=self.campaign_selection_decision_fingerprint,
        )
        object.__setattr__(self, "delivery_receipt_fingerprint", fp)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "decision_id": self.decision_id,
            "signal_id": self.signal_id,
            "canonical_live_decision_fingerprint": self.canonical_live_decision_fingerprint,
            "runtime_authorization_fingerprint": self.runtime_authorization_fingerprint,
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "delivery_status": self.delivery_status.value,
            "attempt_count": self.attempt_count,
            "first_attempt_at_utc": self.first_attempt_at_utc,
            "last_attempt_at_utc": self.last_attempt_at_utc,
            "http_status": self.http_status,
            "remote_event_id": self.remote_event_id,
            "error": self.error,
            "publisher_version": self.publisher_version,
            "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": self.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
            "delivery_receipt_fingerprint": self.delivery_receipt_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PublicationDeliveryReceipt:
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping")
        status_val = data.get("delivery_status")
        if isinstance(status_val, str):
            status_enum = DeliveryStatus(status_val)
        elif isinstance(status_val, DeliveryStatus):
            status_enum = status_val
        else:
            raise ValueError(f"Missing or invalid delivery_status: {status_val}")

        return cls(
            publication_id=str(data["publication_id"]),
            decision_id=str(data["decision_id"]),
            signal_id=str(data["signal_id"]),
            canonical_live_decision_fingerprint=str(data["canonical_live_decision_fingerprint"]),
            runtime_authorization_fingerprint=str(data["runtime_authorization_fingerprint"]),
            candidate_id=str(data["candidate_id"]),
            strategy_name=str(data["strategy_name"]),
            strategy_version=str(data["strategy_version"]),
            symbol=str(data["symbol"]),
            timeframe=str(data["timeframe"]),
            delivery_status=status_enum,
            attempt_count=int(data["attempt_count"]),
            first_attempt_at_utc=str(data["first_attempt_at_utc"]),
            last_attempt_at_utc=str(data["last_attempt_at_utc"]),
            http_status=int(data["http_status"]) if data.get("http_status") is not None else None,
            remote_event_id=str(data["remote_event_id"]) if data.get("remote_event_id") is not None else None,
            error=str(data["error"]) if data.get("error") is not None else None,
            publisher_version=str(data["publisher_version"]) if data.get("publisher_version") is not None else "1.0",
            promoted_artifact_fingerprint=str(data["promoted_artifact_fingerprint"]) if data.get("promoted_artifact_fingerprint") is not None else None,
            governance_decision_fingerprint=str(data["governance_decision_fingerprint"]) if data.get("governance_decision_fingerprint") is not None else None,
            campaign_selection_decision_fingerprint=str(data["campaign_selection_decision_fingerprint"]) if data.get("campaign_selection_decision_fingerprint") is not None else None,
        )


def compute_delivery_receipt_fingerprint(**kwargs: Any) -> str:
    """Compute deterministic SHA-256 fingerprint over canonical delivery evidence."""
    payload = {
        "publication_id": str(kwargs.get("publication_id", "")),
        "decision_id": str(kwargs.get("decision_id", "")),
        "signal_id": str(kwargs.get("signal_id", "")),
        "canonical_live_decision_fingerprint": str(kwargs.get("canonical_live_decision_fingerprint", "")),
        "runtime_authorization_fingerprint": str(kwargs.get("runtime_authorization_fingerprint", "")),
        "candidate_id": str(kwargs.get("candidate_id", "")),
        "strategy_name": str(kwargs.get("strategy_name", "")),
        "strategy_version": str(kwargs.get("strategy_version", "")),
        "symbol": str(kwargs.get("symbol", "")),
        "timeframe": str(kwargs.get("timeframe", "")),
        "delivery_status": str(kwargs.get("delivery_status", "")),
        "attempt_count": int(kwargs.get("attempt_count", 1)),
        "first_attempt_at_utc": str(kwargs.get("first_attempt_at_utc", "")),
        "last_attempt_at_utc": str(kwargs.get("last_attempt_at_utc", "")),
        "http_status": kwargs.get("http_status"),
        "remote_event_id": kwargs.get("remote_event_id"),
        "error": kwargs.get("error"),
        "publisher_version": kwargs.get("publisher_version", "1.0"),
        "promoted_artifact_fingerprint": kwargs.get("promoted_artifact_fingerprint"),
        "governance_decision_fingerprint": kwargs.get("governance_decision_fingerprint"),
        "campaign_selection_decision_fingerprint": kwargs.get("campaign_selection_decision_fingerprint"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def map_publisher_result_to_status(publish_result: Optional[Mapping[str, Any]]) -> tuple[DeliveryStatus, Optional[int], Optional[str], Optional[str]]:
    """Map Project2Publisher.publish() result contract into DeliveryStatus and transport details."""
    if not publish_result or not isinstance(publish_result, Mapping):
        return DeliveryStatus.NOT_ATTEMPTED, None, None, "No publisher execution result"

    raw_status = str(publish_result.get("status", "")).upper()
    http_code = publish_result.get("http_code")
    http_status = int(http_code) if http_code is not None else None
    remote_event_id = publish_result.get("event_id") or publish_result.get("publication_id")
    if remote_event_id is not None:
        remote_event_id = str(remote_event_id)

    error_msg = publish_result.get("error") or publish_result.get("reason")
    if error_msg is not None:
        error_msg = str(error_msg)

    if raw_status == "PUBLISHED" or publish_result.get("published") is True:
        return DeliveryStatus.DELIVERED, http_status, remote_event_id, None

    if raw_status == "SKIPPED_DISABLED":
        return DeliveryStatus.SKIPPED_DISABLED, http_status, remote_event_id, error_msg or "Publisher disabled"

    if raw_status == "SKIPPED_NO_TRADE":
        return DeliveryStatus.SKIPPED_NO_TRADE, http_status, remote_event_id, error_msg or "Decision is NO TRADE"

    if raw_status == "SKIPPED_STALE":
        return DeliveryStatus.SKIPPED_STALE, http_status, remote_event_id, error_msg or "Event timestamp stale"

    if raw_status == "REJECTED":
        return DeliveryStatus.REJECTED, http_status, remote_event_id, error_msg or "Signal rejected by gateway"

    if raw_status in ("TIMED_OUT", "UNAVAILABLE"):
        return DeliveryStatus.FAILED_RETRYABLE, http_status, remote_event_id, error_msg or f"Transport failure: {raw_status}"

    if raw_status in ("AUTH_FAILED", "FORBIDDEN", "INVALID_RESPONSE", "FAILED", "MISCONFIGURED"):
        return DeliveryStatus.FAILED_PERMANENT, http_status, remote_event_id, error_msg or f"Permanent delivery error: {raw_status}"

    return DeliveryStatus.FAILED_PERMANENT, http_status, remote_event_id, error_msg or f"Unknown publisher status: {raw_status}"


def load_delivery_history(path: str | Path = DEFAULT_DELIVERY_STORE_PATH) -> list[PublicationDeliveryReceipt]:
    """Load delivery history receipts from durable ledger."""
    target = Path(path)
    if not target.exists():
        return []

    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("stored delivery history must be a list")

    results: list[PublicationDeliveryReceipt] = []
    for item in raw:
        if isinstance(item, Mapping):
            results.append(PublicationDeliveryReceipt.from_dict(item))
    return results


def save_delivery_history(
    history: Iterable[PublicationDeliveryReceipt],
    path: str | Path = DEFAULT_DELIVERY_STORE_PATH,
) -> Path:
    """Save delivery history list to durable ledger file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [receipt.as_dict() for receipt in history]
    target.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def find_delivery_receipt(
    publication_id: str,
    canonical_live_decision_fingerprint: str,
    path: str | Path = DEFAULT_DELIVERY_STORE_PATH,
) -> Optional[PublicationDeliveryReceipt]:
    """Find a delivery receipt in the ledger matching publication_id or canonical decision fingerprint."""
    history = load_delivery_history(path)
    for receipt in history:
        if (
            receipt.publication_id == publication_id
            or receipt.canonical_live_decision_fingerprint == canonical_live_decision_fingerprint
        ):
            return receipt
    return None


def append_delivery_receipt(
    receipt: PublicationDeliveryReceipt,
    path: str | Path = DEFAULT_DELIVERY_STORE_PATH,
    enforce_idempotency: bool = True,
) -> list[PublicationDeliveryReceipt]:
    """Append or update a delivery receipt in the durable recovery ledger with conflict protection."""
    if not isinstance(receipt, PublicationDeliveryReceipt):
        raise TypeError(f"receipt must be a PublicationDeliveryReceipt, got {type(receipt).__name__}")

    target = Path(path)
    history = load_delivery_history(target)

    # Search for existing receipt matching publication_id or canonical_live_decision_fingerprint
    existing_idx = None
    for idx, existing in enumerate(history):
        if (
            existing.publication_id == receipt.publication_id
            or existing.canonical_live_decision_fingerprint == receipt.canonical_live_decision_fingerprint
        ):
            existing_idx = idx
            break

    if existing_idx is not None:
        existing = history[existing_idx]

        # Conflict protection over authoritative lineage
        lineage_checks = {
            "publication_id": (existing.publication_id, receipt.publication_id),
            "canonical_live_decision_fingerprint": (existing.canonical_live_decision_fingerprint, receipt.canonical_live_decision_fingerprint),
            "runtime_authorization_fingerprint": (existing.runtime_authorization_fingerprint, receipt.runtime_authorization_fingerprint),
            "decision_id": (existing.decision_id, receipt.decision_id),
            "signal_id": (existing.signal_id, receipt.signal_id),
            "candidate_id": (existing.candidate_id, receipt.candidate_id),
            "symbol": (existing.symbol, receipt.symbol),
            "timeframe": (existing.timeframe, receipt.timeframe),
        }

        mismatches = [f"{k} ('{v[0]}' vs '{v[1]}')" for k, v in lineage_checks.items() if v[0] != v[1]]
        if mismatches:
            raise DeliveryIntegrityError(
                f"Conflicting delivery receipt detected for publication_id '{receipt.publication_id}': "
                + ", ".join(mismatches)
            )

        if existing.delivery_status == DeliveryStatus.DELIVERED:
            if enforce_idempotency and receipt.delivery_status == DeliveryStatus.DELIVERED:
                # Identical DELIVERED replay: safe no-op
                return history
            elif receipt.delivery_status != DeliveryStatus.DELIVERED:
                raise DeliveryIntegrityError(
                    f"Cannot regress delivery status of publication '{receipt.publication_id}' "
                    f"from DELIVERED to {receipt.delivery_status.value}"
                )

        # Update existing receipt record with new attempt info
        history[existing_idx] = receipt
    else:
        history.append(receipt)

    save_delivery_history(history, target)
    return history
