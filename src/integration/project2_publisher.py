"""Project 2 Integration Contract v1.0 Publisher Module."""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _redact_secret(text: str, secret: Optional[str]) -> str:
    """Redact secret from string if present."""
    if secret and len(secret) > 0 and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text


def build_contract_v1_payload(
    *,
    symbol: str,
    interval: str,
    decision: str,
    strategy: str,
    stability_score: Optional[float],
    signal_label: str,
    trend: str,
    entry_price: Optional[float],
    stop_loss: Optional[float],
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    take_profit: Optional[float] = None,
    risk_reward_ratio: Optional[float] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a canonical Project 2 Integration Contract v1.0 payload."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    event_timestamp = timestamp if timestamp else now_iso

    # Unique event ID based on deterministic features + timestamp
    hash_input = f"{symbol}:{interval}:{strategy}:{decision}:{event_timestamp}"
    event_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:32]

    return {
        "contract_version": "1.0",
        "event_id": event_id,
        "event_type": "TRADING_SIGNAL",
        "timestamp": event_timestamp,
        "instrument": {
            "symbol": str(symbol),
            "interval": str(interval),
        },
        "signal": {
            "decision": str(decision),
            "strategy": str(strategy),
            "stability_score": float(stability_score) if stability_score is not None else None,
            "signal_label": str(signal_label),
            "trend": str(trend),
        },
        "trade_setup": {
            "entry_price": float(entry_price) if entry_price is not None else None,
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "tp1": float(tp1) if tp1 is not None else None,
            "tp2": float(tp2) if tp2 is not None else None,
            "tp3": float(tp3) if tp3 is not None else None,
            "take_profit": float(take_profit) if take_profit is not None else None,
            "risk_reward_ratio": float(risk_reward_ratio) if risk_reward_ratio is not None else None,
        },
        "provenance": {
            "source": "AI-Trading-Lab",
            "produced_at": now_iso,
        },
    }


class Project2Publisher:
    """Outbound publisher targeting Project 2 API according to Contract v1.0."""

    def __init__(
        self,
        publish_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        max_age_seconds: int = 300,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        self.publish_url = (
            publish_url
            if publish_url is not None
            else os.getenv("PROJECT2_PUBLISH_URL", "")
        )
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("PROJECT2_API_KEY", "")
        )

        if enabled is not None:
            self.enabled = enabled
        else:
            env_enabled = os.getenv("PROJECT2_PUBLISH_ENABLED", "false").lower()
            self.enabled = env_enabled in ("true", "1", "yes")

        self.max_age_seconds = max_age_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def is_stale(self, timestamp_iso: str) -> bool:
        """Check if event timestamp is older than max_age_seconds."""
        try:
            # Handle ISO timestamps ending in 'Z' or offset
            ts_str = timestamp_iso.replace("Z", "+00:00")
            event_dt = datetime.datetime.fromisoformat(ts_str)
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=datetime.timezone.utc)

            now_dt = datetime.datetime.now(datetime.timezone.utc)
            age = (now_dt - event_dt).total_seconds()
            return age > self.max_age_seconds
        except Exception as exc:
            logger.warning("Could not parse timestamp %s for staleness check: %s", timestamp_iso, exc)
            return False

    def publish(
        self,
        payload: Dict[str, Any],
        *,
        skip_if_no_trade: bool = False,
    ) -> Dict[str, Any]:
        """Deliver contract payload to Project 2."""
        if not self.enabled:
            return {
                "status": "SKIPPED_DISABLED",
                "published": False,
                "reason": "Publisher disabled in configuration",
            }

        if not self.publish_url:
            return {
                "status": "FAILED",
                "published": False,
                "reason": "Missing PROJECT2_PUBLISH_URL configuration",
            }

        decision = payload.get("signal", {}).get("decision")
        if skip_if_no_trade and decision == "NO TRADE":
            return {
                "status": "SKIPPED_NO_TRADE",
                "published": False,
                "reason": "Decision is NO TRADE and skip_if_no_trade=True",
            }

        event_ts = payload.get("timestamp", "")
        if event_ts and self.is_stale(event_ts):
            return {
                "status": "SKIPPED_STALE",
                "published": False,
                "reason": f"Event timestamp {event_ts} exceeds max_age_seconds={self.max_age_seconds}",
            }

        body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        event_id = payload.get("event_id", "")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Trading-Lab/1.0",
            "X-Idempotency-Key": event_id,
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"

        attempt = 0
        last_error = ""

        while attempt < self.max_retries:
            attempt += 1
            req = urllib.request.Request(
                self.publish_url,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    code = resp.getcode()
                    resp_body = resp.read().decode("utf-8")
                    if 200 <= code < 300:
                        return {
                            "status": "PUBLISHED",
                            "published": True,
                            "http_code": code,
                            "event_id": event_id,
                            "response": _redact_secret(resp_body, self.api_key),
                            "attempts": attempt,
                        }
                    last_error = f"HTTP status code {code}"
            except urllib.error.HTTPError as exc:
                err_content = exc.read().decode("utf-8") if exc.fp else ""
                last_error = _redact_secret(f"HTTPError {exc.code}: {exc.reason} - {err_content}", self.api_key)
                if exc.code in (400, 401, 403, 422):
                    # Client errors shouldn't be retried
                    break
            except Exception as exc:
                last_error = _redact_secret(f"Connection error: {exc}", self.api_key)

            if attempt < self.max_retries:
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        return {
            "status": "FAILED",
            "published": False,
            "event_id": event_id,
            "error": last_error,
            "attempts": attempt,
        }
