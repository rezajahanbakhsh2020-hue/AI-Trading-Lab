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
    stability_score: Optional[float] = None,
    signal_label: Optional[str] = None,
    trend: Optional[str] = None,
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    tp1: Optional[float] = None,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    take_profit: Optional[float] = None,
    risk_reward_ratio: Optional[float] = None,
    timestamp: Optional[str] = None,
    candle_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a canonical Project 2 Integration Contract v1.0 payload via ProductionIntelligencePublication."""
    from src.evaluation.live_production_decision import ProductionIntelligencePublication

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    event_timestamp = timestamp if timestamp else (candle_timestamp if candle_timestamp else now_iso)
    identity_timestamp = candle_timestamp if candle_timestamp else event_timestamp

    # Normalize event_timestamp to ISO-8601 string
    try:
        dt_check = datetime.datetime.fromisoformat(str(event_timestamp).replace("Z", "+00:00"))
        if dt_check.tzinfo is None:
            dt_check = dt_check.replace(tzinfo=datetime.timezone.utc)
        event_timestamp = dt_check.isoformat()
    except Exception as exc:
        raise ValueError(f"Invalid timestamp '{event_timestamp}': {exc}")

    # Unique deterministic publication ID based on authoritative identity features
    hash_input = f"{symbol}:{interval}:{strategy}:{decision}:{identity_timestamp}"
    pub_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:32]
    sig_id = f"sig_{pub_id[:16]}"
    dec_id = f"dec_{pub_id[16:]}"

    provenance = {
        "source": "AI-Trading-Lab",
        "produced_at": now_iso,
        "publication_contract_version": "1.0",
    }

    eff_tp2 = float(tp2) if tp2 is not None else (float(take_profit) if take_profit is not None else None)

    pub = ProductionIntelligencePublication(
        schema_version="1.0",
        publication_id=pub_id,
        signal_id=sig_id,
        decision_id=dec_id,
        strategy_id=str(strategy),
        candidate_id=f"cand_{strategy}",
        research_evidence_id=f"ev_{strategy}",
        research_fingerprint=pub_id,
        symbol=str(symbol).upper(),
        timeframe=str(interval),
        decision_timestamp=now_iso,
        market_data_timestamp=event_timestamp,
        decision=str(decision),
        confidence=float(stability_score) if stability_score is not None else None,
        entry=float(entry_price) if entry_price is not None else None,
        invalidation="Close below stop_loss or trend turns DOWN" if str(decision) == "BUY" else None,
        stop_loss=float(stop_loss) if stop_loss is not None else None,
        tp1=float(tp1) if tp1 is not None else None,
        tp2=eff_tp2,
        tp3=float(tp3) if tp3 is not None else None,
        trailing_stop=None,
        risk_reward_ratio=float(risk_reward_ratio) if risk_reward_ratio is not None else None,
        provenance=provenance,
    )

    return pub.to_contract_v1_payload()


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
        payload: Any,
        *,
        skip_if_no_trade: bool = False,
    ) -> Dict[str, Any]:
        """Deliver contract payload to Project 2."""
        if hasattr(payload, "to_contract_v1_payload"):
            payload_dict = payload.to_contract_v1_payload()
        elif isinstance(payload, dict):
            payload_dict = payload
        else:
            raise TypeError("payload must be a dictionary or ProductionIntelligencePublication instance.")

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

        if not self.api_key:
            return {
                "status": "FAILED",
                "published": False,
                "reason": "Missing PROJECT2_API_KEY configuration",
            }

        decision = payload_dict.get("signal", {}).get("decision")
        if skip_if_no_trade and decision == "NO TRADE":
            return {
                "status": "SKIPPED_NO_TRADE",
                "published": False,
                "reason": "Decision is NO TRADE and skip_if_no_trade=True",
            }

        event_ts = payload_dict.get("timestamp", "")
        if event_ts and self.is_stale(event_ts):
            return {
                "status": "SKIPPED_STALE",
                "published": False,
                "reason": f"Event timestamp {event_ts} exceeds max_age_seconds={self.max_age_seconds}",
            }

        body_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
        event_id = payload_dict.get("event_id", "")

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
        last_status_code = None
        final_status = "FAILED"

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
                        # Validate acknowledgement receipt
                        receipt_json = None
                        try:
                            receipt_json = json.loads(resp_body)
                        except Exception:
                            receipt_json = None

                        if isinstance(receipt_json, dict):
                            # Verify publication identity in receipt if present
                            ack_id = receipt_json.get("event_id") or receipt_json.get("publication_id") or receipt_json.get("id")
                            if ack_id and ack_id != event_id:
                                return {
                                    "status": "INVALID_RESPONSE",
                                    "published": False,
                                    "http_code": code,
                                    "event_id": event_id,
                                    "error": f"Acknowledgement identity mismatch: expected '{event_id}', got '{ack_id}'",
                                    "attempts": attempt,
                                }

                            ack_status = str(receipt_json.get("status", "")).upper()
                            if ack_status in ("REJECTED", "DECLINED", "INVALID", "FAILED"):
                                return {
                                    "status": "REJECTED",
                                    "published": False,
                                    "http_code": code,
                                    "event_id": event_id,
                                    "error": f"Gateway explicitly rejected signal in receipt with status '{ack_status}'",
                                    "attempts": attempt,
                                }

                        return {
                            "status": "PUBLISHED",
                            "published": True,
                            "http_code": code,
                            "event_id": event_id,
                            "publication_id": event_id,
                            "response": _redact_secret(resp_body, self.api_key),
                            "attempts": attempt,
                        }

                    last_status_code = code
                    last_error = f"HTTP status code {code}"
            except urllib.error.HTTPError as exc:
                last_status_code = exc.code
                err_content = exc.read().decode("utf-8") if exc.fp else ""
                last_error = _redact_secret(f"HTTPError {exc.code}: {exc.reason} - {err_content}", self.api_key)
                if exc.code == 401:
                    final_status = "AUTH_FAILED"
                    break
                elif exc.code == 403:
                    final_status = "FORBIDDEN"
                    break
                elif exc.code in (400, 422):
                    final_status = "REJECTED"
                    break
                elif exc.code in (404, 500, 502, 503, 504):
                    final_status = "UNAVAILABLE"
            except TimeoutError:
                final_status = "TIMED_OUT"
                last_error = "Request timed out"
            except Exception as exc:
                if "timed out" in str(exc).lower():
                    final_status = "TIMED_OUT"
                    last_error = "Request timed out"
                else:
                    final_status = "UNAVAILABLE"
                    last_error = _redact_secret(f"Connection error: {exc}", self.api_key)

            if attempt < self.max_retries and final_status not in ("AUTH_FAILED", "FORBIDDEN", "REJECTED"):
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        result = {
            "status": final_status,
            "published": False,
            "event_id": event_id,
            "error": last_error,
            "attempts": attempt,
        }
        if last_status_code is not None:
            result["http_code"] = last_status_code

        return result
