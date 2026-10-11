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
from urllib.parse import urlparse
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _redact_secret(text: str, secret: Optional[str]) -> str:
    """Redact secret from string if present."""
    if secret and len(secret) > 0 and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text


def validate_publication_configuration(
    publish_url: object,
    api_key: object,
) -> tuple[bool, str]:
    """Deterministically validate publication configuration without network I/O or secret disclosure.

    Returns (is_valid, safe_reason_or_message).
    """
    if api_key is None or not str(api_key).strip():
        return False, "Missing PROJECT2_API_KEY configuration"

    if publish_url is None or not str(publish_url).strip():
        return False, "Missing PROJECT2_PUBLISH_URL configuration"

    url_str = str(publish_url).strip()
    key_str = str(api_key).strip()

    try:
        parsed = urlparse(url_str)
        if not parsed.scheme or parsed.scheme.lower() not in ("http", "https"):
            return False, "Invalid PROJECT2_PUBLISH_URL scheme (must be http or https)"
        if not parsed.netloc:
            return False, "Invalid PROJECT2_PUBLISH_URL missing host"
        if not parsed.hostname or not str(parsed.hostname).strip():
            return False, "Invalid PROJECT2_PUBLISH_URL missing host"
        if any(c.isspace() for c in parsed.netloc):
            return False, "Invalid PROJECT2_PUBLISH_URL host contains invalid whitespace"
        try:
            _ = parsed.hostname
            _ = parsed.port
        except ValueError:
            return False, "Invalid PROJECT2_PUBLISH_URL host or port"

        if key_str in parsed.query:
            return False, "PROJECT2_API_KEY must not be passed in query parameters"
    except Exception:
        return False, "Malformed PROJECT2_PUBLISH_URL"

    return True, ""




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

    def validate_timestamp(self, timestamp_iso: Any) -> tuple[bool, Optional[str], Optional[str]]:
        """Strictly validate event timestamp. Fail closed on missing, malformed, naive, or future timestamps.

        Returns (valid_and_fresh, status_if_failed, reason_if_failed).
        """
        if timestamp_iso is None or not str(timestamp_iso).strip():
            return False, "INVALID_RESPONSE", "Missing required timestamp in payload"

        ts_str = str(timestamp_iso).strip()
        try:
            event_dt = datetime.datetime.fromisoformat(ts_str)
        except Exception as exc:
            return False, "INVALID_RESPONSE", f"Malformed timestamp '{ts_str}': {exc}"

        if event_dt.tzinfo is None or event_dt.tzinfo.utcoffset(event_dt) is None:
            return False, "INVALID_RESPONSE", f"Timestamp '{ts_str}' must be timezone-aware ISO-8601"

        now_dt = datetime.datetime.now(datetime.timezone.utc)
        age_seconds = (now_dt - event_dt).total_seconds()

        if age_seconds < 0:
            return False, "INVALID_RESPONSE", f"Future event timestamp '{ts_str}' relative to '{now_dt.isoformat()}'"

        # IMPORTANT:
        # Publication validity is NOT governed by the 300-second UI LIVE badge.
        # A valid authoritative signal remains publishable/displayable after
        # 300 seconds.  P2 owns the 300-second LIVE indicator semantics.
        return True, None, None

    def is_stale(self, timestamp_iso: str) -> bool:
        """Check if event timestamp is older than max_age_seconds."""
        valid, status, _ = self.validate_timestamp(timestamp_iso)
        if not valid and status == "SKIPPED_STALE":
            return True
        return False

    def publish(
        self,
        payload: Any,
        *,
        skip_if_no_trade: bool = False,
    ) -> Dict[str, Any]:
        """Deliver contract payload to Project 2."""
        from src.evaluation.live_production_decision import ProductionIntelligencePublication

        if isinstance(payload, ProductionIntelligencePublication):
            payload_dict = payload.to_contract_v1_payload()
        else:
            return {
                "status": "INVALID_RESPONSE",
                "published": False,
                "reason": f"Outbound payload must be a ProductionIntelligencePublication instance, got {type(payload).__name__}",
                "error": f"Invalid payload type: {type(payload).__name__}",
            }

        schema_ver = str(payload_dict.get("contract_version", "")).strip()
        if schema_ver != "1.0":
            return {
                "status": "INVALID_RESPONSE",
                "published": False,
                "reason": f"Outbound payload contract_version '{schema_ver}' must be '1.0'",
                "error": f"Invalid contract_version: {schema_ver}",
            }

        if not self.enabled:
            return {
                "status": "SKIPPED_DISABLED",
                "published": False,
                "reason": "Publisher disabled in configuration",
            }

        valid_config, config_reason = validate_publication_configuration(self.publish_url, self.api_key)
        if not valid_config:
            return {
                "status": "FAILED",
                "published": False,
                "reason": config_reason,
            }

        decision = payload_dict.get("signal", {}).get("decision")
        if skip_if_no_trade and decision == "NO TRADE":
            return {
                "status": "SKIPPED_NO_TRADE",
                "published": False,
                "reason": "Decision is NO TRADE and skip_if_no_trade=True",
            }

        event_ts = payload_dict.get("timestamp")
        ts_valid, ts_status, ts_reason = self.validate_timestamp(event_ts)
        if not ts_valid:
            return {
                "status": ts_status,
                "published": False,
                "reason": ts_reason,
                "error": ts_reason,
            }

        provenance = payload_dict.get("provenance")
        produced_at = provenance.get("produced_at") if isinstance(provenance, dict) else None
        publication_ts_valid, publication_ts_status, publication_ts_reason = self.validate_timestamp(produced_at)
        if not publication_ts_valid:
            return {
                "status": publication_ts_status,
                "published": False,
                "reason": f"Invalid authoritative publication timestamp: {publication_ts_reason}",
                "error": publication_ts_reason,
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
                    raw_geturl = resp.geturl() if hasattr(resp, "geturl") and callable(resp.geturl) else None
                    final_url = str(raw_geturl) if isinstance(raw_geturl, str) and raw_geturl.strip() else self.publish_url
                    if final_url and urlparse(final_url).netloc != urlparse(self.publish_url).netloc:
                        return {
                            "status": "REJECTED",
                            "published": False,
                            "http_code": resp.getcode(),
                            "event_id": event_id,
                            "error": f"HTTP redirect to external host '{urlparse(final_url).netloc}' disallowed",
                            "attempts": attempt,
                        }
                    code = resp.getcode()
                    resp_body = resp.read().decode("utf-8")
                    if 200 <= code < 300:
                        # Validate acknowledgement receipt JSON
                        if not resp_body or not resp_body.strip():
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"HTTP {code} response body is empty; identity-bearing acknowledgement required",
                                "attempts": attempt,
                            }

                        try:
                            receipt_json = json.loads(resp_body)
                        except Exception as exc:
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"HTTP {code} response is not valid JSON: {exc}",
                                "attempts": attempt,
                            }

                        if not isinstance(receipt_json, dict):
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"HTTP {code} response JSON is not a object/dict",
                                "attempts": attempt,
                            }

                        # Check explicit rejection statuses first
                        if "status" in receipt_json:
                            ack_status_raw = str(receipt_json.get("status", "")).strip().upper()
                            if ack_status_raw in ("REJECTED", "DECLINED", "INVALID", "FAILED"):
                                return {
                                    "status": "REJECTED",
                                    "published": False,
                                    "http_code": code,
                                    "event_id": event_id,
                                    "error": f"Gateway explicitly rejected signal with status '{ack_status_raw}'",
                                    "attempts": attempt,
                                }

                        # Require identity-bearing acknowledgement: event_id, publication_id, or remote_event_id
                        raw_event_id = receipt_json.get("event_id")
                        raw_pub_id = receipt_json.get("publication_id")
                        raw_remote_id = receipt_json.get("remote_event_id") or receipt_json.get("id")

                        # Disagree check if multiple identity fields are present
                        all_id_vals = [str(v).strip() for v in (raw_event_id, raw_pub_id, raw_remote_id) if v is not None and str(v).strip()]
                        if len(set(all_id_vals)) > 1:
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"Conflicting identity fields in acknowledgement receipt: {receipt_json}",
                                "attempts": attempt,
                            }

                        ack_id = all_id_vals[0] if all_id_vals else None
                        if not ack_id:
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"HTTP {code} response missing remote event_id/publication_id acknowledgement identity",
                                "attempts": attempt,
                            }

                        if ack_id != event_id:
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"Acknowledgement identity mismatch: expected '{event_id}', got '{ack_id}'",
                                "attempts": attempt,
                            }

                        # Require explicit accepted status
                        if "status" not in receipt_json:
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": "HTTP 2xx acknowledgement receipt missing required 'status' field",
                                "attempts": attempt,
                            }

                        ACCEPTED_STATUSES = ("INGESTED", "DUPLICATE_ACCEPTED", "ACCEPTED", "ACKNOWLEDGED", "PUBLISHED", "OK", "SUCCESS", "DELIVERED")
                        ack_status = str(receipt_json.get("status", "")).strip().upper()
                        if ack_status not in ACCEPTED_STATUSES:
                            return {
                                "status": "INVALID_RESPONSE",
                                "published": False,
                                "http_code": code,
                                "event_id": event_id,
                                "error": f"Unrecognized or unaccepted acknowledgement status '{ack_status}'",
                                "attempts": attempt,
                            }

                        return {
                            "status": "PUBLISHED",
                            "published": True,
                            "http_code": code,
                            "event_id": event_id,
                            "publication_id": event_id,
                            "remote_event_id": ack_id,
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
                elif exc.code in (301, 302, 303, 307, 308, 400, 422):
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
