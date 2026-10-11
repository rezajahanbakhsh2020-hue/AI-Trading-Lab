"""Focused regression test suite for P1 continuous production worker configuration,
per-timeframe readiness, candidate mapping, publisher invocation, timestamp contracts,
and gateway acknowledgement handling.
"""

import datetime
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ContinuousLiveRuntime,
    parse_continuous_candidate_ids,
    verify_timeframe_production_readiness,
    PromotionUnavailable,
    DEFAULT_RESEARCH_DIR,
)
from src.evaluation.live_production_decision import (
    PromotedCandidateArtifact,
    ProductionRuntimeAuthorization,
    ProductionAuthorizationReceipt,
    ProductionIntelligencePublication,
    ProductionSignal,
    evaluate_production_decision,
    calculate_production_risk_levels,
    authorize_production_runtime,
)
from src.evaluation.research_store import resolve_promoted_candidate
from src.evaluation.mtf_intelligence import CanonicalTimeframe
from src.integration.project2_publisher import Project2Publisher, validate_publication_configuration


def _create_real_pub_artifact(timeframe: str = "5m", candidate_id: str = "cand_moving_average_5m") -> ProductionIntelligencePublication:
    """Helper to create a genuine, fully-authorized ProductionIntelligencePublication instance."""
    cand = resolve_promoted_candidate(candidate_id=candidate_id, symbol="XAUUSD", timeframe=timeframe)
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe=timeframe)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    df = pd.DataFrame([
        {
            "timestamp": now_utc - datetime.timedelta(minutes=10),
            "open": 2000.0,
            "high": 2005.0,
            "low": 1995.0,
            "close": 2002.0,
            "volume": 100.0,
            "isClosed": True,
        },
        {
            "timestamp": now_utc - datetime.timedelta(minutes=5),
            "open": 2002.0,
            "high": 2010.0,
            "low": 2001.0,
            "close": 2008.0,
            "volume": 150.0,
            "isClosed": True,
        },
    ])
    dec = evaluate_production_decision(cand, df)
    sig = ProductionSignal.from_decision(dec)
    risk = calculate_production_risk_levels(dec, cand)
    return ProductionIntelligencePublication.from_artifacts(
        decision=dec,
        signal=sig,
        risk=risk,
        candidate=cand,
        authorization=auth,
    )


class TestP1WorkerReadinessAndPublicationForensics:
    """Regression test suite validating seven-timeframe candidate mapping, readiness, publication, and security contracts."""

    def test_live_execution_runtime_default_construction_respects_env_disabled(self, monkeypatch):
        """Verify LiveExecutionRuntime default publisher construction respects PROJECT2_PUBLISH_ENABLED=false."""
        monkeypatch.delenv("PROJECT2_PUBLISH_ENABLED", raising=False)
        runtime = LiveExecutionRuntime(symbol="XAUUSD", interval="5m")
        assert runtime.publisher.enabled is False

    def test_continuous_live_runtime_publish_flag_enables_publisher(self, monkeypatch):
        """Verify ContinuousLiveRuntime(publish=True) enables publisher regardless of env default."""
        monkeypatch.delenv("PROJECT2_PUBLISH_ENABLED", raising=False)
        runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], publish=True)
        assert runtime.publish is True
        assert runtime.publisher.enabled is True

        runtime_off = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], publish=False)
        assert runtime_off.publisher.enabled is False

    def test_publisher_disabled_is_explicit(self):
        """1. Verify publisher disabled status is explicit and controlled by configuration/enabled setting."""
        pub = Project2Publisher(enabled=False, publish_url="https://example.com/api", api_key="test_key")
        assert not pub.enabled
        pub_artifact = _create_real_pub_artifact()
        res = pub.publish(pub_artifact)
        assert res["status"] == "SKIPPED_DISABLED"
        assert res["published"] is False

    def test_missing_publisher_configuration_fails_closed(self):
        """2. Verify missing publisher URL or API key fails closed."""
        pub_artifact = _create_real_pub_artifact()

        # Missing API Key
        pub_no_key = Project2Publisher(enabled=True, publish_url="https://example.com/api", api_key="")
        res = pub_no_key.publish(pub_artifact)
        assert res["status"] == "FAILED"
        assert res["published"] is False
        assert "Missing PROJECT2_API_KEY" in res["reason"]

        # Missing URL
        pub_no_url = Project2Publisher(enabled=True, publish_url="", api_key="key_123")
        res2 = pub_no_url.publish(pub_artifact)
        assert res2["status"] == "FAILED"
        assert res2["published"] is False
        assert "Missing PROJECT2_PUBLISH_URL" in res2["reason"]

    def test_worker_resolves_candidate_for_exact_timeframe(self):
        """3. Verify readiness check resolves candidates for exact timeframe without borrowing across timeframes."""
        res_5m = verify_timeframe_production_readiness(
            symbol="XAUUSD", timeframe="5m", candidate_id="cand_moving_average_5m", research_dir=DEFAULT_RESEARCH_DIR
        )
        assert res_5m["status"] == "READY"
        assert res_5m["candidate_id"] == "cand_moving_average_5m"

        res_1m = verify_timeframe_production_readiness(
            symbol="XAUUSD", timeframe="1m", candidate_id=None, research_dir=DEFAULT_RESEARCH_DIR
        )
        assert res_1m["status"] == "BLOCKED"
        assert res_1m["reason_code"] == "PromotionUnavailable"

    def test_worker_does_not_copy_candidate_across_timeframes(self):
        """4. Verify candidate IDs are not copied or reused across multiple timeframes without explicit mapping."""
        tfs = ["5m", "15m"]
        arg_str = "cand_single"
        with pytest.raises(ValueError) as exc_info:
            parse_continuous_candidate_ids(arg_str, timeframes=tfs)
        assert "rather than copying a single candidate ID" in str(exc_info.value)

    def test_actual_publisher_sends_contract_to_p2_ingest(self):
        """5. Verify actual Project2Publisher sends valid Contract v1 payload over HTTP transport."""
        pub = Project2Publisher(
            publish_url="https://example.com/api/v1/integration/project1/ingest",
            api_key="valid_api_key_123",
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()
        event_id = pub_artifact.publication_id

        resp_body = json.dumps({
            "status": "ACCEPTED",
            "event_id": event_id,
            "publication_id": event_id,
        }).encode("utf-8")

        mock_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = resp_body
        mock_resp.geturl.return_value = "https://example.com/api/v1/integration/project1/ingest"
        mock_cm.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            res = pub.publish(pub_artifact)

        assert res["status"] == "PUBLISHED"
        assert res["published"] is True
        assert res["http_code"] == 200
        assert res["event_id"] == event_id
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        assert headers_lower.get("x-api-key") == "valid_api_key_123"
        assert headers_lower.get("x-idempotency-key") == event_id

    def test_valid_p2_acknowledgement_is_accepted(self):
        """6. Verify valid identity-bearing P2 acknowledgement is accepted."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="valid_key",
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()
        event_id = pub_artifact.publication_id

        resp_body = json.dumps({
            "status": "INGESTED",
            "publication_id": event_id,
        }).encode("utf-8")

        mock_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = resp_body
        mock_resp.geturl.return_value = "https://example.com/api"
        mock_cm.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_cm):
            res = pub.publish(pub_artifact)

        assert res["status"] == "PUBLISHED"
        assert res["published"] is True
        assert res["remote_event_id"] == event_id

    def test_conflicting_acknowledgement_identity_is_rejected(self):
        """7. Verify conflicting identity fields in gateway acknowledgement receipt fail closed."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="valid_key",
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()
        event_id = pub_artifact.publication_id

        # Conflicting event_id vs publication_id in response
        resp_body = json.dumps({
            "status": "ACCEPTED",
            "event_id": event_id,
            "publication_id": "evt_DIFFERENT_999",
        }).encode("utf-8")

        mock_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = resp_body
        mock_resp.geturl.return_value = "https://example.com/api"
        mock_cm.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_cm):
            res = pub.publish(pub_artifact)

        assert res["status"] == "INVALID_RESPONSE"
        assert res["published"] is False
        assert "Conflicting identity fields" in res["error"]

    def test_http_400_is_classified_correctly(self):
        """8. Verify HTTP 400 Bad Request is classified as REJECTED without retry."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="valid_key",
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()

        mock_http_err = urllib.error.HTTPError(
            url="https://example.com/api",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=MagicMock(read=lambda: b"Schema validation error"),
        )

        with patch("urllib.request.urlopen", side_effect=mock_http_err):
            res = pub.publish(pub_artifact)

        assert res["status"] == "REJECTED"
        assert res["published"] is False
        assert res["http_code"] == 400
        assert res["attempts"] == 1  # Non-retryable!

    def test_http_401_and_403_are_classified_correctly(self):
        """9. Verify HTTP 401 Unauthorized and 403 Forbidden are classified as AUTH_FAILED / FORBIDDEN without retry."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="bad_token",
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()

        # 401
        err_401 = urllib.error.HTTPError(
            url="https://example.com/api",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b"Invalid credentials"),
        )
        with patch("urllib.request.urlopen", side_effect=err_401):
            res_401 = pub.publish(pub_artifact)

        assert res_401["status"] == "AUTH_FAILED"
        assert res_401["published"] is False
        assert res_401["http_code"] == 401
        assert res_401["attempts"] == 1

        # 403
        err_403 = urllib.error.HTTPError(
            url="https://example.com/api",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=MagicMock(read=lambda: b"Forbidden access"),
        )
        with patch("urllib.request.urlopen", side_effect=err_403):
            res_403 = pub.publish(pub_artifact)

        assert res_403["status"] == "FORBIDDEN"
        assert res_403["published"] is False
        assert res_403["http_code"] == 403
        assert res_403["attempts"] == 1

    def test_invalid_or_empty_success_response_is_not_published(self):
        """10. Verify empty or non-JSON HTTP 200 response is rejected as INVALID_RESPONSE."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="valid_key",
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()

        # Empty body HTTP 200
        mock_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = b""
        mock_resp.geturl.return_value = "https://example.com/api"
        mock_cm.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_cm):
            res = pub.publish(pub_artifact)

        assert res["status"] == "INVALID_RESPONSE"
        assert res["published"] is False
        assert "response body is empty" in res["error"]

    def test_secret_is_redacted_from_failure_details(self):
        """11. Verify secrets (e.g. bearer tokens / API keys) are redacted from logs and error messages."""
        secret_token = "my_super_secret_bearer_token_999"
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key=secret_token,
            enabled=True,
        )

        pub_artifact = _create_real_pub_artifact()

        err = urllib.error.HTTPError(
            url="https://example.com/api",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=MagicMock(read=lambda: f"Failed request with token {secret_token}".encode("utf-8")),
        )

        with patch("urllib.request.urlopen", side_effect=err):
            res = pub.publish(pub_artifact)

        assert secret_token not in res["error"]
        assert "[REDACTED]" in res["error"]

    def test_failed_delivery_is_not_reported_as_healthy(self, tmp_path):
        """12. Verify failed publication attempts remain visible in worker health and are not reported as HEALTHY."""
        mock_publisher = MagicMock(spec=Project2Publisher)
        mock_publisher.enabled = True
        mock_publisher.publish_url = "https://example.com/api"
        mock_publisher.api_key = "valid_key"
        mock_publisher.publish.return_value = {
            "status": "AUTH_FAILED",
            "published": False,
            "http_code": 401,
            "error": "HTTP 401 Unauthorized",
        }

        runtime = ContinuousLiveRuntime(
            symbol="XAUUSD",
            timeframes=["5m"],
            candidate_ids={"5m": "cand_moving_average_5m"},
            store_path=tmp_path / "live_decision_history.json",
            snapshot_path=tmp_path / "latest_snapshot.json",
            research_dir=DEFAULT_RESEARCH_DIR,
            publish=True,
            publisher=mock_publisher,
        )

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        open_time_1 = now_utc - datetime.timedelta(minutes=15)
        open_time_2 = now_utc - datetime.timedelta(minutes=10)

        df = pd.DataFrame([
            {
                "timestamp": open_time_1,
                "open": 2000.0,
                "high": 2005.0,
                "low": 1995.0,
                "close": 2002.0,
                "volume": 100.0,
                "isClosed": True,
            },
            {
                "timestamp": open_time_2,
                "open": 2002.0,
                "high": 2010.0,
                "low": 2001.0,
                "close": 2008.0,
                "volume": 150.0,
                "isClosed": True,
            },
        ])

        with patch("src.data.provider.BiQuoteProvider.get_candles", return_value=df):
            tick_summary = runtime.tick(reference_now=now_utc)

        health = tick_summary["health"]
        assert health["status"] in ("WORKER_DEGRADED", "WORKER_STALE")
        assert health["execution_healthy"] is False

    def test_stale_but_valid_signal_is_not_blocked_by_ui_ttl(self):
        """13. Verify publisher allows publication of old/stale signals (>300s) as valid authoritative signals,
        disentangling P1 publication from P2 300s UI LIVE badge TTL.
        """
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="test_key_123",
            enabled=True,
            max_age_seconds=300,
        )

        # 2 hours old UTC timestamp
        stale_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
        valid, status, reason = pub.validate_timestamp(stale_dt.isoformat())

        assert valid is True
        assert status is None
        assert reason is None
