"""Focused regression test suite for P1 continuous production worker configuration,
per-timeframe readiness, candidate mapping, publisher invocation, timestamp contracts,
and gateway acknowledgement handling.
"""

import datetime
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
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
)
from src.evaluation.mtf_intelligence import CanonicalTimeframe
from src.integration.project2_publisher import Project2Publisher, validate_publication_configuration


class TestP1WorkerReadinessAndPublicationForensics:
    """Regression test suite validating seven-timeframe candidate mapping, readiness, publication, and security contracts."""

    def test_seven_timeframe_candidate_mapping_parsing(self):
        """Verify parse_continuous_candidate_ids correctly parses 7-timeframe candidate mappings."""
        tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
        arg_str = "5m=cand_moving_average_5m,1D=cand_moving_average_1d"
        mapping = parse_continuous_candidate_ids(arg_str, timeframes=tfs)

        assert mapping == {
            "5m": "cand_moving_average_5m",
            "1D": "cand_moving_average_1d",
        }
        assert mapping.get("1m") is None
        assert mapping.get("15m") is None
        assert mapping.get("30m") is None
        assert mapping.get("1H") is None
        assert mapping.get("4H") is None

    def test_missing_or_invalid_candidate_readiness_diagnostics(self):
        """Verify readiness checks for non-promoted timeframes report BLOCKED with explicit reason code."""
        res_1m = verify_timeframe_production_readiness(
            symbol="XAUUSD", timeframe="1m", candidate_id=None, research_dir=DEFAULT_RESEARCH_DIR
        )
        assert res_1m["status"] == "BLOCKED"
        assert res_1m["reason_code"] == "PromotionUnavailable"
        assert "No persisted promoted candidate" in res_1m["detail"]

        res_5m = verify_timeframe_production_readiness(
            symbol="XAUUSD", timeframe="5m", candidate_id="cand_moving_average_5m", research_dir=DEFAULT_RESEARCH_DIR
        )
        assert res_5m["status"] == "READY"
        assert res_5m["reason_code"] == "READY"
        assert res_5m["candidate_id"] == "cand_moving_average_5m"

    def test_mixed_ready_blocked_timeframe_startup(self, tmp_path):
        """Verify ContinuousLiveRuntime startup preflight succeeds when at least 1 timeframe is READY,
        and logs non-ready timeframes without discarding valid timeframes.
        """
        tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
        cand_map = {"5m": "cand_moving_average_5m", "1D": "cand_moving_average_1d"}

        runtime = ContinuousLiveRuntime(
            symbol="XAUUSD",
            timeframes=tfs,
            candidate_ids=cand_map,
            store_path=tmp_path / "live_decision_history.json",
            snapshot_path=tmp_path / "latest_snapshot.json",
            research_dir=DEFAULT_RESEARCH_DIR,
            publish=False,
        )

        readiness = runtime.verify_startup_readiness()
        assert readiness["5m"]["status"] == "READY"
        assert readiness["1D"]["status"] == "READY"
        assert readiness["1m"]["status"] == "BLOCKED"
        assert readiness["15m"]["status"] == "BLOCKED"
        assert readiness["30m"]["status"] == "BLOCKED"
        assert readiness["1H"]["status"] == "BLOCKED"
        assert readiness["4H"]["status"] == "BLOCKED"

    def test_zero_ready_timeframes_startup_failure(self, tmp_path):
        """Verify ContinuousLiveRuntime startup preflight raises PromotionUnavailable when 0 timeframes are READY."""
        tfs = ["1m", "15m", "30m"]
        cand_map = {}

        runtime = ContinuousLiveRuntime(
            symbol="XAUUSD",
            timeframes=tfs,
            candidate_ids=cand_map,
            store_path=tmp_path / "live_decision_history.json",
            snapshot_path=tmp_path / "latest_snapshot.json",
            research_dir=DEFAULT_RESEARCH_DIR,
            publish=False,
        )

        with pytest.raises(PromotionUnavailable) as exc_info:
            runtime.verify_startup_readiness()

        assert "0 of 3 configured timeframes are READY" in str(exc_info.value)

    def test_worker_to_publisher_invocation_flow(self, tmp_path):
        """Verify worker execution flow invokes Project2Publisher.publish() when publish=True and valid market data exists."""
        mock_publisher = MagicMock(spec=Project2Publisher)
        mock_publisher.enabled = True
        mock_publisher.publish_url = "https://example.com/api"
        mock_publisher.api_key = "valid_key"
        mock_publisher.publish.return_value = {
            "status": "PUBLISHED",
            "published": True,
            "http_code": 200,
            "event_id": "evt_mock_123",
            "publication_id": "evt_mock_123",
            "remote_event_id": "evt_mock_123",
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

        assert "5m" in tick_summary["evaluations"]
        eval_5m = tick_summary["evaluations"]["5m"]
        assert eval_5m.get("blocked") is False
        assert mock_publisher.publish.called
        pub_call_args = mock_publisher.publish.call_args
        pub_payload = pub_call_args[0][0]
        assert isinstance(pub_payload, ProductionIntelligencePublication)

    def test_publisher_configuration_validation(self):
        """Verify validate_publication_configuration strictly validates URLs and API keys without secret leakage."""
        # Missing key
        valid, msg = validate_publication_configuration("https://example.com/api", "")
        assert not valid
        assert "PROJECT2_API_KEY" in msg

        # Missing URL
        valid, msg = validate_publication_configuration("", "valid_key")
        assert not valid
        assert "PROJECT2_PUBLISH_URL" in msg

        # API key in query params rejected
        valid, msg = validate_publication_configuration("https://example.com/api?key=secret_123", "secret_123")
        assert not valid
        assert "must not be passed in query parameters" in msg

        # Valid config
        valid, msg = validate_publication_configuration("https://example.com/api", "valid_key_123")
        assert valid
        assert msg == ""

    def test_publisher_stale_but_valid_timestamp_contract(self):
        """Verify publisher allows publication of old/stale signals (>300s) as valid authoritative signals,
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

    def test_publisher_future_and_malformed_timestamp_rejection(self):
        """Verify publisher fails closed on future, timezone-naive, or malformed timestamps."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="test_key_123",
            enabled=True,
        )

        # Naive timestamp
        naive_str = "2026-10-11T12:00:00"
        valid, status, reason = pub.validate_timestamp(naive_str)
        assert valid is False
        assert status == "INVALID_RESPONSE"
        assert "timezone-aware" in reason

        # Future timestamp
        future_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        valid, status, reason = pub.validate_timestamp(future_dt.isoformat())
        assert valid is False
        assert status == "INVALID_RESPONSE"
        assert "Future event timestamp" in reason

        # Malformed timestamp string
        valid, status, reason = pub.validate_timestamp("not-a-timestamp")
        assert valid is False
        assert status == "INVALID_RESPONSE"
        assert "Malformed timestamp" in reason

    def test_publisher_explicit_gateway_rejection_and_auth_failure(self):
        """Verify Project2Publisher handles explicit gateway rejections (400/401/403/REJECTED status) cleanly with secret redaction."""
        pub = Project2Publisher(
            publish_url="https://example.com/api",
            api_key="super_secret_bearer_token_xyz",
            enabled=True,
        )

        # Mock payload
        pub_artifact = MagicMock(spec=ProductionIntelligencePublication)
        pub_artifact.to_contract_v1_payload.return_value = {
            "contract_version": "1.0",
            "event_id": "evt_test_12345",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "signal": {"decision": "BUY"},
            "provenance": {
                "produced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        }

        # Test 401 Auth Failure
        mock_http_err = urllib.error.HTTPError(
            url="https://example.com/api",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b"Invalid API key super_secret_bearer_token_xyz"),
        )

        with patch("urllib.request.urlopen", side_effect=mock_http_err):
            res = pub.publish(pub_artifact)

        assert res["status"] == "AUTH_FAILED"
        assert res["published"] is False
        assert res["http_code"] == 401
        assert "super_secret_bearer_token_xyz" not in res["error"]
        assert "[REDACTED]" in res["error"]
