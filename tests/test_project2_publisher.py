"""Tests for Project 2 Outbound Integration Publisher & Delivery Boundary."""

from datetime import datetime, timezone
import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from src.evaluation.live_production_decision import (
    Direction,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionSignal,
    PromotedCandidateArtifact,
    calculate_production_risk_levels,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.integration.project2_publisher import (
    Project2Publisher,
    _redact_secret,
)


def make_test_artifacts(
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    direction: Direction = Direction.BUY,
    mkt_ts: str | None = None,
    stability_score: float = 0.85,
    confidence: float = 0.85,
):
    if mkt_ts is None:
        mkt_ts = datetime.now(timezone.utc).isoformat()
    ds = DatasetScope(
        dataset_id=f"ds_{symbol.lower()}_{timeframe}",
        symbol=symbol,
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c")
    spec = ResearchExperimentSpec(
        hypothesis="Publisher unit test hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-02",
        total_return=0.20,
        max_drawdown=0.05,
        sharpe_ratio=2.0,
        observations=50,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-02T00:00:00+00:00",
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is,),
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )
    cand = PromotedCandidateArtifact(
        candidate_id="cand_pub_unit",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=evidence,
        symbol=symbol,
        timeframe=timeframe,
        operational_stability_score=stability_score,
        governance_decision_fingerprint="gov_fp_test_123",
    )
    dec = ProductionDecision(
        candidate_id=cand.candidate_id,
        evidence_id=evidence.evidence_id,
        experiment_fingerprint=evidence.experiment_fingerprint,
        symbol=symbol,
        timeframe=timeframe,
        decision_timestamp=mkt_ts,
        market_timestamp=mkt_ts,
        direction=direction,
        reason="test_direction",
        entry_price=2000.0 if direction == Direction.BUY else None,
        invalidation_condition="Close below SL" if direction == Direction.BUY else None,
        confidence=confidence,
        parameters=cand.parameters,
    )
    sig = ProductionSignal.from_decision(dec)
    risk = calculate_production_risk_levels(dec, cand)
    pub = ProductionIntelligencePublication.from_artifacts(
        decision=dec,
        signal=sig,
        risk=risk,
        candidate=cand,
        confidence=confidence,
    )
    return cand, dec, sig, risk, pub


def test_redact_secret() -> None:
    secret = "super-secret-api-key"
    text = f"Failed request with Bearer {secret} to endpoint"
    redacted = _redact_secret(text, secret)
    assert secret not in redacted
    assert "[REDACTED]" in redacted


def test_publisher_disabled_by_default() -> None:
    publisher = Project2Publisher(enabled=False)
    _, _, _, _, pub = make_test_artifacts()
    res = publisher.publish(pub)
    assert res["status"] == "SKIPPED_DISABLED"
    assert res["published"] is False


def test_publisher_skip_no_trade() -> None:
    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="test-key", enabled=True)
    _, _, _, _, pub = make_test_artifacts(direction=Direction.NO_TRADE)
    res = publisher.publish(pub, skip_if_no_trade=True)
    assert res["status"] == "SKIPPED_NO_TRADE"
    assert res["published"] is False


def test_publisher_stale_detection() -> None:
    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="test-key",
        enabled=True,
        max_age_seconds=60,
    )
    _, _, _, _, pub = make_test_artifacts(mkt_ts="2020-01-01T00:00:00+00:00")
    res = publisher.publish(pub)
    assert res["status"] == "SKIPPED_STALE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_publisher_successful_delivery(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()
    evt_id = pub.publication_id

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"status": "INGESTED", "event_id": evt_id}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="test-key",
        enabled=True,
    )
    res = publisher.publish(pub)

    assert res["status"] == "PUBLISHED"
    assert res["published"] is True
    assert res["http_code"] == 200
    assert res["remote_event_id"] == evt_id
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_publisher_retry_and_failure(mock_urlopen) -> None:
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="secret-key",
        enabled=True,
        max_retries=2,
        backoff_factor=0.01,
    )
    _, _, _, _, pub = make_test_artifacts()
    res = publisher.publish(pub)

    assert res["status"] in ("UNAVAILABLE", "FAILED")
    assert res["published"] is False
    assert res["attempts"] == 2
    assert "secret-key" not in res["error"]


def test_publisher_missing_api_key() -> None:
    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="",
        enabled=True,
    )
    _, _, _, _, pub = make_test_artifacts()
    res = publisher.publish(pub)
    assert res["status"] == "FAILED"
    assert "PROJECT2_API_KEY" in res["reason"]


@patch("urllib.request.urlopen")
def test_publisher_rejected_http_401(mock_urlopen) -> None:
    err = urllib.error.HTTPError(
        url="https://api.example.com/signals",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error": "Invalid API Key"}'),
    )
    mock_urlopen.side_effect = err

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="bad-key",
        enabled=True,
    )
    _, _, _, _, pub = make_test_artifacts()
    res = publisher.publish(pub)
    assert res["status"] == "AUTH_FAILED"
    assert res["http_code"] == 401
    assert res["attempts"] == 1


# --- Blocker 10 Detailed Matrix Tests ---

@patch("urllib.request.urlopen")
def test_matrix_A_http_200_valid_identity_and_accepted_status(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()
    evt_id = pub.publication_id

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"status": "INGESTED", "event_id": evt_id}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "PUBLISHED"
    assert res["published"] is True


@patch("urllib.request.urlopen")
def test_matrix_B_http_200_no_identity_fails(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"status": "INGESTED"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_matrix_C_http_200_malformed_json_fails(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b'NOT VALID JSON'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_matrix_D_http_200_wrong_identity_fails(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"status": "INGESTED", "event_id": "wrong_id_123"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_matrix_E_http_200_conflicting_identity_fields_fails(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({
        "status": "INGESTED",
        "event_id": pub.publication_id,
        "publication_id": "conflicting_pub_id",
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_matrix_F_http_204_empty_body_fails(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 204
    mock_resp.read.return_value = b""
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_matrix_G_http_200_unrecognized_status_fails(mock_urlopen) -> None:
    _, _, _, _, pub = make_test_artifacts()

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"status": "UNRECOGNIZED_FOO", "event_id": pub.publication_id}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True)
    res = publisher.publish(pub)
    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False


def test_matrix_H_timestamp_freshness_boundaries() -> None:
    publisher = Project2Publisher(publish_url="https://api.example.com/signals", api_key="key", enabled=True, max_age_seconds=300)
    cand, dec, sig, risk, pub = make_test_artifacts()

    # 1. Naive timestamp -> INVALID_RESPONSE
    pub_dict_naive = pub.as_dict()
    pub_dict_naive["market_data_timestamp"] = "2025-01-01T12:00:00"  # missing offset!
    res_naive = publisher.publish(pub_dict_naive)
    assert res_naive["status"] == "INVALID_RESPONSE"

    # 2. Malformed timestamp -> INVALID_RESPONSE
    pub_dict_malformed = pub.as_dict()
    pub_dict_malformed["market_data_timestamp"] = "NOT_A_TIMESTAMP"
    res_malformed = publisher.publish(pub_dict_malformed)
    assert res_malformed["status"] == "INVALID_RESPONSE"

    # 3. Future timestamp -> INVALID_RESPONSE
    pub_dict_future = pub.as_dict()
    pub_dict_future["market_data_timestamp"] = "2099-01-01T12:00:00+00:00"
    res_future = publisher.publish(pub_dict_future)
    assert res_future["status"] == "INVALID_RESPONSE"

    # 4. Older than 300 seconds -> SKIPPED_STALE
    pub_dict_stale = pub.as_dict()
    pub_dict_stale["market_data_timestamp"] = "2020-01-01T12:00:00+00:00"
    stale_dec = ProductionDecision(
        candidate_id=cand.candidate_id,
        evidence_id=cand.evidence.evidence_id,
        experiment_fingerprint=cand.evidence.experiment_fingerprint,
        symbol=cand.symbol,
        timeframe=cand.timeframe,
        decision_timestamp="2020-01-01T12:00:00+00:00",
        market_timestamp="2020-01-01T12:00:00+00:00",
        direction=Direction.BUY,
        reason="test_stale",
        entry_price=2000.0,
        invalidation_condition="Close below SL",
        confidence=0.85,
        parameters=cand.parameters,
    )
    stale_sig = ProductionSignal.from_decision(stale_dec)
    stale_risk = calculate_production_risk_levels(stale_dec, cand)
    stale_payload = ProductionIntelligencePublication.from_artifacts(
        stale_dec,
        stale_sig,
        stale_risk,
        cand,
    )
    res_stale = publisher.publish(stale_payload)
    assert res_stale["status"] == "SKIPPED_STALE"
