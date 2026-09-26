"""Comprehensive unit and integration tests for Project 1 -> Project 2 publication boundary.

Covers full failure matrix, state machine transitions, receipt validation,
lineage preservation, secret redaction, replay idempotency, and regression checks.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import urllib.error

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import LiveExecutionRuntime
from src.evaluation.live_production_decision import (
    Direction,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionSignal,
    PromotedCandidateArtifact,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_publication_store import (
    PublicationIntegrityError,
    append_publication_record,
    load_publication_history,
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
    build_contract_v1_payload,
)


def make_test_promoted_evidence() -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id="ds_pub_test",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-02",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c")
    spec = ResearchExperimentSpec(
        hypothesis="Publication boundary test hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    part = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-02",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
    )
    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part,),
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )


def make_test_artifacts():
    ev = make_test_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_pub_01", "momentum", "1.0", ev, "XAUUSD", "5m")
    now_iso = datetime.now(timezone.utc).isoformat()
    dec = ProductionDecision(
        candidate_id=cand.candidate_id,
        evidence_id=ev.evidence_id,
        experiment_fingerprint=ev.experiment_fingerprint,
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=now_iso,
        market_timestamp=now_iso,
        direction=Direction.BUY,
        reason="momentum_signal_confirmed",
        entry_price=2000.0,
        invalidation_condition="Close below stop loss",
        confidence=0.85,
        parameters=cand.parameters,
    )
    sig = ProductionSignal.from_decision(dec)
    risk = calculate_production_risk_levels(dec, cand)
    pub = ProductionIntelligencePublication.from_artifacts(dec, sig, risk, cand, confidence=0.85)
    return cand, dec, sig, risk, pub


# --- Failure Matrix Scenario Tests ---

def test_failure_matrix_1_publisher_disabled() -> None:
    pub = Project2Publisher(enabled=False)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)
    assert res["status"] == "SKIPPED_DISABLED"
    assert res["published"] is False


def test_failure_matrix_2_missing_url() -> None:
    pub = Project2Publisher(publish_url="", api_key="valid-key", enabled=True)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)
    assert res["status"] == "FAILED"
    assert "Missing PROJECT2_PUBLISH_URL" in res["reason"]


def test_failure_matrix_3_missing_credential() -> None:
    pub = Project2Publisher(publish_url="https://api.example.com/ingest", api_key="", enabled=True)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)
    assert res["status"] == "FAILED"
    assert "Missing PROJECT2_API_KEY" in res["reason"]


@patch("urllib.request.urlopen")
def test_failure_matrix_4_invalid_credential_401_auth_failed(mock_urlopen) -> None:
    err = urllib.error.HTTPError(
        url="https://api.example.com/ingest",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error": "Invalid API key"}'),
    )
    mock_urlopen.side_effect = err

    pub = Project2Publisher(publish_url="https://api.example.com/ingest", api_key="bad-key", enabled=True)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "AUTH_FAILED"
    assert res["http_code"] == 401
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_failure_matrix_5_unauthorized_403_forbidden(mock_urlopen) -> None:
    err = urllib.error.HTTPError(
        url="https://api.example.com/ingest",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error": "IP not whitelisted"}'),
    )
    mock_urlopen.side_effect = err

    pub = Project2Publisher(publish_url="https://api.example.com/ingest", api_key="key", enabled=True)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "FORBIDDEN"
    assert res["http_code"] == 403
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_failure_matrix_6_timeout(mock_urlopen) -> None:
    mock_urlopen.side_effect = TimeoutError("Request timed out")

    pub = Project2Publisher(
        publish_url="https://api.example.com/ingest",
        api_key="key",
        enabled=True,
        max_retries=1,
    )
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "TIMED_OUT"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_failure_matrix_7_connection_failure_unavailable(mock_urlopen) -> None:
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    pub = Project2Publisher(
        publish_url="https://api.example.com/ingest",
        api_key="key",
        enabled=True,
        max_retries=1,
    )
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "UNAVAILABLE"
    assert res["published"] is False


@patch("urllib.request.urlopen")
def test_failure_matrix_8_server_error_5xx_unavailable(mock_urlopen) -> None:
    err = urllib.error.HTTPError(
        url="https://api.example.com/ingest",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error": "Gateway timeout"}'),
    )
    mock_urlopen.side_effect = err

    pub = Project2Publisher(
        publish_url="https://api.example.com/ingest",
        api_key="key",
        enabled=True,
        max_retries=1,
    )
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "UNAVAILABLE"
    assert res["http_code"] == 503


@patch("urllib.request.urlopen")
def test_failure_matrix_9_acknowledgement_identity_mismatch(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b'{"status": "ACCEPTED", "event_id": "mismatch_pub_id_123"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    pub = Project2Publisher(publish_url="https://api.example.com/ingest", api_key="key", enabled=True)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "INVALID_RESPONSE"
    assert res["published"] is False
    assert "identity mismatch" in res["error"]


@patch("urllib.request.urlopen")
def test_failure_matrix_10_remote_rejection_in_receipt(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b'{"status": "REJECTED", "reason": "Risk limit exceeded"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    pub = Project2Publisher(publish_url="https://api.example.com/ingest", api_key="key", enabled=True)
    _, _, _, _, artifact = make_test_artifacts()
    res = pub.publish(artifact)

    assert res["status"] == "REJECTED"
    assert res["published"] is False


def test_failure_matrix_11_identical_replay_idempotency(tmp_path) -> None:
    _, _, _, _, artifact = make_test_artifacts()
    store_file = tmp_path / "publication_history.json"

    rec1 = artifact.as_dict()
    hist1 = append_publication_record(rec1, store_file)
    assert len(hist1) == 1

    # Replay identical record
    hist2 = append_publication_record(rec1, store_file)
    assert len(hist2) == 1  # Safe no-op


def test_failure_matrix_12_conflicting_replay_rejection(tmp_path) -> None:
    _, _, _, _, artifact = make_test_artifacts()
    store_file = tmp_path / "publication_history.json"

    rec1 = artifact.as_dict()
    append_publication_record(rec1, store_file)

    rec2 = dict(rec1)
    rec2["entry"] = 2500.0  # Conflicting payload for same publication_id

    with pytest.raises(PublicationIntegrityError, match="Conflicting publication replay detected"):
        append_publication_record(rec2, store_file)


# --- Lineage, Secret Redaction & Validation Tests ---

def test_lineage_preservation_from_decision_to_publication() -> None:
    cand, dec, sig, risk, pub = make_test_artifacts()

    assert pub.signal_id == sig.signal_id
    assert pub.decision_id == dec.decision_id
    assert pub.candidate_id == cand.candidate_id
    assert pub.research_evidence_id == cand.evidence.evidence_id
    assert pub.research_fingerprint == cand.evidence.experiment_fingerprint
    assert pub.symbol == dec.symbol
    assert pub.decision == dec.direction.value
    assert pub.entry == dec.entry_price
    assert pub.stop_loss == risk.stop_loss
    assert pub.tp1 == risk.tp1


def test_secret_redaction() -> None:
    secret_key = "super-secret-api-key-99"
    text = f"HTTP Error with Authorization Bearer {secret_key}"
    redacted = _redact_secret(text, secret_key)

    assert secret_key not in redacted
    assert "[REDACTED]" in redacted


def test_publication_payload_contract_v1_conversion() -> None:
    _, _, _, _, pub = make_test_artifacts()
    payload = pub.to_contract_v1_payload()

    assert payload["contract_version"] == "1.0"
    assert payload["event_id"] == pub.publication_id
    assert payload["signal"]["publication_id"] == pub.publication_id
    assert payload["signal"]["signal_id"] == pub.signal_id
    assert payload["signal"]["decision_id"] == pub.decision_id
    assert payload["trade_setup"]["entry_price"] == pub.entry
    assert payload["trade_setup"]["stop_loss"] == pub.stop_loss
    assert payload["trade_setup"]["tp1"] == pub.tp1
