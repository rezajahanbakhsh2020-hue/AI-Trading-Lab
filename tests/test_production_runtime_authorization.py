import dataclasses
from datetime import datetime, timezone
import pytest

from src.evaluation.live_production_decision import (
    PromotedCandidateArtifact,
    ProductionRuntimeAuthorization,
    ProductionRuntimeAuthorizationError,
    authorize_production_runtime,
    PromotionStatus,
)
from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
    ProductionBlocked,
)
from tests.test_production_decision_integrity import make_promoted_evidence


def test_authorization_success():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_12345",
        campaign_selection_decision_fingerprint="camp_fp_67890",
    )

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    auth = authorize_production_runtime(
        cand,
        symbol="XAUUSD",
        timeframe="5m",
        now=now,
    )

    assert isinstance(auth, ProductionRuntimeAuthorization)
    assert auth.candidate_id == "cand_auth_01"
    assert auth.strategy_name == "momentum"
    assert auth.strategy_version == "1.0"
    assert auth.symbol == "XAUUSD"
    assert auth.timeframe == "5m"
    assert auth.governance_decision_fingerprint == "gov_fp_12345"
    assert auth.campaign_selection_decision_fingerprint == "camp_fp_67890"
    assert auth.promoted_artifact_fingerprint == cand.artifact_fingerprint
    assert auth.authorization_fingerprint is not None
    assert len(auth.authorization_fingerprint) == 64


def test_missing_governance_fingerprint_fails():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_02",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_valid",
    )

    # Force invalid governance_decision_fingerprint on artifact
    object.__setattr__(cand, "governance_decision_fingerprint", None)

    with pytest.raises(ProductionRuntimeAuthorizationError, match="governance_decision_fingerprint"):
        authorize_production_runtime(
            cand,
            symbol="XAUUSD",
            timeframe="5m",
        )


def test_mismatched_candidate_scope_fails():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_03",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_123",
    )

    with pytest.raises(ProductionRuntimeAuthorizationError, match="symbol"):
        authorize_production_runtime(
            cand,
            symbol="BTCUSD",
            timeframe="5m",
        )

    with pytest.raises(ProductionRuntimeAuthorizationError, match="timeframe"):
        authorize_production_runtime(
            cand,
            symbol="XAUUSD",
            timeframe="1h",
        )


def test_campaign_lineage_preservation_to_authorization():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_04",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_abc",
        campaign_selection_decision_fingerprint="camp_fp_xyz",
    )

    auth = authorize_production_runtime(
        cand,
        symbol="XAUUSD",
        timeframe="5m",
    )

    assert auth.campaign_selection_decision_fingerprint == "camp_fp_xyz"


def test_deterministic_authorization_fingerprint():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_05",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_det",
    )

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    auth1 = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=now)
    auth2 = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=now)

    assert auth1.authorization_fingerprint == auth2.authorization_fingerprint


def test_authorization_is_immutable():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_06",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_imm",
    )

    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")

    with pytest.raises(dataclasses.FrozenInstanceError):
        auth.candidate_id = "mutated_id"


def test_live_execution_runtime_blocks_if_authorization_fails(tmp_path):
    # Setup runtime with missing candidate/promotion
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "history.json",
        snapshot_path=tmp_path / "latest.json",
        production_config=ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="nonexistent_candidate",
            strategy_id="momentum",
            research_dir=tmp_path,
        )
    )

    res = runtime.run_once(publish=False, persist=True)

    assert res["blocked"] is True
    assert res["decision"] == "NO TRADE"
    assert "Promotion" in res["reason"] or "Unavailable" in res["reason"]
