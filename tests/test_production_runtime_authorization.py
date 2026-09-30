import dataclasses
from datetime import datetime, timezone
import pytest

import json
import pandas as pd
from unittest.mock import patch

from src.evaluation.live_production_decision import (
    PromotedCandidateArtifact,
    ProductionRuntimeAuthorization,
    ProductionAuthorizationReceipt,
    ProductionRuntimeAuthorizationError,
    ProductionDecision,
    ProductionSignal,
    ProductionRiskLevels,
    ProductionIntelligencePublication,
    Direction,
    authorize_production_runtime,
    PromotionStatus,
)
from src.evaluation.live_decision_store import (
    append_live_decision_to_store,
    load_live_decision_history,
)
from src.evaluation.live_publication_store import (
    append_publication_record,
    load_publication_history,
    PublicationIntegrityError,
)
from src.evaluation.research_store import persist_promoted_candidate_binding, save_research_experiment
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


def test_receipt_projection_and_invariants():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_receipt_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_rec",
        campaign_selection_decision_fingerprint="camp_fp_rec",
    )

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=now)

    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    assert isinstance(receipt, ProductionAuthorizationReceipt)
    assert receipt.candidate_id == auth.candidate_id
    assert receipt.strategy_name == auth.strategy_name
    assert receipt.strategy_version == auth.strategy_version
    assert receipt.symbol == auth.symbol
    assert receipt.timeframe == auth.timeframe
    assert receipt.promoted_artifact_fingerprint == auth.promoted_artifact_fingerprint
    assert receipt.governance_decision_fingerprint == auth.governance_decision_fingerprint
    assert receipt.campaign_selection_decision_fingerprint == auth.campaign_selection_decision_fingerprint
    assert receipt.authorization_policy_version == auth.authorization_policy_version
    assert receipt.authorized_at_utc == auth.authorized_at_utc
    assert receipt.authorization_fingerprint == auth.authorization_fingerprint

    # Test immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.candidate_id = "mutated_candidate"


def test_missing_mandatory_receipt_field_fails_closed():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_receipt_02",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_rec",
    )

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=now)

    # Missing candidate_id
    with pytest.raises(ProductionRuntimeAuthorizationError, match="candidate_id"):
        ProductionAuthorizationReceipt(
            candidate_id="",
            strategy_name=auth.strategy_name,
            strategy_version=auth.strategy_version,
            symbol=auth.symbol,
            timeframe=auth.timeframe,
            promoted_artifact_fingerprint=auth.promoted_artifact_fingerprint,
            governance_decision_fingerprint=auth.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=auth.campaign_selection_decision_fingerprint,
            authorization_policy_version=auth.authorization_policy_version,
            authorized_at_utc=auth.authorized_at_utc,
            authorization_fingerprint=auth.authorization_fingerprint,
        )

    # Missing authorization_fingerprint
    with pytest.raises(ProductionRuntimeAuthorizationError, match="authorization_fingerprint"):
        ProductionAuthorizationReceipt(
            candidate_id=auth.candidate_id,
            strategy_name=auth.strategy_name,
            strategy_version=auth.strategy_version,
            symbol=auth.symbol,
            timeframe=auth.timeframe,
            promoted_artifact_fingerprint=auth.promoted_artifact_fingerprint,
            governance_decision_fingerprint=auth.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=auth.campaign_selection_decision_fingerprint,
            authorization_policy_version=auth.authorization_policy_version,
            authorized_at_utc=auth.authorized_at_utc,
            authorization_fingerprint="",
        )


def test_runtime_persistence_contains_complete_lineage(tmp_path):
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_persist_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_persist",
        campaign_selection_decision_fingerprint="camp_fp_persist",
    )
    save_research_experiment(cand.evidence, base_dir=tmp_path)
    persist_promoted_candidate_binding(
        candidate_id=cand.candidate_id,
        evidence=cand.evidence,
        base_dir=tmp_path,
        governance_decision=type("Gov", (), {"qualified": True, "experiment_fingerprint": cand.evidence.experiment_fingerprint, "decision_fingerprint": cand.governance_decision_fingerprint})(),
        campaign_selection_decision=type("CampSel", (), {"decision_fingerprint": "camp_fp_persist", "campaign_id": "camp_1", "decision_status": "SELECTED", "selected_candidate_ids": ("cand_persist_01",)})(),
    )

    store_p = tmp_path / "decision_history.json"
    snapshot_p = tmp_path / "latest_execution.json"
    pub_p = tmp_path / "publication_history.json"

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=store_p,
        snapshot_path=snapshot_p,
        research_dir=tmp_path,
        production_config=ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_persist_01",
            strategy_id="momentum",
            research_dir=tmp_path,
        )
    )

    now_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    candle_ts = now_dt.isoformat()
    mock_df = pd.DataFrame([{
        "openTime": candle_ts,
        "timestamp": pd.to_datetime(candle_ts),
        "open": 2000.0,
        "high": 2010.0,
        "low": 1990.0,
        "close": 2005.0,
    }])

    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=mock_df):
        res = runtime.run_once(publish=False, persist=True, reference_now=now_dt)

    assert res["blocked"] is False

    # Check decision_history.json
    dec_hist = load_live_decision_history(store_p)
    assert len(dec_hist) == 1
    rec = dec_hist[0]
    assert rec["runtime_authorization_fingerprint"] == res["runtime_authorization_fingerprint"]
    assert rec["governance_decision_fingerprint"] == "gov_fp_persist"
    assert rec["campaign_selection_decision_fingerprint"] == "camp_fp_persist"
    assert rec["candidate_id"] == "cand_persist_01"

    # Check latest_execution.json snapshot
    snapshot_data = json.loads(snapshot_p.read_text(encoding="utf-8"))
    assert "runtime_authorization" in snapshot_data
    auth_dict = snapshot_data["runtime_authorization"]
    assert auth_dict["candidate_id"] == "cand_persist_01"
    assert auth_dict["governance_decision_fingerprint"] == "gov_fp_persist"
    assert auth_dict["campaign_selection_decision_fingerprint"] == "camp_fp_persist"
    assert auth_dict["authorization_fingerprint"] == res["runtime_authorization_fingerprint"]

    # Check publication_history.json
    pub_hist = load_publication_history(pub_p)
    assert len(pub_hist) == 1
    pub = pub_hist[0]
    prov = pub["provenance"]
    assert prov["runtime_authorization_fingerprint"] == res["runtime_authorization_fingerprint"]
    assert prov["governance_decision_fingerprint"] == "gov_fp_persist"
    assert prov["campaign_selection_decision_fingerprint"] == "camp_fp_persist"


def test_replay_idempotency_and_conflict_safety(tmp_path):
    store_p = tmp_path / "decision_history.json"
    pub_p = tmp_path / "publication_history.json"

    rec1 = {
        "timestamp": "2026-01-01T12:00:00+00:00",
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 2000.0,
        "stop_loss": 1980.0,
        "take_profit": 2040.0,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.85,
        "market_state": "OPEN",
        "quote_age_seconds": 10.0,
        "quote_stale": False,
        "candle_count": 100,
        "decision_id": "dec_replay_01",
        "signal_id": "sig_replay_01",
        "runtime_authorization_fingerprint": "auth_fp_111",
        "authorization_policy_version": "runtime_auth_v1.0",
        "authorized_at_utc": "2026-01-01T12:00:00+00:00",
        "promoted_artifact_fingerprint": "art_fp_111",
        "governance_decision_fingerprint": "gov_fp_111",
        "campaign_selection_decision_fingerprint": "camp_fp_111",
        "candidate_id": "cand_01",
        "strategy_name": "momentum",
        "strategy_version": "1.0",
    }

    # 1. Append initial decision
    append_live_decision_to_store(rec1, store_p)

    # 2. Replay identical record -> idempotent
    append_live_decision_to_store(rec1, store_p)
    hist = load_live_decision_history(store_p)
    assert len(hist) == 1

    # 3. Conflicting replay (same decision_id, different authorization fingerprint) -> fail closed
    conflicting_rec = dict(rec1)
    conflicting_rec["runtime_authorization_fingerprint"] = "auth_fp_DIFFERENT"
    with pytest.raises(ValueError, match="Conflicting replay"):
        append_live_decision_to_store(conflicting_rec, store_p)

    # 4. Asymmetric replay: existing authorized + replay missing authorization -> fail closed
    rec1_no_auth = {k: v for k, v in rec1.items() if k not in ("runtime_authorization_fingerprint", "authorization_policy_version", "authorized_at_utc", "promoted_artifact_fingerprint", "governance_decision_fingerprint", "campaign_selection_decision_fingerprint", "candidate_id", "strategy_name", "strategy_version")}
    with pytest.raises(ValueError, match="Conflicting replay"):
        append_live_decision_to_store(rec1_no_auth, store_p)

    # 5. Asymmetric replay: existing no-auth + replay authorized -> fail closed
    store_unauth_p = tmp_path / "decision_history_unauth.json"
    legacy_rec = {
        "timestamp": "2026-01-01T12:00:00+00:00",
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 2000.0,
        "stop_loss": 1980.0,
        "take_profit": 2040.0,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.85,
        "market_state": "OPEN",
        "quote_age_seconds": 10.0,
        "quote_stale": False,
        "candle_count": 100,
        "decision_id": "dec_asym_01",
        "signal_id": "sig_asym_01",
    }
    append_live_decision_to_store(legacy_rec, store_unauth_p)

    legacy_replay_authorized = dict(rec1)
    legacy_replay_authorized["decision_id"] = "dec_asym_01"
    legacy_replay_authorized["signal_id"] = "sig_asym_01"
    with pytest.raises(ValueError, match="Conflicting replay"):
        append_live_decision_to_store(legacy_replay_authorized, store_unauth_p)

    # 6. Publication store replay check
    pub_rec = {
        "publication_id": "pub_01",
        "signal_id": "sig_replay_01",
        "decision_id": "dec_replay_01",
        "symbol": "XAUUSD",
        "timeframe": "5m",
        "decision": "BUY",
        "entry": 2000.0,
        "stop_loss": 1980.0,
        "tp1": 2040.0,
        "provenance": {
            "runtime_authorization_fingerprint": "auth_fp_111",
            "governance_decision_fingerprint": "gov_fp_111",
        }
    }
    append_publication_record(pub_rec, pub_p)

    # Replay identical publication -> idempotent
    append_publication_record(pub_rec, pub_p)
    pub_hist = load_publication_history(pub_p)
    assert len(pub_hist) == 1

    # Conflicting publication (different auth fingerprint) -> fail closed
    conflicting_pub = json.loads(json.dumps(pub_rec))
    conflicting_pub["provenance"]["runtime_authorization_fingerprint"] = "auth_fp_DIFFERENT"
    with pytest.raises(PublicationIntegrityError, match="Conflicting publication replay"):
        append_publication_record(conflicting_pub, pub_p)


def test_single_authorization_invocation_in_runtime(tmp_path):
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_single_auth",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_single",
    )
    save_research_experiment(cand.evidence, base_dir=tmp_path)
    persist_promoted_candidate_binding(
        candidate_id=cand.candidate_id,
        evidence=cand.evidence,
        base_dir=tmp_path,
        governance_decision=type("Gov", (), {"qualified": True, "experiment_fingerprint": cand.evidence.experiment_fingerprint, "decision_fingerprint": cand.governance_decision_fingerprint})(),
    )

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "decision_history.json",
        snapshot_path=tmp_path / "latest_execution.json",
        research_dir=tmp_path,
        production_config=ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_single_auth",
            strategy_id="momentum",
            research_dir=tmp_path,
        )
    )

    now_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_df = pd.DataFrame([{
        "openTime": now_dt.isoformat(),
        "timestamp": pd.to_datetime(now_dt.isoformat()),
        "open": 2000.0,
        "high": 2010.0,
        "low": 1990.0,
        "close": 2005.0,
    }])

    with patch("src.evaluation.live_execution_runtime.authorize_production_runtime", wraps=authorize_production_runtime) as mock_auth, \
         patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=mock_df):
        res = runtime.run_once(publish=False, persist=True, reference_now=now_dt)

    assert res["blocked"] is False
    assert mock_auth.call_count == 1


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
