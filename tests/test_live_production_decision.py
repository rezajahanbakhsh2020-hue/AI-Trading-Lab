import pandas as pd
import pytest

import dataclasses
from unittest.mock import patch

from src.evaluation.research_constitution import RejectionReason
from src.evaluation.live_production_decision import (
    DEFAULT_MIN_STABILITY_SCORE,
    AuthorizedLiveDecision,
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionRiskLevels,
    ProductionRuntimeAuthorizationError,
    PromotedCandidateArtifact,
    PromotionStatus,
    authorize_production_runtime,
    build_live_production_decision,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.live_trade_display import build_live_trade_display
from src.evaluation.research_store import persist_promoted_candidate_binding, save_research_experiment
from tests.test_production_decision_integrity import make_promoted_evidence


def make_rising_market(rows: int = 80) -> pd.DataFrame:
    close = [100.0 + index for index in range(rows)]
    now = pd.Timestamp.now(tz="UTC")
    timestamps = [now - pd.Timedelta(minutes=5 * (rows - 1 - i)) for i in range(rows)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
        }
    )


def test_stable_momentum_and_uptrend_produce_buy_decision():
    data = make_rising_market()

    result = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.517268,
    )

    assert result["decision"] == "BUY"
    assert (
        result["reason"]
        == "stable_strategy_live_signal_and_trend_confirmed"
    )
    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == pytest.approx(0.517268)
    assert result["signal"] == 1
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["entry_price"] > 0
    assert result["stop_loss"] < result["entry_price"]
    assert result["take_profit"] > result["entry_price"]
    assert result["risk_reward_ratio"] == pytest.approx(2.0)


def test_stability_score_below_threshold_blocks_trade():
    data = make_rising_market()

    result = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.49,
    )

    assert result["decision"] == "NO TRADE"
    assert result["reason"] == "stability_score_below_threshold"
    assert result["signal"] == 1
    assert result["trend"] == "UP"


def test_unsupported_stable_strategy_blocks_trade():
    data = make_rising_market()

    result = build_live_production_decision(
        data,
        stable_strategy="moving_average",
        stability_score=0.80,
    )

    assert result["decision"] == "NO TRADE"
    assert (
        result["reason"]
        == "stable_strategy_not_supported_by_live_signal"
    )
    assert result["strategy_supported"] is False


def test_downtrend_blocks_buy_decision():
    data = make_rising_market()

    data["close"] = list(range(200, 120, -1))
    data["open"] = data["close"]
    data["high"] = data["close"] + 1.0
    data["low"] = data["close"] - 1.0

    result = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert result["decision"] == "NO TRADE"
    assert result["reason"] == "trend_not_confirmed"
    assert result["signal"] == 0
    assert result["trend"] == "DOWN"


def test_no_trade_signal_is_preserved_without_inventing_sell():
    data = make_rising_market()

    data["close"] = [100.0] * len(data)
    data["open"] = data["close"]
    data["high"] = data["close"] + 1.0
    data["low"] = data["close"] - 1.0

    result = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
    )

    assert result["decision"] == "NO TRADE"
    assert result["signal"] == 0
    assert result["signal_label"] == "NO TRADE"
    assert result["decision"] != "SELL"
    assert result["stop_loss"] is None
    assert result["take_profit"] is None


def test_default_threshold_matches_current_stability_gate():
    assert DEFAULT_MIN_STABILITY_SCORE == 0.50


def test_invalid_stability_score_is_rejected():
    data = make_rising_market()

    with pytest.raises(ValueError, match="between 0 and 1"):
        build_live_production_decision(
            data,
            stable_strategy="momentum",
            stability_score=1.5,
        )


def test_empty_strategy_is_rejected():
    data = make_rising_market()

    with pytest.raises(ValueError, match="must not be empty"):
        build_live_production_decision(
            data,
            stable_strategy="",
            stability_score=0.80,
        )


# --- PR #47 Focused Tests (Tests 1 - 9) ---

def test_1_authorized_live_decision_creation_and_identity_consistency():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_auth_dec_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_test_01",
    )
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    data = make_rising_market()
    dec = evaluate_production_decision(cand, data, authorization=receipt)
    risk = calculate_production_risk_levels(dec, cand)

    authorized_decision = AuthorizedLiveDecision.create(
        authorization=receipt,
        candidate=cand,
        decision=dec,
        risk_levels=risk,
    )

    assert isinstance(authorized_decision, AuthorizedLiveDecision)
    assert authorized_decision.candidate_id == cand.candidate_id
    assert authorized_decision.strategy_name == cand.strategy_name
    assert authorized_decision.strategy_version == cand.strategy_version
    assert authorized_decision.symbol == cand.symbol
    assert authorized_decision.timeframe == cand.timeframe
    assert authorized_decision.promoted_artifact_fingerprint == cand.artifact_fingerprint
    assert authorized_decision.governance_decision_fingerprint == auth.governance_decision_fingerprint
    assert authorized_decision.authorization_fingerprint == auth.authorization_fingerprint
    assert authorized_decision.decision_artifact_fingerprint is not None
    assert len(authorized_decision.decision_artifact_fingerprint) == 64


def test_2_missing_candidate_produces_blocked_and_fails_closed(tmp_path):
    data = make_rising_market()
    with pytest.raises(ValueError, match="No authoritative promoted candidate resolved"):
        build_live_runtime(
            data,
            stable_strategy="momentum",
            stability_score=0.80,
            symbol="XAUUSD",
            interval="5m",
            candidate_id="nonexistent_candidate",
            research_dir=tmp_path,
        )


def test_3_authorization_failure_produces_no_authorized_decision(tmp_path):
    ev = make_promoted_evidence(PromotionStatus.REJECTED, rejection_reasons=(RejectionReason.FAILED_ROBUSTNESS,))
    save_research_experiment(ev, base_dir=tmp_path)

    data = make_rising_market()
    with pytest.raises(ValueError, match="not allowed for production"):
        PromotedCandidateArtifact("cand_rejected", "momentum", "1.0", ev, "XAUUSD", "5m")


def test_4_display_purity_consumes_authorized_live_decision():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_purity_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_purity_01",
    )
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    data = make_rising_market()
    dec = evaluate_production_decision(cand, data, authorization=receipt)
    risk = calculate_production_risk_levels(dec, cand)

    authorized_decision = AuthorizedLiveDecision.create(
        authorization=receipt,
        candidate=cand,
        decision=dec,
        risk_levels=risk,
    )

    display = build_live_trade_display(authorized_decision=authorized_decision)

    assert display["candidate_id"] == cand.candidate_id
    assert display["authorization_fingerprint"] == auth.authorization_fingerprint


def test_5_legacy_bypass_regression_build_live_runtime_uses_canonical_flow(tmp_path):
    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    save_research_experiment(ev, base_dir=tmp_path)
    persist_promoted_candidate_binding(
        candidate_id="cand_runtime_01",
        evidence=ev,
        base_dir=tmp_path,
        governance_decision=type("Gov", (), {"qualified": True, "experiment_fingerprint": ev.experiment_fingerprint, "decision_fingerprint": "gov_fp_runtime_01"})(),
    )

    data = make_rising_market()
    res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
        symbol="XAUUSD",
        interval="5m",
        candidate_id="cand_runtime_01",
        research_dir=tmp_path,
    )

    assert res.authorized_decision is not None
    assert isinstance(res.authorized_decision, AuthorizedLiveDecision)
    assert res.authorized_decision.candidate_id == "cand_runtime_01"


def test_6_lineage_preservation_in_authorized_live_decision():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        "cand_lineage_01", "momentum", "1.0", ev, "XAUUSD", "5m",
        governance_decision_fingerprint="gov_fp_lin_01",
        campaign_selection_decision_fingerprint="camp_fp_lin_01",
    )
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    data = make_rising_market()
    dec = evaluate_production_decision(cand, data, authorization=receipt)
    risk = calculate_production_risk_levels(dec, cand)

    authorized_decision = AuthorizedLiveDecision.create(
        authorization=receipt,
        candidate=cand,
        decision=dec,
        risk_levels=risk,
    )

    assert authorized_decision.candidate_id == cand.candidate_id
    assert authorized_decision.strategy_name == cand.strategy_name
    assert authorized_decision.strategy_version == cand.strategy_version
    assert authorized_decision.symbol == cand.symbol
    assert authorized_decision.timeframe == cand.timeframe
    assert authorized_decision.promoted_artifact_fingerprint == cand.artifact_fingerprint
    assert authorized_decision.governance_decision_fingerprint == "gov_fp_lin_01"
    assert authorized_decision.campaign_selection_decision_fingerprint == "camp_fp_lin_01"
    assert authorized_decision.authorization_fingerprint == auth.authorization_fingerprint


def test_7_authorized_live_decision_immutability():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact(
        candidate_id="cand_immut_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_immut_01",
    )
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    data = make_rising_market()
    dec = evaluate_production_decision(cand, data, authorization=receipt)
    risk = calculate_production_risk_levels(dec, cand)

    authorized_decision = AuthorizedLiveDecision.create(
        authorization=receipt,
        candidate=cand,
        decision=dec,
        risk_levels=risk,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        authorized_decision.candidate_id = "mutated_candidate"


def test_8_mismatched_authorization_attributes_fail_closed():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand1 = PromotedCandidateArtifact(
        candidate_id="cand_mismatch_01",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_mis_01",
    )
    cand2 = PromotedCandidateArtifact(
        candidate_id="cand_mismatch_02",
        strategy_name="momentum",
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_mis_02",
    )

    auth1 = authorize_production_runtime(cand1, symbol="XAUUSD", timeframe="5m")
    receipt1 = ProductionAuthorizationReceipt.from_authorization(auth1)

    data = make_rising_market()
    dec1 = evaluate_production_decision(cand1, data, authorization=receipt1)
    risk1 = calculate_production_risk_levels(dec1, cand1)

    # Candidate ID mismatch
    with pytest.raises(ProductionRuntimeAuthorizationError, match="candidate ID"):
        AuthorizedLiveDecision.create(
            authorization=receipt1,
            candidate=cand2,
            decision=dec1,
            risk_levels=risk1,
        )


def test_9_no_fabricated_fallback_for_missing_authorization():
    with pytest.raises(TypeError, match="authorized_decision must be an AuthorizedLiveDecision instance"):
        build_live_trade_display(authorized_decision=None)  # type: ignore[arg-type]
