"""Comprehensive Adversarial Unit and Integration Tests for Project 1 MTF Intelligence."""

import datetime
from unittest.mock import MagicMock

import pytest

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionSignal,
    PromotedCandidateArtifact,
)
from src.evaluation.mtf_intelligence import (
    CanonicalTimeframe,
    HigherTimeframeContext,
    MTFClassification,
    MTFIntelligence,
    PerTimeframeSignal,
    build_mtf_intelligence,
)


def _make_signal(
    timeframe: str,
    direction: Direction,
    symbol: str = "XAUUSD",
    decision_id: str | None = None,
    signal_id: str | None = None,
    candidate_id: str = "cand_test_001",
    strategy_name: str = "momentum",
    strategy_version: str = "1.0",
) -> PerTimeframeSignal:
    tf_enum = CanonicalTimeframe.from_str(timeframe)
    dec_id = decision_id or f"dec_{tf_enum.value}"
    sig_id = signal_id or f"sig_{tf_enum.value}"
    return PerTimeframeSignal(
        symbol=symbol,
        timeframe=tf_enum,
        direction=direction,
        decision_id=dec_id,
        signal_id=sig_id,
        decision_timestamp="2026-03-30T12:00:00+00:00",
        market_timestamp="2026-03-30T11:59:00+00:00",
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        candidate_id=candidate_id,
        evidence_id="ev_test_001",
        experiment_fingerprint="exp_fp_001",
        canonical_live_decision_fingerprint=f"cld_fp_{tf_enum.value}",
        authorization_fingerprint=f"auth_fp_{tf_enum.value}",
        provenance={"test": True},
    )


def test_phase1_canonical_timeframe_ordering_and_parsing():
    """Phase 1: Canonical ordering and fail-closed timeframe parsing."""
    assert CanonicalTimeframe.FIVE_MINUTES < CanonicalTimeframe.FIFTEEN_MINUTES
    assert CanonicalTimeframe.FIFTEEN_MINUTES < CanonicalTimeframe.THIRTY_MINUTES
    assert CanonicalTimeframe.THIRTY_MINUTES < CanonicalTimeframe.ONE_HOUR
    assert CanonicalTimeframe.ONE_HOUR < CanonicalTimeframe.FOUR_HOURS
    assert CanonicalTimeframe.FOUR_HOURS < CanonicalTimeframe.ONE_DAY

    # Parsing valid
    assert CanonicalTimeframe.from_str("5m") == CanonicalTimeframe.FIVE_MINUTES
    assert CanonicalTimeframe.from_str("15m") == CanonicalTimeframe.FIFTEEN_MINUTES
    assert CanonicalTimeframe.from_str("30m") == CanonicalTimeframe.THIRTY_MINUTES
    assert CanonicalTimeframe.from_str("1H") == CanonicalTimeframe.ONE_HOUR
    assert CanonicalTimeframe.from_str("1h") == CanonicalTimeframe.ONE_HOUR
    assert CanonicalTimeframe.from_str("4H") == CanonicalTimeframe.FOUR_HOURS
    assert CanonicalTimeframe.from_str("1D") == CanonicalTimeframe.ONE_DAY

    # Fail closed on unknown/invalid
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe"):
        CanonicalTimeframe.from_str("10m")

    with pytest.raises(ValueError, match="Unknown or unsupported timeframe"):
        CanonicalTimeframe.from_str("2H")

    with pytest.raises(ValueError, match="Invalid timeframe value"):
        CanonicalTimeframe.from_str("")


def test_phase11_a_full_alignment():
    """Phase 11.A: Full alignment across 5m, 15m, 30m, 1H, 4H, 1D BUY."""
    signals = [
        _make_signal("5m", Direction.BUY),
        _make_signal("15m", Direction.BUY),
        _make_signal("30m", Direction.BUY),
        _make_signal("1H", Direction.BUY),
        _make_signal("4H", Direction.BUY),
        _make_signal("1D", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_count == 6
    assert mtf.alignment_coverage == 6
    assert mtf.classification == MTFClassification.ALIGNED
    assert mtf.star_representation == "⭐⭐⭐⭐⭐⭐"
    assert len(mtf.higher_timeframe_context.agreed_higher_timeframes) == 5
    assert len(mtf.higher_timeframe_context.disagreed_higher_timeframes) == 0


def test_phase11_b_counter_trend():
    """Phase 11.B: Counter-trend scenario with higher-timeframe disagreement."""
    signals = [
        _make_signal("1D", Direction.SELL),
        _make_signal("4H", Direction.SELL),
        _make_signal("1H", Direction.SELL),
        _make_signal("30m", Direction.SELL),
        _make_signal("15m", Direction.BUY),
        _make_signal("5m", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    # Local 5m decision remains BUY
    sig_5m = next(s for s in mtf.signals if s.timeframe == CanonicalTimeframe.FIVE_MINUTES)
    assert sig_5m.direction == Direction.BUY

    # 15m decision remains BUY
    sig_15m = next(s for s in mtf.signals if s.timeframe == CanonicalTimeframe.FIFTEEN_MINUTES)
    assert sig_15m.direction == Direction.BUY

    assert mtf.classification == MTFClassification.COUNTER_TREND
    assert mtf.alignment_coverage == 2
    assert CanonicalTimeframe.FIFTEEN_MINUTES in mtf.higher_timeframe_context.agreed_higher_timeframes
    assert CanonicalTimeframe.ONE_DAY in mtf.higher_timeframe_context.disagreed_higher_timeframes


def test_phase11_c_local_only():
    """Phase 11.C: Local timeframe signal only (5m BUY)."""
    signals = [_make_signal("5m", Direction.BUY)]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_coverage == 1
    assert mtf.classification == MTFClassification.INSUFFICIENT_CONTEXT
    assert mtf.star_representation == "⭐"
    assert mtf.classification != MTFClassification.ALIGNED


def test_phase11_d_missing_timeframe():
    """Phase 11.D: Missing timeframe (15m absent)."""
    signals = [
        _make_signal("5m", Direction.BUY),
        _make_signal("30m", Direction.BUY),
        _make_signal("1H", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert CanonicalTimeframe.FIFTEEN_MINUTES in mtf.higher_timeframe_context.missing_higher_timeframes
    assert CanonicalTimeframe.FIFTEEN_MINUTES not in mtf.higher_timeframe_context.present_higher_timeframes
    assert mtf.alignment_coverage == 3
    assert mtf.alignment_coverage != 6  # No silent six-star fabricated alignment


def test_phase11_e_duplicate_timeframe():
    """Phase 11.E: Duplicate timeframe inputs fail closed."""
    signals = [
        _make_signal("5m", Direction.BUY, decision_id="dec_5m_1"),
        _make_signal("5m", Direction.BUY, decision_id="dec_5m_2"),
    ]
    with pytest.raises(ValueError, match="Duplicate timeframe signal provided"):
        build_mtf_intelligence(signals, local_timeframe="5m")


def test_phase11_f_unknown_timeframe():
    """Phase 11.F: Unknown timeframe string fails closed."""
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe"):
        build_mtf_intelligence([_make_signal("5m", Direction.BUY)], local_timeframe="99m")


def test_phase11_g_symbol_mix():
    """Phase 11.G: Mismatched symbols across signals fail closed."""
    signals = [
        _make_signal("5m", Direction.BUY, symbol="XAUUSD"),
        _make_signal("15m", Direction.BUY, symbol="BTCUSDT"),
    ]
    with pytest.raises(ValueError, match="Mismatched symbols across signals"):
        build_mtf_intelligence(signals, local_timeframe="5m")


def test_phase11_h_identity_separation():
    """Phase 11.H: 5m BUY and 15m BUY with identical metadata have distinct constituent fingerprints."""
    sig5m = _make_signal("5m", Direction.BUY, decision_id="dec_same", signal_id="sig_same")
    sig15m = _make_signal("15m", Direction.BUY, decision_id="dec_same", signal_id="sig_same")

    assert sig5m.timeframe == CanonicalTimeframe.FIVE_MINUTES
    assert sig15m.timeframe == CanonicalTimeframe.FIFTEEN_MINUTES
    assert sig5m.constituent_fingerprint != sig15m.constituent_fingerprint


def test_phase11_i_immutability():
    """Phase 11.I: MTFIntelligence domain model is strictly immutable."""
    signals = [_make_signal("5m", Direction.BUY)]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    with pytest.raises(Exception):
        mtf.alignment_coverage = 5  # Frozen dataclass mutation failure

    with pytest.raises(Exception):
        mtf.signals.append(_make_signal("15m", Direction.BUY))  # Tuple has no append


def test_phase11_j_k_determinism_and_sensitivity():
    """Phase 11.J & K: Determinism and fingerprint sensitivity."""
    sig1 = _make_signal("5m", Direction.BUY)
    sig2 = _make_signal("15m", Direction.BUY)

    mtf1 = build_mtf_intelligence([sig1, sig2], local_timeframe="5m")
    mtf2 = build_mtf_intelligence([sig1, sig2], local_timeframe="5m")

    # J: Deterministic fingerprint
    assert mtf1.intelligence_fingerprint == mtf2.intelligence_fingerprint

    # K: Changing one constituent signal changes fingerprint
    sig2_changed = _make_signal("15m", Direction.SELL)
    mtf_changed = build_mtf_intelligence([sig1, sig2_changed], local_timeframe="5m")

    assert mtf1.intelligence_fingerprint != mtf_changed.intelligence_fingerprint


def test_phase11_l_source_preservation():
    """Phase 11.L: MTF construction does not mutate source ProductionDecision or CanonicalLiveDecision."""
    dec = ProductionDecision(
        candidate_id="cand_test",
        evidence_id="ev_test",
        experiment_fingerprint="exp_fp",
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp="2026-03-30T12:00:00+00:00",
        market_timestamp="2026-03-30T11:59:00+00:00",
        direction=Direction.BUY,
        reason="test",
        entry_price=2000.0,
        invalidation_condition="Close below SL",
        confidence=0.9,
    )
    receipt = ProductionAuthorizationReceipt(
        operational_stability_score=0.9,
        candidate_id="cand_test",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        promoted_artifact_fingerprint="prom_fp",
        governance_decision_fingerprint="gov_fp",
        campaign_selection_decision_fingerprint="csd_fp",
        authorization_policy_version="1.0",
        authorized_at_utc="2026-03-30T12:00:00+00:00",
        authorization_fingerprint="auth_fp",
    )
    sig = ProductionSignal.from_decision(dec)
    risk = ProductionRiskLevels(
        decision_id=dec.decision_id,
        candidate_id="cand_test",
        evidence_id="ev_test",
        symbol="XAUUSD",
        timeframe="5m",
        direction=Direction.BUY,
        entry_price=2000.0,
        stop_loss=1990.0,
        tp1=2010.0,
        tp2=2020.0,
        tp3=2030.0,
        risk_reward_ratio=2.0,
    )

    cld = create_canonical_live_decision(
        authorization_receipt=receipt,
        decision=dec,
        signal=sig,
        risk_levels=risk,
        actor="test",
        timestamp_utc="2026-03-30T12:00:00+00:00",
        reason="test_init",
    )

    ptf_sig = PerTimeframeSignal.from_canonical_live_decision(cld)
    build_mtf_intelligence([ptf_sig], local_timeframe="5m")

    # Assert source objects remain completely unchanged
    assert dec.direction == Direction.BUY
    assert sig.direction == Direction.BUY
    assert cld.decision.direction == Direction.BUY


def test_phase11_n_publication_regression():
    """Phase 11.N: Publication provenance and Contract v1 payload preserve required fields."""
    sig5m = _make_signal("5m", Direction.BUY)
    mtf = build_mtf_intelligence([sig5m], local_timeframe="5m")

    dec = ProductionDecision(
        candidate_id="cand_test",
        evidence_id="ev_test",
        experiment_fingerprint="exp_fp",
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp="2026-03-30T12:00:00+00:00",
        market_timestamp="2026-03-30T11:59:00+00:00",
        direction=Direction.BUY,
        reason="test",
        entry_price=2000.0,
        invalidation_condition="Close below SL",
        confidence=0.9,
    )
    sig = ProductionSignal.from_decision(dec)
    risk = ProductionRiskLevels(
        decision_id=dec.decision_id,
        candidate_id="cand_test",
        evidence_id="ev_test",
        symbol="XAUUSD",
        timeframe="5m",
        direction=Direction.BUY,
        entry_price=2000.0,
        stop_loss=1990.0,
        tp1=2010.0,
        tp2=2020.0,
        tp3=2030.0,
        risk_reward_ratio=2.0,
    )
    cand = MagicMock(spec=PromotedCandidateArtifact)
    cand.strategy_name = "momentum"
    cand.candidate_id = "cand_test"
    cand.evidence = MagicMock()
    cand.evidence.evidence_id = "ev_test"
    cand.evidence.experiment_fingerprint = "exp_fp"
    cand.operational_stability_score = 0.9
    cand.artifact_fingerprint = "art_fp"
    cand.policy = MagicMock()
    cand.policy.policy_version = "1.0"
    cand.strategy_version = "1.0"
    cand.campaign_selection_decision_fingerprint = "csd_fp"
    cand.governance_decision_fingerprint = "gov_fp"
    cand.parameters = {"stop_loss_pct": 0.01}

    pub = ProductionIntelligencePublication.from_artifacts(
        decision=dec,
        signal=sig,
        risk=risk,
        candidate=cand,
        mtf_intelligence=mtf,
    )

    assert pub.provenance["source"] == "AI-Trading-Lab"
    assert pub.provenance["provenance_type"] == "live_signal"
    assert pub.provenance["is_live"] is True

    payload = pub.to_contract_v1_payload()
    assert payload["event_type"] == "TRADING_SIGNAL"
    assert payload["signal"]["decision"] == "BUY"
    assert "mtf" in payload
    assert payload["mtf"]["star_representation"] == "⭐"
