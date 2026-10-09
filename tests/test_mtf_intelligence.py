"""Comprehensive Adversarial Unit and Integration Tests for Project 1 MTF Intelligence."""

from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation import live_runtime as live_rt_mod
from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
    transition_live_decision,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionRuntimeAuthorization,
    ProductionSignal,
    PromotedCandidateArtifact,
    calculate_production_risk_levels,
)
from src.evaluation.live_publication_store import publish_canonical_live_decision
from src.evaluation.mtf_intelligence import (
    CanonicalTimeframe,
    HigherTimeframeContext,
    MTFClassification,
    MTFIntelligence,
    MTFLiveRuntimeResult,
    PerTimeframeSignal,
    build_mtf_intelligence,
    evaluate_mtf_live_runtime,
)


def _make_signal(
    timeframe: str,
    direction: Direction,
    symbol: str = "XAUUSD",
    decision_id: str | None = None,
    signal_id: str | None = None,
    candidate_id: str | None = None,
    strategy_name: str = "momentum",
    strategy_version: str = "1.0",
) -> PerTimeframeSignal:
    tf_enum = CanonicalTimeframe.from_str(timeframe)
    dec_id = decision_id or f"dec_{tf_enum.value}"
    sig_id = signal_id or f"sig_{tf_enum.value}"
    cand_id = candidate_id or f"cand_{tf_enum.value}"
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
        candidate_id=cand_id,
        evidence_id="ev_test_001",
        experiment_fingerprint="exp_fp_001",
        canonical_live_decision_fingerprint=f"cld_fp_{tf_enum.value}",
        authorization_fingerprint=f"auth_fp_{tf_enum.value}",
        provenance={"tf": tf_enum.value},
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


def test_phase11_a_full_six_timeframe_alignment():
    """Phase 11.A: Full 6-timeframe alignment across 5m, 15m, 30m, 1H, 4H, 1D BUY."""
    signals = [
        _make_signal("5m", Direction.BUY),
        _make_signal("15m", Direction.BUY),
        _make_signal("30m", Direction.BUY),
        _make_signal("1H", Direction.BUY),
        _make_signal("4H", Direction.BUY),
        _make_signal("1D", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_coverage == 6
    assert mtf.star_representation == "⭐⭐⭐⭐⭐⭐"
    assert mtf.classification == MTFClassification.ALIGNED


def test_phase11_b_five_timeframe_alignment():
    """Phase 11.B: 5-timeframe alignment with 1D missing."""
    signals = [
        _make_signal("5m", Direction.BUY),
        _make_signal("15m", Direction.BUY),
        _make_signal("30m", Direction.BUY),
        _make_signal("1H", Direction.BUY),
        _make_signal("4H", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_coverage == 5
    assert mtf.star_representation == "⭐⭐⭐⭐⭐"
    assert mtf.classification == MTFClassification.ALIGNED


def test_phase11_c_gap_in_ladder():
    """Phase 11.C: Gap in ladder (15m missing between 5m and 30m) produces 1 star."""
    signals = [
        _make_signal("5m", Direction.BUY),
        _make_signal("30m", Direction.BUY),
        _make_signal("1H", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_coverage == 1
    assert mtf.star_representation == "⭐"
    assert mtf.matching_signal_count == 3


def test_phase11_d_noncontiguous_higher_signal():
    """Phase 11.D: Noncontiguous higher signal (1H missing) stops contiguous ladder at 3 stars."""
    signals = [
        _make_signal("5m", Direction.BUY),
        _make_signal("15m", Direction.BUY),
        _make_signal("30m", Direction.BUY),
        _make_signal("4H", Direction.BUY),
        _make_signal("1D", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_coverage == 3
    assert mtf.star_representation == "⭐⭐⭐"


def test_phase11_e_counter_trend():
    """Phase 11.E: Counter-trend scenario preserves lower-timeframe BUY decision."""
    signals = [
        _make_signal("1D", Direction.SELL),
        _make_signal("4H", Direction.SELL),
        _make_signal("1H", Direction.SELL),
        _make_signal("30m", Direction.SELL),
        _make_signal("15m", Direction.BUY),
        _make_signal("5m", Direction.BUY),
    ]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    sig_5m = next(s for s in mtf.signals if s.timeframe == CanonicalTimeframe.FIVE_MINUTES)
    assert sig_5m.direction == Direction.BUY

    sig_15m = next(s for s in mtf.signals if s.timeframe == CanonicalTimeframe.FIFTEEN_MINUTES)
    assert sig_15m.direction == Direction.BUY

    assert mtf.classification == MTFClassification.COUNTER_TREND
    assert mtf.alignment_coverage == 2


def test_phase11_f_local_only():
    """Phase 11.F: Local timeframe signal only (5m BUY)."""
    signals = [_make_signal("5m", Direction.BUY)]
    mtf = build_mtf_intelligence(signals, local_timeframe="5m")

    assert mtf.alignment_coverage == 1
    assert mtf.classification == MTFClassification.INSUFFICIENT_CONTEXT
    assert mtf.star_representation == "⭐"


def test_phase11_g_no_local_signal():
    """Phase 11.G: Missing local timeframe signal fails closed."""
    signals = [_make_signal("15m", Direction.BUY)]
    with pytest.raises(ValueError, match="Local timeframe '5m' signal not found"):
        build_mtf_intelligence(signals, local_timeframe="5m")


def test_phase11_h_duplicate_timeframe():
    """Phase 11.H: Duplicate timeframe inputs fail closed."""
    signals = [
        _make_signal("5m", Direction.BUY, decision_id="dec_5m_1"),
        _make_signal("5m", Direction.BUY, decision_id="dec_5m_2"),
    ]
    with pytest.raises(ValueError, match="Duplicate timeframe signal provided"):
        build_mtf_intelligence(signals, local_timeframe="5m")


def test_phase11_i_unknown_timeframe():
    """Phase 11.I: Unknown timeframe string fails closed."""
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe"):
        build_mtf_intelligence([_make_signal("5m", Direction.BUY)], local_timeframe="99m")


def test_phase11_j_mixed_symbols():
    """Phase 11.J: Mismatched symbols across signals fail closed."""
    signals = [
        _make_signal("5m", Direction.BUY, symbol="XAUUSD"),
        _make_signal("15m", Direction.BUY, symbol="BTCUSDT"),
    ]
    with pytest.raises(ValueError, match="Mismatched symbols across signals"):
        build_mtf_intelligence(signals, local_timeframe="5m")


def test_phase11_k_distinct_timeframe_identity():
    """Phase 11.K: 5m BUY and 15m BUY with otherwise identical metadata have distinct constituent fingerprints."""
    sig5m = _make_signal("5m", Direction.BUY, decision_id="dec_same", signal_id="sig_same")
    sig15m = _make_signal("15m", Direction.BUY, decision_id="dec_same", signal_id="sig_same")

    assert sig5m.timeframe == CanonicalTimeframe.FIVE_MINUTES
    assert sig15m.timeframe == CanonicalTimeframe.FIFTEEN_MINUTES
    assert sig5m.constituent_fingerprint != sig15m.constituent_fingerprint


def test_phase11_l_m_fingerprint_determinism_and_sensitivity():
    """Phase 11.L & M: Determinism and fingerprint sensitivity."""
    sig1 = _make_signal("5m", Direction.BUY)
    sig2 = _make_signal("15m", Direction.BUY)

    mtf1 = build_mtf_intelligence([sig1, sig2], local_timeframe="5m")
    mtf2 = build_mtf_intelligence([sig1, sig2], local_timeframe="5m")

    # L: Deterministic fingerprint
    assert mtf1.intelligence_fingerprint == mtf2.intelligence_fingerprint

    # M: Sensitivity: changing one constituent decision changes fingerprint
    sig2_changed = _make_signal("15m", Direction.SELL)
    mtf_changed = build_mtf_intelligence([sig1, sig2_changed], local_timeframe="5m")

    assert mtf1.intelligence_fingerprint != mtf_changed.intelligence_fingerprint


def test_phase11_n_nested_immutability():
    """Phase 11.N: Attempt to mutate nested provenance or result mapping raises TypeError."""
    sig = _make_signal("5m", Direction.BUY)
    assert isinstance(sig.provenance, MappingProxyType)

    with pytest.raises(TypeError):
        sig.provenance["new_key"] = "hacked"

    mtf = build_mtf_intelligence([sig], local_timeframe="5m")
    res = MTFLiveRuntimeResult(
        local_result="mock",
        mtf_intelligence=mtf,
        per_timeframe_results={CanonicalTimeframe.FIVE_MINUTES: "mock"},
    )
    assert isinstance(res.per_timeframe_results, MappingProxyType)

    with pytest.raises(TypeError):
        res.per_timeframe_results[CanonicalTimeframe.FIVE_MINUTES] = "hacked"


def test_phase11_o_source_preservation():
    """Phase 11.O: MTF construction does not mutate source ProductionDecision or CanonicalLiveDecision."""
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

    # Source objects remain unchanged
    assert dec.direction == Direction.BUY
    assert sig.direction == Direction.BUY
    assert cld.decision.direction == Direction.BUY


def test_phase11_p_authorization_isolation():
    """Phase 11.P: Each timeframe signal preserves its own candidate_id, timeframe, and authorization_fingerprint."""
    sig5m = _make_signal("5m", Direction.BUY, candidate_id="cand_5m")
    sig15m = _make_signal("15m", Direction.BUY, candidate_id="cand_15m")

    assert sig5m.candidate_id == "cand_5m"
    assert sig15m.candidate_id == "cand_15m"
    assert sig5m.authorization_fingerprint != sig15m.authorization_fingerprint


def test_phase11_q_r_publication_attachment():
    """Phase 11.Q & R: MTF artifact attaches to real authoritative publication path without second event."""
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
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02}

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

    risk = calculate_production_risk_levels(decision=dec, candidate=cand)

    cld = create_canonical_live_decision(
        authorization_receipt=receipt,
        decision=dec,
        signal=sig,
        risk_levels=risk,
        actor="test",
        timestamp_utc="2026-03-30T12:00:00+00:00",
        reason="test_init",
    )
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-03-30T12:00:00+00:00", reason="eval")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test", timestamp_utc="2026-03-30T12:00:00+00:00", reason="risk")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test", timestamp_utc="2026-03-30T12:00:00+00:00", reason="pres")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PERSISTED, actor="test", timestamp_utc="2026-03-30T12:00:00+00:00", reason="pers")

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
        "event_id": "pub_123",
        "remote_event_id": "pub_123",
    }

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        pub_path = Path(tmpdir) / "publication_history.json"
        pub_cld, pub_art, pub_res = publish_canonical_live_decision(
            canonical_decision=cld,
            publisher=mock_publisher,
            candidate=cand,
            path=pub_path,
            mtf_intelligence=mtf,
        )

        # R: Exactly ONE publication call
        assert mock_publisher.publish.call_count == 1
        published_pub = mock_publisher.publish.call_args[0][0]

        # Q: Attachment to real publication path
        assert published_pub.mtf_intelligence is not None
        assert published_pub.mtf_intelligence.star_representation == "⭐"

        payload = published_pub.to_contract_v1_payload()
        assert "mtf" in payload
        assert payload["mtf"]["star_representation"] == "⭐"


def test_phase11_t_provenance_regression():
    """Phase 11.T: Authoritative publication provenance remains intact."""
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
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02}

    risk = calculate_production_risk_levels(decision=dec, candidate=cand)

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


def test_phase11_exactly_once_evaluation_and_identity_continuity():
    """Anti-recurrence test: Proves each timeframe is evaluated EXACTLY ONCE and identity is continuous."""
    eval_calls = []

    orig_eval = live_rt_mod.evaluate_authorized_live_runtime

    def spy_eval(*args, **kwargs):
        ctx = kwargs.get("context")
        if ctx is not None:
            eval_calls.append((ctx.timeframe, ctx.candidate_id))
        return orig_eval(*args, **kwargs)

    timeframes = ["5m", "15m", "30m", "1H", "4H", "1D"]
    reference_now = pd.Timestamp("2026-03-31T12:00:00Z")
    interval_durations = {"5m": "5min", "15m": "15min", "30m": "30min", "1H": "1h", "4H": "4h", "1D": "1 day"}
    data_by_tf = {}
    for tf in timeframes:
        duration = pd.Timedelta(interval_durations[tf])
        dates = pd.date_range(end=reference_now - duration, periods=50, freq=duration)
        data_by_tf[tf] = pd.DataFrame({
            "openTime": [d.isoformat() for d in dates],
            "open": [2000.0 + i for i in range(50)],
            "high": [2005.0 + i for i in range(50)],
            "low": [1995.0 + i for i in range(50)],
            "close": [2002.0 + i for i in range(50)],
        })

    cand = MagicMock(spec=PromotedCandidateArtifact)
    cand.strategy_name = "momentum"
    cand.candidate_id = "cand_test"
    cand.symbol = "XAUUSD"
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
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02, "window": 10}

    def mock_auth(candidate, symbol, timeframe, now=None):
        candidate.timeframe = timeframe
        return ProductionRuntimeAuthorization(
            operational_stability_score=0.9,
            candidate_id=candidate.candidate_id,
            strategy_name=candidate.strategy_name,
            strategy_version=candidate.strategy_version,
            symbol=symbol,
            timeframe=timeframe,
            promoted_artifact_fingerprint=candidate.artifact_fingerprint,
            governance_decision_fingerprint=candidate.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=candidate.campaign_selection_decision_fingerprint,
            authorization_policy_version="1.0",
            authorized_at_utc="2026-03-30T12:00:00+00:00",
        )

    mock_pub = MagicMock()
    mock_pub.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
        "event_id": "pub_123",
        "remote_event_id": "pub_123",
    }

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = Path(tmpdir) / "decision_history.json"
        with patch("src.evaluation.research_store.resolve_promoted_candidate", return_value=cand):
            with patch("src.evaluation.live_production_decision.authorize_production_runtime", side_effect=mock_auth):
                with patch("src.evaluation.live_runtime.evaluate_authorized_live_runtime", side_effect=spy_eval):
                    result = evaluate_mtf_live_runtime(
                        data_by_timeframe=data_by_tf,
                        symbol="XAUUSD",
                        local_timeframe="5m",
                        stable_strategy="momentum",
                        candidate_id="cand_test",
                        store_path=st_path,
                        publisher=mock_pub,
                        publish=True,
                        persist=True,
                        reference_now=reference_now.to_pydatetime(),
                    )

    # 1. Assert EXACTLY ONE evaluation call per timeframe (Anti-recurrence control)
    tf_call_counts = {tf: 0 for tf in timeframes}
    for tf, c_id in eval_calls:
        tf_call_counts[tf] += 1

    for tf in timeframes:
        assert tf_call_counts[tf] == 1, f"Timeframe {tf} was evaluated {tf_call_counts[tf]} times instead of exactly 1"

    # 2. Assert Complete Identity and Lineage Continuity across MTF evaluation & publication
    published_cld = result.local_result.canonical_decision
    source_5m_signal = next(s for s in result.mtf_intelligence.signals if s.timeframe == CanonicalTimeframe.FIVE_MINUTES)

    # A. Decision Identity
    assert published_cld.decision.decision_id == source_5m_signal.decision_id
    assert published_cld.signal.signal_id == source_5m_signal.signal_id
    assert published_cld.live_decision_id == source_5m_signal.decision_id

    # B. Canonical Live Decision Identity
    matching_tr = next(
        (tr for tr in published_cld.transition_history if tr.artifact_fingerprint == source_5m_signal.canonical_live_decision_fingerprint),
        None,
    )
    assert matching_tr is not None

    # C. Authorization Lineage
    assert published_cld.authorization_receipt.authorization_fingerprint == source_5m_signal.authorization_fingerprint
    assert published_cld.authorization_receipt.candidate_id == source_5m_signal.candidate_id

    # D. Research / Candidate Lineage
    assert published_cld.decision.candidate_id == source_5m_signal.candidate_id
    assert published_cld.decision.evidence_id == source_5m_signal.evidence_id
    assert published_cld.decision.experiment_fingerprint == source_5m_signal.experiment_fingerprint

    # E. Scope Identity
    assert published_cld.decision.symbol == source_5m_signal.symbol
    assert published_cld.decision.timeframe == source_5m_signal.timeframe.value
    assert published_cld.authorization_receipt.strategy_name == source_5m_signal.strategy_name
    assert published_cld.authorization_receipt.strategy_version == source_5m_signal.strategy_version

    # F. Publication Continuity (Exactly ONE publication call)
    assert mock_pub.publish.call_count == 1
    pub_payload = mock_pub.publish.call_args[0][0]
    assert pub_payload.decision_id == source_5m_signal.decision_id
    assert pub_payload.signal_id == source_5m_signal.signal_id
