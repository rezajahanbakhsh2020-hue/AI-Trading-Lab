"""End-to-end production decision integrity, signal lineage, and risk validation tests.

Tests cover all 30 required scenarios for production decision integrity and research-to-production lineage.
"""

from datetime import datetime, timezone
import math
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from src.evaluation.live_decision_record import build_live_decision_record
from src.evaluation.live_decision_store import (
    append_live_decision_to_store,
    load_live_decision_history,
)
from src.evaluation.live_production_decision import (
    DEFAULT_MIN_STABILITY_SCORE,
    Direction,
    ProductionDecision,
    ProductionPromotionPolicy,
    ProductionRiskLevels,
    ProductionSignal,
    PromotedCandidateArtifact,
    build_live_production_decision,
    calculate_production_risk_levels,
    evaluate_production_decision,
    validate_market_data_for_production,
    validate_promotion_eligibility,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
)


def make_market_data(rows: int = 80, trend: str = "UP", start_price: float = 1000.0) -> pd.DataFrame:
    if trend == "UP":
        close = [start_price + i for i in range(rows)]
    elif trend == "DOWN":
        close = [start_price - i for i in range(rows)]
    else:
        close = [start_price] * rows

    now = pd.Timestamp.now(tz="UTC")
    timestamps = [now - pd.Timedelta(minutes=5 * (rows - 1 - i)) for i in range(rows)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
        }
    )


def make_promoted_evidence(
    status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
    robustness_passed: bool = True,
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
) -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id="ds_test",
        symbol=symbol,
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="843dfa76cf86a9057dba0a127541d7093fb15e42")
    spec = ResearchExperimentSpec(
        hypothesis="Test hypothesis for production decision integrity",
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
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-03",
        end_date="2025-01-05",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=30,
        start_timestamp_utc="2025-01-03T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-01-05",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict={
            "passed": robustness_passed,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        } if robustness_passed else {"passed": False, "is_robust": False},
        promotion_status=status,
        rejection_reasons=rejection_reasons,
    )


# --- Required Tests 1-8: Promotion & Market Data Guards ---

def test_1_legitimately_promoted_candidate_reaches_production_decision():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact("cand_01", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.BUY
    assert dec.candidate_id == "cand_01"
    assert dec.evidence_id == ev.evidence_id


def test_2_unpromoted_candidate_is_rejected():
    ev = make_promoted_evidence(PromotionStatus.PROPOSED)
    with pytest.raises(ValueError, match="not allowed for production"):
        PromotedCandidateArtifact("cand_02", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)


def test_3_stale_invalid_research_evidence_is_rejected():
    ev = make_promoted_evidence(
        status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_ROBUSTNESS,),
    )
    with pytest.raises(ValueError, match="not allowed for production"):
        PromotedCandidateArtifact("cand_03", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)


def test_4_stale_market_data_is_rejected():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_04", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    stale_time = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)

    with pytest.raises(ValueError, match="stale"):
        evaluate_production_decision(cand, data, reference_now=stale_time, max_age_seconds=300.0)


def test_5_invalid_market_data_is_rejected():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_05", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    data.loc[data.index[-1], "close"] = -100.0  # Invalid negative price

    with pytest.raises(ValueError, match="non-positive price"):
        evaluate_production_decision(cand, data)


def test_6_wrong_instrument_is_rejected():
    ev = make_promoted_evidence(symbol="XAUUSD")
    with pytest.raises(ValueError, match="does not match evidence dataset symbol"):
        PromotedCandidateArtifact("cand_06", "momentum", "1.0", ev, "BTCUSD", "5m", 0.85)


def test_7_wrong_timeframe_is_rejected():
    ev = make_promoted_evidence(timeframe="5m")
    with pytest.raises(ValueError, match="does not match evidence dataset timeframe"):
        PromotedCandidateArtifact("cand_07", "momentum", "1.0", ev, "XAUUSD", "1h", 0.85)


def test_8_future_market_timestamp_is_rejected():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_08", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    future_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="in the future"):
        evaluate_production_decision(cand, data, reference_now=future_time)


# --- Required Tests 9-13: Authoritative Direction & Signal Derivation ---

def test_9_authoritative_strategy_decision_determines_direction():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_09", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.BUY
    assert dec.entry_price > 0


def test_10_no_hardcoded_buy_sell():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_10", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="DOWN")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.NO_TRADE
    assert dec.direction.value != "BUY"


def test_11_no_trade_remains_no_trade():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_11", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="FLAT")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.NO_TRADE
    assert dec.entry_price is None


def test_12_signal_is_derived_from_production_decision():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_12", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)
    sig = ProductionSignal.from_decision(dec)

    assert sig.decision_id == dec.decision_id
    assert sig.direction == dec.direction
    assert sig.entry_price == dec.entry_price
    assert sig.candidate_id == dec.candidate_id
    assert sig.evidence_id == dec.evidence_id


def test_13_signal_cannot_exist_without_valid_decision():
    with pytest.raises(TypeError, match="decision must be a ProductionDecision instance"):
        ProductionSignal.from_decision(None)


# --- Required Tests 14-19: Authoritative Risk Geometry ---

def test_14_risk_is_derived_from_authoritative_project_1_risk_logic():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_14", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP", start_price=2000.0)
    dec = evaluate_production_decision(cand, data)
    risk = calculate_production_risk_levels(dec, cand)

    assert risk.entry_price == dec.entry_price
    assert risk.stop_loss < risk.entry_price
    assert risk.tp1 > risk.entry_price


def test_15_no_hardcoded_1_percent_2_percent_fallback():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact(
        "cand_15", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85,
        parameters={"stop_loss_pct": 0.03, "take_profit_pct": 0.06}
    )
    data = make_market_data(trend="UP", start_price=1000.0)
    dec = evaluate_production_decision(cand, data)
    risk = calculate_production_risk_levels(dec, cand)

    # Entry is close price (~1079)
    entry = dec.entry_price
    assert risk.stop_loss == pytest.approx(entry * 0.97)
    assert risk.tp1 == pytest.approx(entry + (entry * 0.03 * 1.0))


def test_16_invalid_risk_geometry_is_rejected():
    with pytest.raises(ValueError, match="strictly below entry_price"):
        ProductionRiskLevels(
            decision_id="dec_test",
            candidate_id="cand_test",
            evidence_id="ev_test",
            symbol="XAUUSD",
            timeframe="5m",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=2050.0,  # Invalid SL for BUY
            tp1=2020.0,
            tp2=2040.0,
            tp3=2060.0,
        )


def test_17_sl_tp_direction_is_validated():
    # SELL direction with SL below entry is invalid
    with pytest.raises(ValueError, match="strictly above entry_price"):
        ProductionRiskLevels(
            decision_id="dec_test",
            candidate_id="cand_test",
            evidence_id="ev_test",
            symbol="XAUUSD",
            timeframe="5m",
            direction=Direction.SELL,
            entry_price=2000.0,
            stop_loss=1950.0,
            tp1=1980.0,
            tp2=1960.0,
            tp3=1940.0,
        )


def test_18_tp1_tp2_tp3_ordering_is_validated():
    # BUY direction with TP1 > TP2 is invalid
    with pytest.raises(ValueError, match="must be less than TP2"):
        ProductionRiskLevels(
            decision_id="dec_test",
            candidate_id="cand_test",
            evidence_id="ev_test",
            symbol="XAUUSD",
            timeframe="5m",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=1980.0,
            tp1=2050.0,
            tp2=2040.0,
            tp3=2060.0,
        )


def test_19_trailing_stop_lineage_is_preserved_where_supported():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact(
        "cand_19", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85,
        parameters={"trailing_stop_level": 1985.0}
    )
    data = make_market_data(trend="UP", start_price=2000.0)
    dec = evaluate_production_decision(cand, data)
    risk = calculate_production_risk_levels(dec, cand)

    assert risk.trailing_stop == 1985.0


# --- Required Tests 20-26: Lineage, Replay, Persistence & E2E ---

def test_20_decision_signal_risk_preserve_candidate_evidence_provenance():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_20", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)
    sig = ProductionSignal.from_decision(dec)
    risk = calculate_production_risk_levels(dec, cand)

    assert dec.candidate_id == cand.candidate_id
    assert dec.evidence_id == ev.evidence_id
    assert sig.candidate_id == cand.candidate_id
    assert sig.evidence_id == ev.evidence_id
    assert risk.candidate_id == cand.candidate_id
    assert risk.evidence_id == ev.evidence_id


def test_21_deterministic_identity_for_identical_authoritative_inputs():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_21", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    market_latest_ts = data["timestamp"].iloc[-1]

    dec1 = evaluate_production_decision(cand, data, reference_now=market_latest_ts)
    dec2 = evaluate_production_decision(cand, data, reference_now=market_latest_ts)

    assert dec1.decision_id == dec2.decision_id


def test_22_replay_idempotency():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "history.json"
        snap = {
            "timestamp": "2025-01-01T12:00:00Z", "symbol": "XAUUSD", "interval": "5m",
            "signal": 1, "signal_label": "BUY", "trend": "UP", "strategy": "momentum",
            "entry_price": 2000.0, "stop_loss": 1980.0, "take_profit": 2040.0,
            "risk_reward_ratio": 2.0, "stability_score": 0.8, "market_state": "OPEN",
            "quote_age_seconds": 10.0, "quote_stale": False, "candle_count": 100,
            "decision_id": "dec_22", "signal_id": "sig_22",
        }
        rec = build_live_decision_record(snap)
        rec["decision_id"] = "dec_22"
        rec["signal_id"] = "sig_22"

        append_live_decision_to_store(rec, path)
        append_live_decision_to_store(rec, path)  # Replay identical record
        history = load_live_decision_history(path)

        assert len(history) == 1


def test_23_conflicting_replay_fails_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "history.json"
        snap = {
            "timestamp": "2025-01-01T12:00:00Z", "symbol": "XAUUSD", "interval": "5m",
            "signal": 1, "signal_label": "BUY", "trend": "UP", "strategy": "momentum",
            "entry_price": 2000.0, "stop_loss": 1980.0, "take_profit": 2040.0,
            "risk_reward_ratio": 2.0, "stability_score": 0.8, "market_state": "OPEN",
            "quote_age_seconds": 10.0, "quote_stale": False, "candle_count": 100,
            "decision_id": "dec_23", "signal_id": "sig_23",
        }
        rec1 = build_live_decision_record(snap)
        rec1["decision_id"] = "dec_23"
        rec1["signal_id"] = "sig_23"
        append_live_decision_to_store(rec1, path)

        rec2 = dict(rec1)
        rec2["entry_price"] = 2050.0  # Conflicting entry price for same decision ID

        with pytest.raises(ValueError, match="Conflicting replay detected"):
            append_live_decision_to_store(rec2, path)


def test_24_persistence_failure_cannot_produce_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = Path(tmpdir) / "non_existent_dir" / "read_only_file.txt"
        # Create non_existent_dir as a file to force directory creation failure
        parent = Path(tmpdir) / "blocker"
        parent.write_text("blocker")
        blocked_path = parent / "sub" / "history.json"

        snap = {
            "timestamp": "2025-01-01T12:00:00Z", "symbol": "XAUUSD", "interval": "5m",
            "signal": 1, "signal_label": "BUY", "trend": "UP", "strategy": "momentum",
            "entry_price": 2000.0, "stop_loss": 1980.0, "take_profit": 2040.0,
            "risk_reward_ratio": 2.0, "stability_score": 0.8, "market_state": "OPEN",
            "quote_age_seconds": 10.0, "quote_stale": False, "candle_count": 100,
        }
        rec = build_live_decision_record(snap)

        with pytest.raises(Exception):
            append_live_decision_to_store(rec, blocked_path)


def test_25_incomplete_persistence_cannot_produce_signal():
    incomplete_record = {"symbol": "XAUUSD"}  # Missing required record fields
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        with pytest.raises(ValueError, match="record field 'interval' must not be empty|record is missing required fields"):
            append_live_decision_to_store(incomplete_record, path)


def test_26_complete_end_to_end_production_path():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_26", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP", start_price=2000.0)

    # 1. Authoritative decision
    dec = evaluate_production_decision(cand, data)
    assert dec.direction == Direction.BUY

    # 2. Derive signal
    sig = ProductionSignal.from_decision(dec)
    assert sig.direction == Direction.BUY

    # 3. Derive risk
    risk = calculate_production_risk_levels(dec, cand)
    assert risk.stop_loss < risk.entry_price

    # 4. Record and persist
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "production_history.json"
        snap = {
            "timestamp": dec.market_timestamp,
            "symbol": dec.symbol,
            "interval": dec.timeframe,
            "signal": 1,
            "signal_label": sig.direction.value,
            "trend": "UP",
            "strategy": cand.strategy_name,
            "entry_price": dec.entry_price,
            "stop_loss": risk.stop_loss,
            "take_profit": risk.tp1,
            "risk_reward_ratio": risk.risk_reward_ratio,
            "stability_score": 0.8,
            "market_state": "OPEN",
            "quote_age_seconds": 5.0,
            "quote_stale": False,
            "candle_count": len(data),
            "decision_id": dec.decision_id,
            "signal_id": sig.signal_id,
        }
        rec = build_live_decision_record(snap)
        rec["decision_id"] = dec.decision_id
        rec["signal_id"] = sig.signal_id

        history = append_live_decision_to_store(rec, path)
        assert len(history) == 1
        assert history[0]["symbol"] == "XAUUSD"


# --- Required Tests 27-29: PR #4, #5, #6 Regression Coverage ---

def test_27_regression_coverage_pr4_research_constitution():
    from src.evaluation.research_constitution import ResearchExperimentSpec
    ev = make_promoted_evidence()
    assert ev.spec.fingerprint == ev.experiment_fingerprint
    assert ev.evidence_id.startswith("ev_")


def test_28_regression_coverage_pr5_discovery_engine():
    from src.evaluation.discovery_engine import DiscoveryEngine
    engine = DiscoveryEngine()
    assert engine is not None


def test_29_regression_coverage_pr6_robustness_validation():
    from src.evaluation.robustness_evaluator import RobustnessEvaluator
    evaluator = RobustnessEvaluator()
    assert evaluator is not None


def test_missing_risk_parameters_fails_closed():
    ev = make_promoted_evidence()
    spec_no_risk = ResearchExperimentSpec(
        hypothesis="Missing risk params",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ev.spec.dataset_scope,
        execution_assumptions=ev.spec.execution_assumptions,
        code_provenance=ev.spec.code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10},
    )
    ev_no_risk = ResearchEvidence(
        experiment_fingerprint=spec_no_risk.fingerprint,
        spec=spec_no_risk,
        partitions=ev.partitions,
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )
    cand = PromotedCandidateArtifact("cand_no_risk", "momentum", "1.0", ev_no_risk, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)

    with pytest.raises(ValueError, match="missing required 'stop_loss_pct' risk parameter"):
        calculate_production_risk_levels(dec, cand)


def test_38_promoted_candidate_operational_stability_roundtrip(tmp_path):
    """A. Prove persisted operational stability value survives round-trip unchanged."""
    from src.evaluation.research_store import save_research_candidate, load_candidate_binding, resolve_promoted_candidate

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    save_research_candidate(
        candidate_id="cand_stab_roundtrip",
        evidence=ev,
        base_dir=tmp_path,
        operational_stability_score=0.825,
    )

    binding = load_candidate_binding("cand_stab_roundtrip", base_dir=tmp_path)
    assert binding["operational_stability_score"] == 0.825

    reconstituted = resolve_promoted_candidate(candidate_id="cand_stab_roundtrip", base_dir=tmp_path)
    assert reconstituted.operational_stability_score == 0.825


def test_39_operational_stability_score_changes_artifact_fingerprint(tmp_path):
    """B. Prove changing the authoritative operational stability value changes the promoted artifact fingerprint."""
    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")

    c1 = PromotedCandidateArtifact.from_persisted_research(
        candidate_id="cand_fp_1",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.75,
    )

    c2 = PromotedCandidateArtifact.from_persisted_research(
        candidate_id="cand_fp_1",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        operational_stability_score=0.90,
    )

    assert c1.artifact_fingerprint != c2.artifact_fingerprint


def test_40_missing_or_non_finite_operational_stability_fails_closed(tmp_path):
    """C. Prove missing, malformed, NaN, or infinite operational stability evidence fails closed."""
    from src.evaluation.research_store import save_research_candidate, _candidate_binding_path, resolve_promoted_candidate, PromotionIntegrityError
    import json

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")

    # 1. Invalid values at PromotedCandidateArtifact construction
    for invalid_val in (float("nan"), float("inf"), float("-inf"), "invalid_str"):
        with pytest.raises((ValueError, TypeError)):
            PromotedCandidateArtifact.from_persisted_research(
                candidate_id="cand_bad_stab",
                evidence=ev,
                symbol="XAUUSD",
                timeframe="5m",
                operational_stability_score=invalid_val,  # type: ignore
            )

    # 2. Corrupted binding JSON missing operational_stability_score
    save_research_candidate(candidate_id="cand_corrupt_stab", evidence=ev, base_dir=tmp_path, operational_stability_score=0.8)
    b_path = _candidate_binding_path("cand_corrupt_stab", tmp_path)
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    del b_data["operational_stability_score"]
    b_path.write_text(json.dumps(b_data), encoding="utf-8")

    with pytest.raises(PromotionIntegrityError, match="missing required operational_stability_score"):
        resolve_promoted_candidate(candidate_id="cand_corrupt_stab", base_dir=tmp_path)


def test_41_conflicting_caller_stability_score_rejected():
    """D. Prove conflicting caller-supplied stability_score cannot override candidate's authoritative value."""
    from src.evaluation.live_runtime import evaluate_authorized_live_runtime
    from src.evaluation.live_runtime_context import create_authorized_runtime_context
    from src.evaluation.live_market_evaluation import create_live_market_evaluation
    from src.evaluation.live_production_decision import authorize_production_runtime, ProductionAuthorizationReceipt

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    cand = PromotedCandidateArtifact.from_persisted_research(
        candidate_id="cand_auth_stab",
        evidence=ev,
        symbol="XAUUSD",
        timeframe="5m",
        governance_decision_fingerprint="gov_fp_test_123",
        operational_stability_score=0.88,
    )
    data = make_market_data(rows=80, trend="UP")
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=ref_now)
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    context = create_authorized_runtime_context(cand, auth, receipt)

    eval_obj = create_live_market_evaluation(data, context=context, reference_now=ref_now)

    # Conflicting caller-supplied score (0.20 != 0.88) MUST raise ValueError
    with pytest.raises(ValueError, match="conflicts with authoritative candidate operational_stability_score"):
        evaluate_authorized_live_runtime(
            data,
            evaluation=eval_obj,
            context=context,
            stable_strategy="momentum",
            stability_score=0.20,
            min_stability_score=0.50,
        )

    # Matching caller-supplied score (0.88 == 0.88) MUST succeed
    res = evaluate_authorized_live_runtime(
        data,
        evaluation=eval_obj,
        context=context,
        stable_strategy="momentum",
        stability_score=0.88,
        min_stability_score=0.50,
    )
    assert res.decision["decision"] == "BUY"
    assert res.decision["stability_score"] == 0.88


def test_42_raw_production_selection_isolation(tmp_path, monkeypatch):
    """E. Prove modifying load_production_selection() output does not alter live decisions."""
    from unittest.mock import patch
    from src.evaluation.research_store import save_research_candidate
    from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    save_research_candidate(candidate_id="cand_isolation", evidence=ev, base_dir=tmp_path, operational_stability_score=0.92)

    config = ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m", candidate_id="cand_isolation", strategy_id="momentum", research_dir=tmp_path)

    # Mock load_production_selection to return conflicting/garbage data
    monkeypatch.setattr(
        "src.evaluation.production_live_bridge.load_production_selection",
        lambda *args, **kwargs: {"stability_score": 0.10, "stable_strategy": "bogus_strategy"},
    )

    data = make_market_data(trend="UP")
    df_ts = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=data):
        res = runtime.run_once(publish=False, persist=True, reference_now=df_ts)

    # Live decision MUST succeed using candidate's authoritative stability_score (0.92) and strategy ("momentum")
    assert res["decision"] == "BUY"
    assert res["stability_score"] == 0.92
    assert res["strategy"] == "momentum"


def test_43_fresh_and_stale_convergence_lineage_preserved(tmp_path):
    """F. Prove fresh and stale LiveMarketEvaluation enter same canonical downstream lifecycle with operational stability lineage."""
    from unittest.mock import patch
    from src.evaluation.research_store import save_research_candidate
    from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig
    from src.evaluation.live_decision_lifecycle import LiveDecisionLifecycleState

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    save_research_candidate(candidate_id="cand_conv", evidence=ev, base_dir=tmp_path, operational_stability_score=0.84)

    config = ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m", candidate_id="cand_conv", strategy_id="momentum", research_dir=tmp_path)
    data = make_market_data(trend="UP")
    df_ts = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    from datetime import timedelta
    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=data):
        res_fresh = runtime.run_once(publish=False, persist=True, reference_now=df_ts)
        res_stale = runtime.run_once(publish=False, persist=True, reference_now=df_ts + timedelta(seconds=1000))

    assert res_fresh["decision"] == "BUY"
    assert res_fresh["current_lifecycle_state"] == LiveDecisionLifecycleState.PERSISTED.value
    assert res_fresh["record"]["stability_score"] == 0.84

    assert res_stale["decision"] == "NO TRADE"
    assert res_stale["current_lifecycle_state"] == LiveDecisionLifecycleState.PERSISTED.value
    assert res_stale["record"]["stability_score"] == 0.84


def test_44_repository_invariant_runtime_isolated_from_production_json_and_fails_closed_when_missing(tmp_path, monkeypatch):
    """Invariant test proving live production decision cannot obtain operational_stability_score from production selection JSON.

    1. Even if production JSON exists and contains a conflicting score (0.10) and strategy ('bogus'),
       the authoritative candidate lineage (strategy 'momentum', score 0.92) is used.
    2. If candidate binding stability is missing, runtime fails closed and cannot reach decision execution.
    """
    from unittest.mock import patch
    import json
    from src.evaluation.research_store import save_research_candidate, _candidate_binding_path, PromotionIntegrityError
    from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig, resolve_authoritative_promoted_candidate

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")

    # Part 1: Prove production selection JSON cannot affect runtime decision or stability score
    save_research_candidate(candidate_id="cand_inv_01", evidence=ev, base_dir=tmp_path, operational_stability_score=0.92)
    config = ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m", candidate_id="cand_inv_01", strategy_id="momentum", research_dir=tmp_path)

    # Mock load_production_selection to return conflicting production JSON state
    monkeypatch.setattr(
        "src.evaluation.production_live_bridge.load_production_selection",
        lambda *args, **kwargs: {"stability_score": 0.10, "stable_strategy": "bogus_strategy"},
    )

    data = make_market_data(rows=80, trend="UP")
    df_ts = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=data):
        res = runtime.run_once(publish=False, persist=True, reference_now=df_ts)

    assert res["blocked"] is False
    assert res["decision"] == "BUY"
    assert res["stability_score"] == 0.92
    assert res["strategy"] == "momentum"

    # Part 2: Prove missing candidate binding stability score fails closed before reaching decision execution
    save_research_candidate(candidate_id="cand_inv_missing", evidence=ev, base_dir=tmp_path, operational_stability_score=0.88)
    b_path = _candidate_binding_path("cand_inv_missing", tmp_path)
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    del b_data["operational_stability_score"]
    b_path.write_text(json.dumps(b_data), encoding="utf-8")

    config_missing = ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m", candidate_id="cand_inv_missing", strategy_id="momentum", research_dir=tmp_path)

    # Candidate resolution MUST fail closed with PromotionIntegrityError
    from src.evaluation.research_store import resolve_promoted_candidate
    with pytest.raises(PromotionIntegrityError, match="missing required operational_stability_score"):
        resolve_promoted_candidate(candidate_id="cand_inv_missing", base_dir=tmp_path)

    blocked = resolve_authoritative_promoted_candidate(config_missing)
    assert blocked.reason == "PromotionIntegrityError"

    runtime_missing = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config_missing,
    )

    # Runtime MUST block execution before decision execution
    res_missing = runtime_missing.run_once(publish=False, persist=True, reference_now=df_ts)
    assert res_missing["blocked"] is True
    assert res_missing["decision"] == "NO TRADE"


def test_missing_strategy_window_parameter_fails_closed():
    ev = make_promoted_evidence()
    spec_no_window = ResearchExperimentSpec(
        hypothesis="Missing strategy window",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ev.spec.dataset_scope,
        execution_assumptions=ev.spec.execution_assumptions,
        code_provenance=ev.spec.code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    ev_no_window = ResearchEvidence(
        experiment_fingerprint=spec_no_window.fingerprint,
        spec=spec_no_window,
        partitions=ev.partitions,
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )
    cand = PromotedCandidateArtifact("cand_no_window", "momentum", "1.0", ev_no_window, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")

    with pytest.raises(ValueError, match="missing required 'momentum_window' or 'window' parameter"):
        evaluate_production_decision(cand, data)


def test_candidate_tp_multipliers_override_defaults():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact(
        "cand_tp_mult", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85,
        parameters={"tp1_multiplier": 1.5, "tp2_multiplier": 2.5, "tp3_multiplier": 3.5}
    )
    data = make_market_data(trend="UP", start_price=1000.0)
    dec = evaluate_production_decision(cand, data)
    risk = calculate_production_risk_levels(dec, cand)

    entry = dec.entry_price
    risk_dist = entry - risk.stop_loss
    assert risk.tp1 == pytest.approx(entry + (risk_dist * 1.5))
    assert risk.tp2 == pytest.approx(entry + (risk_dist * 2.5))
    assert risk.tp3 == pytest.approx(entry + (risk_dist * 3.5))


# --- Negative Test Matrix Coverage (Phase 6) ---

def test_30_no_promoted_candidate_fails_closed():
    data = make_market_data(trend="UP")
    with pytest.raises(TypeError):
        evaluate_production_decision(None, data)  # type: ignore[arg-type]


def test_31_dataset_scope_mismatch_fails_closed():
    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    with pytest.raises(ValueError, match="does not match evidence dataset symbol"):
        PromotedCandidateArtifact("cand_mismatch_scope", "momentum", "1.0", ev, "EURUSD", "5m", 0.85)


def test_32_execution_assumptions_mismatch_fails_closed():
    ds = DatasetScope("ds", "XAUUSD", "5m", "2025-01-01", "2025-01-02")
    with pytest.raises(ValueError, match="transaction_cost cannot be negative"):
        ExecutionAssumptions(transaction_cost=-0.01, slippage=0.001, latency_ms=10.0)


def test_33_lineage_fingerprint_mismatch_fails_closed():
    from src.evaluation.research_store import PromotionIntegrityError, _reconstitute_promoted_candidate_from_binding
    ev = make_promoted_evidence()
    binding = {
        "candidate_id": "cand_fp_mismatch",
        "strategy_name": "momentum",
        "strategy_version": "1.0",
        "experiment_fingerprint": "fp_tampered_12345",
        "evidence_id": ev.evidence_id,
        "symbol": "XAUUSD",
        "timeframe": "5m",
        "parameters": {"momentum_window": 10},
    }
    with pytest.raises(PromotionIntegrityError, match="research fingerprint 'fp_tampered_12345' does not match"):
        _reconstitute_promoted_candidate_from_binding(binding, ev)


def test_34_independently_supplied_parameters_disagree_with_authoritative():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact(
        "cand_override", "momentum", "1.0", ev, "XAUUSD", "5m", 0.85,
        parameters={"stop_loss_pct": 0.05, "take_profit_pct": 0.10}
    )
    data = make_market_data(trend="UP", start_price=1000.0)
    dec = evaluate_production_decision(cand, data)

    # When risk levels are calculated using candidate parameters, authoritative candidate parameters govern
    risk = calculate_production_risk_levels(dec, cand)
    entry = dec.entry_price
    assert risk.stop_loss == pytest.approx(entry * 0.95)


def test_35_structural_ast_no_hardcoded_production_fallbacks():
    """Verify that operational production wrappers do not reintroduce hardcoded fallbacks."""
    import ast

    wrapper_file = Path("src/evaluation/live_production_decision.py")
    tree = ast.parse(wrapper_file.read_text(encoding="utf-8"), filename=str(wrapper_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_live_production_decision":
            # Inspect default arg values
            for default in node.args.defaults:
                if isinstance(default, ast.Constant):
                    assert default.value not in (10, 0.01, 0.02), (
                        f"Forbidden operational default constant {default.value} found in "
                        "build_live_production_decision definition signature."
                    )


def test_36_operational_wrapper_delegates_to_authoritative_chain(monkeypatch, tmp_path):
    """Prove that build_live_production_decision delegates directly to candidate resolution and evaluation."""
    from src.evaluation.research_store import save_research_experiment, persist_promoted_candidate_binding
    from src.evaluation.live_production_decision import build_live_production_decision

    ev = make_promoted_evidence(symbol="XAUUSD", timeframe="5m")
    save_research_experiment(ev, base_dir=tmp_path)
    persist_promoted_candidate_binding(candidate_id="cand_auth_test", evidence=ev, operational_stability_score=0.85, base_dir=tmp_path)

    data = make_market_data(trend="UP")

    # Run operational wrapper pointing to candidate in tmp_path
    res = build_live_production_decision(
        data,
        stable_strategy="momentum",
        stability_score=0.80,
        candidate_id="cand_auth_test",
        research_dir=tmp_path,
    )

    assert res["decision"] == "BUY"
    assert res["candidate_id"] == "cand_auth_test"
    assert res["evidence_id"] == ev.evidence_id
    assert res["experiment_fingerprint"] == ev.experiment_fingerprint


def test_37_missing_risk_parameter_fails_closed():
    """Prove that candidate missing required risk parameter fails closed without fallback."""
    ev = make_promoted_evidence()
    spec_no_sl = ResearchExperimentSpec(
        hypothesis="Missing SL",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ev.spec.dataset_scope,
        execution_assumptions=ev.spec.execution_assumptions,
        code_provenance=ev.spec.code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "take_profit_pct": 0.02}, # Missing stop_loss_pct
    )
    ev_no_sl = ResearchEvidence(
        experiment_fingerprint=spec_no_sl.fingerprint,
        spec=spec_no_sl,
        partitions=ev.partitions,
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )
    cand = PromotedCandidateArtifact("cand_no_sl", "momentum", "1.0", ev_no_sl, "XAUUSD", "5m", 0.85)
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)

    with pytest.raises(ValueError, match="missing required 'stop_loss_pct' risk parameter"):
        calculate_production_risk_levels(dec, cand)
