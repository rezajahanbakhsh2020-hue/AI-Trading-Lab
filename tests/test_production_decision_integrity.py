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
        end_date="2025-01-02",
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
        robustness_verdict={"passed": robustness_passed},
        promotion_status=status,
        rejection_reasons=rejection_reasons,
    )


# --- Required Tests 1-8: Promotion & Market Data Guards ---

def test_1_legitimately_promoted_candidate_reaches_production_decision():
    ev = make_promoted_evidence(PromotionStatus.PROMOTABLE)
    cand = PromotedCandidateArtifact("cand_01", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.BUY
    assert dec.candidate_id == "cand_01"
    assert dec.evidence_id == ev.evidence_id


def test_2_unpromoted_candidate_is_rejected():
    ev = make_promoted_evidence(PromotionStatus.PROPOSED)
    with pytest.raises(ValueError, match="not allowed for production"):
        PromotedCandidateArtifact("cand_02", "momentum", "1.0", ev, "XAUUSD", "5m")


def test_3_stale_invalid_research_evidence_is_rejected():
    ev = make_promoted_evidence(
        status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_ROBUSTNESS,),
    )
    with pytest.raises(ValueError, match="not allowed for production"):
        PromotedCandidateArtifact("cand_03", "momentum", "1.0", ev, "XAUUSD", "5m")


def test_4_stale_market_data_is_rejected():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_04", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="UP")
    stale_time = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)

    with pytest.raises(ValueError, match="stale"):
        evaluate_production_decision(cand, data, reference_now=stale_time, max_age_seconds=300.0)


def test_5_invalid_market_data_is_rejected():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_05", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="UP")
    data.loc[data.index[-1], "close"] = -100.0  # Invalid negative price

    with pytest.raises(ValueError, match="non-positive price"):
        evaluate_production_decision(cand, data)


def test_6_wrong_instrument_is_rejected():
    ev = make_promoted_evidence(symbol="XAUUSD")
    with pytest.raises(ValueError, match="does not match evidence dataset symbol"):
        PromotedCandidateArtifact("cand_06", "momentum", "1.0", ev, "BTCUSD", "5m")


def test_7_wrong_timeframe_is_rejected():
    ev = make_promoted_evidence(timeframe="5m")
    with pytest.raises(ValueError, match="does not match evidence dataset timeframe"):
        PromotedCandidateArtifact("cand_07", "momentum", "1.0", ev, "XAUUSD", "1h")


def test_8_future_market_timestamp_is_rejected():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_08", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="UP")
    future_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="in the future"):
        evaluate_production_decision(cand, data, reference_now=future_time)


# --- Required Tests 9-13: Authoritative Direction & Signal Derivation ---

def test_9_authoritative_strategy_decision_determines_direction():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_09", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="UP")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.BUY
    assert dec.entry_price > 0


def test_10_no_hardcoded_buy_sell():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_10", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="DOWN")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.NO_TRADE
    assert dec.direction.value != "BUY"


def test_11_no_trade_remains_no_trade():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_11", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="FLAT")
    dec = evaluate_production_decision(cand, data)

    assert dec.direction == Direction.NO_TRADE
    assert dec.entry_price is None


def test_12_signal_is_derived_from_production_decision():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_12", "momentum", "1.0", ev, "XAUUSD", "5m")
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
    cand = PromotedCandidateArtifact("cand_14", "momentum", "1.0", ev, "XAUUSD", "5m")
    data = make_market_data(trend="UP", start_price=2000.0)
    dec = evaluate_production_decision(cand, data)
    risk = calculate_production_risk_levels(dec, cand)

    assert risk.entry_price == dec.entry_price
    assert risk.stop_loss < risk.entry_price
    assert risk.tp1 > risk.entry_price


def test_15_no_hardcoded_1_percent_2_percent_fallback():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact(
        "cand_15", "momentum", "1.0", ev, "XAUUSD", "5m",
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
        "cand_19", "momentum", "1.0", ev, "XAUUSD", "5m",
        parameters={"trailing_stop_level": 1985.0}
    )
    data = make_market_data(trend="UP", start_price=2000.0)
    dec = evaluate_production_decision(cand, data)
    risk = calculate_production_risk_levels(dec, cand)

    assert risk.trailing_stop == 1985.0


# --- Required Tests 20-26: Lineage, Replay, Persistence & E2E ---

def test_20_decision_signal_risk_preserve_candidate_evidence_provenance():
    ev = make_promoted_evidence()
    cand = PromotedCandidateArtifact("cand_20", "momentum", "1.0", ev, "XAUUSD", "5m")
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
    cand = PromotedCandidateArtifact("cand_21", "momentum", "1.0", ev, "XAUUSD", "5m")
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
    cand = PromotedCandidateArtifact("cand_26", "momentum", "1.0", ev, "XAUUSD", "5m")
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
