"""Authoritative promoted-candidate resolution: persist research, resolve by identity, fail closed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
    resolve_authoritative_promoted_candidate,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionPromotionPolicy,
    calculate_production_risk_levels,
    evaluate_production_decision,
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
from src.evaluation.research_store import (
    PromotionEligibilityError,
    PromotionIntegrityError,
    persist_promoted_candidate_binding,
    resolve_promoted_candidate,
    save_research_candidate,
    save_research_experiment,
)


def make_research_evidence(
    *,
    status: PromotionStatus = PromotionStatus.PROMOTABLE,
    strategy_name: str = "momentum",
    strategy_version: str = "1.0",
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    hypothesis: str = "Authoritative promoted momentum candidate",
    robustness_passed: bool = True,
    rejection_reasons: tuple[RejectionReason, ...] = (),
    created_at_utc: str = "2025-01-01T00:00:00+00:00",
    parameters: dict | None = None,
) -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id=f"ds_{symbol.lower()}_{timeframe}",
        symbol=symbol,
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-02",
    )
    spec = ResearchExperimentSpec(
        hypothesis=hypothesis,
        methodology_version="1.0",
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001, slippage=0.001, latency_ms=10.0
        ),
        code_provenance=CodeProvenance(commit_sha="843dfa76cf86a9057dba0a127541d7093fb15e42"),
        benchmark_reference="buy_and_hold",
        parameters=parameters
        or {"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
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
        created_at_utc=created_at_utc,
    )


def persist_test_candidate(
    base_dir: Path,
    *,
    candidate_id: str = "cand_momentum_auth",
    **evidence_kwargs,
) -> tuple[str, ResearchEvidence]:
    evidence = make_research_evidence(**evidence_kwargs)
    save_research_candidate(candidate_id=candidate_id, evidence=evidence, base_dir=base_dir)
    return candidate_id, evidence


def make_buy_market_data() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 10:00", periods=100, freq="5min", tz="UTC")
    prices = [2000.0 + (i * 2.0) for i in range(100)]
    return pd.DataFrame(
        {
            "openTime": timestamps,
            "open": [p - 1.0 for p in prices],
            "high": [p + 3.0 for p in prices],
            "low": [p - 2.0 for p in prices],
            "close": prices,
            "timestamp": timestamps,
        }
    )


def test_valid_persisted_promoted_candidate_allows_production(tmp_path: Path) -> None:
    candidate_id, evidence = persist_test_candidate(tmp_path)
    resolved = resolve_promoted_candidate(
        candidate_id=candidate_id,
        strategy_id="momentum",
        symbol="XAUUSD",
        timeframe="5m",
        base_dir=tmp_path,
    )
    assert resolved is not None
    assert resolved.candidate_id == candidate_id
    assert resolved.evidence.evidence_id == evidence.evidence_id
    assert resolved.evidence.experiment_fingerprint == evidence.experiment_fingerprint
    assert resolved.evidence.promotion_status == PromotionStatus.PROMOTABLE
    data = make_buy_market_data()
    ref_now = data["timestamp"].iloc[-1].to_pydatetime()
    decision = evaluate_production_decision(resolved, data, reference_now=ref_now)
    assert decision.direction == Direction.BUY
    assert decision.candidate_id == candidate_id
    assert decision.evidence_id == evidence.evidence_id


def test_candidate_missing_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_present")
    resolved = resolve_promoted_candidate(candidate_id="cand_missing", base_dir=tmp_path)
    assert resolved is None
    blocked = resolve_authoritative_promoted_candidate(
        ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_missing",
            research_dir=tmp_path,
        )
    )
    assert blocked.reason == "PromotionUnavailable"


def test_evidence_missing_is_blocked(tmp_path: Path) -> None:
    evidence = make_research_evidence()
    persist_promoted_candidate_binding(
        candidate_id="cand_orphan",
        evidence=evidence,
        base_dir=tmp_path,
    )
    with pytest.raises(PromotionIntegrityError, match="Evidence missing"):
        resolve_promoted_candidate(candidate_id="cand_orphan", base_dir=tmp_path)


def test_evidence_not_promoted_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(
        tmp_path,
        candidate_id="cand_proposed",
        status=PromotionStatus.PROPOSED,
        hypothesis="Unpromoted proposed candidate",
    )
    with pytest.raises(PromotionEligibilityError, match="not allowed for production"):
        resolve_promoted_candidate(candidate_id="cand_proposed", base_dir=tmp_path)


def test_fingerprint_mismatch_is_blocked(tmp_path: Path) -> None:
    real_id, real_evidence = persist_test_candidate(tmp_path, candidate_id="cand_real")
    decoy = make_research_evidence(hypothesis="Decoy experiment for fingerprint mismatch")
    fake_dir = tmp_path / decoy.experiment_fingerprint
    fake_dir.mkdir(parents=True)
    real_json = (tmp_path / real_evidence.experiment_fingerprint / "evidence.json").read_text(
        encoding="utf-8"
    )
    (fake_dir / "evidence.json").write_text(real_json, encoding="utf-8")
    persist_promoted_candidate_binding(
        candidate_id="cand_mismatch_fp",
        evidence=decoy,
        base_dir=tmp_path,
    )
    with pytest.raises(PromotionIntegrityError, match="research fingerprint"):
        resolve_promoted_candidate(candidate_id="cand_mismatch_fp", base_dir=tmp_path)
    assert real_id == "cand_real"


def test_candidate_id_mismatch_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_alpha")
    binding_path = tmp_path / "by_candidate" / "cand_alpha" / "candidate.json"
    payload = binding_path.read_text(encoding="utf-8").replace("cand_alpha", "cand_beta", 1)
    other_dir = tmp_path / "by_candidate" / "cand_spoofed"
    other_dir.mkdir(parents=True)
    (other_dir / "candidate.json").write_text(payload, encoding="utf-8")
    with pytest.raises(PromotionIntegrityError, match="does not match requested candidate_id"):
        resolve_promoted_candidate(candidate_id="cand_spoofed", base_dir=tmp_path)


def test_strategy_version_mismatch_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_v1", strategy_version="1.0")
    with pytest.raises(PromotionIntegrityError, match="strategy_version"):
        resolve_promoted_candidate(
            candidate_id="cand_v1",
            strategy_version="2.0",
            base_dir=tmp_path,
        )


def test_symbol_mismatch_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_xau")
    with pytest.raises(PromotionEligibilityError, match="symbol"):
        resolve_promoted_candidate(
            candidate_id="cand_xau",
            symbol="EURUSD",
            base_dir=tmp_path,
        )


def test_timeframe_mismatch_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_5m")
    with pytest.raises(PromotionEligibilityError, match="timeframe"):
        resolve_promoted_candidate(
            candidate_id="cand_5m",
            timeframe="1h",
            base_dir=tmp_path,
        )


def test_stale_invalid_evidence_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(
        tmp_path,
        candidate_id="cand_stale",
        created_at_utc="2020-01-01T00:00:00+00:00",
    )
    policy = ProductionPromotionPolicy(max_evidence_age_days=30.0)
    with pytest.raises(PromotionEligibilityError, match="stale"):
        resolve_promoted_candidate(
            candidate_id="cand_stale",
            base_dir=tmp_path,
            policy=policy,
        )
    persist_test_candidate(
        tmp_path,
        candidate_id="cand_rejected",
        status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_ROBUSTNESS,),
        hypothesis="Rejected evidence must not enter production",
        robustness_passed=False,
    )
    with pytest.raises(PromotionEligibilityError, match="not allowed for production"):
        resolve_promoted_candidate(candidate_id="cand_rejected", base_dir=tmp_path)


def test_configuration_selecting_unpromoted_candidate_is_blocked(tmp_path: Path) -> None:
    persist_test_candidate(
        tmp_path,
        candidate_id="cand_experimental",
        status=PromotionStatus.EXPERIMENTAL,
        hypothesis="Experimental candidate is not production eligible",
    )
    blocked = resolve_authoritative_promoted_candidate(
        ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_experimental",
            strategy_id="momentum",
            research_dir=tmp_path,
        )
    )
    assert blocked.reason == "PromotionEligibilityError"


def test_no_fallback_candidate(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_other")
    blocked = resolve_authoritative_promoted_candidate(
        ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="cand_requested_missing",
            research_dir=tmp_path,
        )
    )
    assert blocked.reason == "PromotionUnavailable"
    resolved = resolve_promoted_candidate(candidate_id="cand_requested_missing", base_dir=tmp_path)
    assert resolved is None


def test_no_fallback_strategy(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path, candidate_id="cand_momentum_only")
    blocked = resolve_authoritative_promoted_candidate(
        ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            strategy_id="moving_average",
            research_dir=tmp_path,
        )
    )
    assert blocked.reason == "PromotionUnavailable"
    assert resolve_promoted_candidate(strategy_id="moving_average", base_dir=tmp_path) is None


def test_restart_resolves_identical_authoritative_candidate(tmp_path: Path) -> None:
    candidate_id, evidence = persist_test_candidate(tmp_path, candidate_id="cand_restart")
    first = resolve_promoted_candidate(candidate_id=candidate_id, base_dir=tmp_path)
    second = resolve_promoted_candidate(candidate_id=candidate_id, base_dir=tmp_path)
    assert first is not None and second is not None
    assert first.candidate_id == second.candidate_id == candidate_id
    assert first.evidence.evidence_id == second.evidence.evidence_id == evidence.evidence_id
    assert first.evidence.experiment_fingerprint == second.evidence.experiment_fingerprint
    assert first.strategy_name == second.strategy_name
    assert first.artifact_fingerprint == second.artifact_fingerprint


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_production_lineage_preserved_through_decision_signal_risk_publication(
    mock_load_data,
    tmp_path: Path,
) -> None:
    candidate_id, evidence = persist_test_candidate(tmp_path, candidate_id="cand_lineage")
    mock_load_data.return_value = make_buy_market_data()
    ref_now = mock_load_data.return_value["timestamp"].iloc[-1].to_pydatetime()
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=MagicMock(publish=MagicMock(return_value={"status": "PUBLISHED", "published": True})),
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id=candidate_id,
            strategy_id="momentum",
            strategy_version="1.0",
            research_dir=tmp_path,
        ),
    )
    result = runtime.run_once(publish=True, persist=True, reference_now=ref_now)
    assert result["blocked"] is False
    assert result["candidate_id"] == candidate_id
    assert result["evidence_id"] == evidence.evidence_id
    assert result["research_fingerprint"] == evidence.experiment_fingerprint
    publication = result["publication"]
    assert publication["candidate_id"] == candidate_id
    assert publication["research_evidence_id"] == evidence.evidence_id
    assert publication["research_fingerprint"] == evidence.experiment_fingerprint
    assert publication["strategy_id"] == "momentum"
    assert result["contract_payload"]["signal"]["candidate_id"] == candidate_id
    resolved = resolve_promoted_candidate(candidate_id=candidate_id, base_dir=tmp_path)
    decision = evaluate_production_decision(resolved, mock_load_data.return_value, reference_now=ref_now)
    assert decision.candidate_id == candidate_id
    assert decision.evidence_id == evidence.evidence_id
    from src.evaluation.live_production_decision import ProductionSignal

    sig = ProductionSignal.from_decision(decision)
    risk = calculate_production_risk_levels(decision, resolved)
    assert sig.candidate_id == candidate_id
    assert sig.evidence_id == evidence.evidence_id
    assert risk.candidate_id == candidate_id
    assert risk.evidence_id == evidence.evidence_id
    assert result["publication"]["decision_id"] == decision.decision_id
    assert result["publication"]["signal_id"] == sig.signal_id


def test_runtime_does_not_manufacture_promoted_candidate(tmp_path: Path) -> None:
    blocked = resolve_authoritative_promoted_candidate(
        ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            strategy_id="momentum",
            research_dir=tmp_path,
        )
    )
    assert blocked.reason == "PromotionUnavailable"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        research_dir=tmp_path,
        production_config=ProductionRuntimeConfig(
            symbol="XAUUSD",
            timeframe="5m",
            strategy_id="momentum",
            research_dir=tmp_path,
        ),
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
    )
    result = runtime.run_once(publish=False, persist=False)
    assert result["blocked"] is True
    assert result["publication"] is None
    assert result["contract_payload"] is None


def test_missing_production_candidate_configuration_blocks(tmp_path: Path) -> None:
    persist_test_candidate(tmp_path)
    blocked = resolve_authoritative_promoted_candidate(
        ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m", research_dir=tmp_path)
    )
    assert blocked.reason == "PromotionUnavailable"
    assert "missing candidate_id and strategy_id" in blocked.detail


def test_validate_promotion_eligibility_rejects_unpromoted_evidence() -> None:
    evidence = make_research_evidence(status=PromotionStatus.PROPOSED)
    with pytest.raises(ValueError, match="not allowed for production"):
        validate_promotion_eligibility(evidence)


def test_stale_evidence_freshness_policy() -> None:
    evidence = make_research_evidence(created_at_utc="2020-01-01T00:00:00+00:00")
    policy = ProductionPromotionPolicy(max_evidence_age_days=1.0)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="stale"):
        validate_promotion_eligibility(evidence, policy=policy, now=now)
    fresh = make_research_evidence(
        created_at_utc=(now - timedelta(hours=1)).isoformat(),
        hypothesis="Fresh evidence remains eligible",
    )
    assert validate_promotion_eligibility(fresh, policy=policy, now=now) is True


def test_save_research_experiment_still_round_trips(tmp_path: Path) -> None:
    evidence = make_research_evidence()
    path = save_research_experiment(evidence, base_dir=tmp_path)
    assert path.exists()
    assert resolve_promoted_candidate(candidate_id="no_binding", base_dir=tmp_path) is None
