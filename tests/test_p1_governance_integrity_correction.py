"""Regression tests for P1 Governance Integrity Correction.

Proves fail-closed boundaries for:
1. Missing exact UTC partition timestamps
2. Overlapping or reversed in-sample / out-of-sample partitions
3. Partially evaluated canonical robustness (unevaluated dimensions)
4. Mismatched or unverifiable governance decision fingerprint
5. Positive-control: genuinely qualified, correctly scoped candidate resolves & authorizes
6. Independent timeframe isolation: blocked timeframes do not borrow candidates across timeframes
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from src.evaluation.live_execution_runtime import verify_timeframe_production_readiness
from src.evaluation.live_production_decision import (
    PromotedCandidateArtifact,
    authorize_production_runtime,
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
from src.evaluation.research_qualification import (
    ResearchQualificationResult,
    qualify_research_evidence,
)
from src.evaluation.research_robustness import (
    ResearchRobustnessAssessment,
    RobustnessStatus,
    assess_research_robustness,
)
from src.evaluation.research_store import (
    PromotionEligibilityError,
    resolve_promoted_candidate,
    save_research_candidate,
)


def make_test_evidence(
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    start_utc_is: str | None = "2025-01-01T00:00:00+00:00",
    end_utc_is: str | None = "2025-01-02T00:00:00+00:00",
    start_utc_oos: str | None = "2025-01-03T00:00:00+00:00",
    end_utc_oos: str | None = "2025-01-05T00:00:00+00:00",
    start_date_is: str = "2025-01-01",
    end_date_is: str = "2025-01-02",
    start_date_oos: str = "2025-01-03",
    end_date_oos: str = "2025-01-05",
    robustness_passed: bool = True,
    unevaluated_dimensions: tuple[str, ...] = (),
) -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id=f"ds_{symbol.lower()}_{timeframe}",
        symbol=symbol,
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="843dfa76cf86a9057dba0a127541d7093fb15e42")
    spec = ResearchExperimentSpec(
        hypothesis="Governance test hypothesis",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )

    partitions = [
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date=start_date_is,
            end_date=end_date_is,
            total_return=0.20,
            max_drawdown=0.05,
            sharpe_ratio=2.0,
            observations=50,
            start_timestamp_utc=start_utc_is,
            end_timestamp_utc=end_utc_is,
        ),
        EvidencePartition(
            role=EvidencePartitionRole.OUT_OF_SAMPLE,
            start_date=start_date_oos,
            end_date=end_date_oos,
            total_return=0.15,
            max_drawdown=0.05,
            sharpe_ratio=1.8,
            observations=30,
            start_timestamp_utc=start_utc_oos,
            end_timestamp_utc=end_utc_oos,
        ),
        EvidencePartition(
            role=EvidencePartitionRole.WALK_FORWARD,
            start_date="2025-01-01",
            end_date="2025-01-05",
            total_return=0.10,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            observations=30,
            start_timestamp_utc="2025-01-01T00:00:00+00:00",
            end_timestamp_utc="2025-01-05T00:00:00+00:00",
        ),
    ]

    rob_verdict = {
        "passed": robustness_passed,
        "is_robust": robustness_passed,
        "parameter_sensitivity": {"passed": True},
        "subsample_stability": {"passed": True},
        "execution_cost_stress": {"passed": True},
        "statistical_validation": {"passed": True},
        "anti_overfitting": {"passed": True},
    }
    for dim in unevaluated_dimensions:
        if dim in rob_verdict:
            del rob_verdict[dim]

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=tuple(partitions),
        robustness_verdict=rob_verdict,
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        created_at_utc="2025-01-01T00:00:00+00:00",
    )


def test_missing_exact_utc_partition_timestamps_fails_closed(tmp_path: Path) -> None:
    evidence = make_test_evidence(start_utc_is=None, end_utc_is=None)

    with pytest.raises(ValueError, match="missing required exact UTC timestamps"):
        validate_promotion_eligibility(evidence)

    # Cannot save as candidate
    with pytest.raises(PromotionEligibilityError, match="missing required exact UTC timestamps"):
        save_research_candidate(
            candidate_id="cand_no_utc",
            evidence=evidence,
            operational_stability_score=0.8,
            base_dir=tmp_path,
        )

    # Readiness fails closed
    res = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="cand_no_utc",
        research_dir=tmp_path,
    )
    assert res["status"] == "BLOCKED"


def test_overlapping_or_reversed_partitions_fails_closed(tmp_path: Path) -> None:
    # Overlapping dates: IS end (2025-01-04) > OOS start (2025-01-03)
    evidence = make_test_evidence(
        start_date_is="2025-01-01",
        end_date_is="2025-01-04",
        start_date_oos="2025-01-03",
        end_date_oos="2025-01-05",
    )

    with pytest.raises(ValueError, match="date/timestamp mismatch"):
        validate_promotion_eligibility(evidence)


def test_partially_evaluated_robustness_fails_closed(tmp_path: Path) -> None:
    evidence = make_test_evidence(unevaluated_dimensions=("parameter_sensitivity",))
    rob = assess_research_robustness(evidence)
    qual = qualify_research_evidence(evidence, robustness_assessment=rob)

    assert qual.qualified is False
    assert RejectionReason.FAILED_ROBUSTNESS in qual.rejection_reasons

    with pytest.raises(ValueError, match="unevaluated dimensions"):
        validate_promotion_eligibility(evidence)


def test_mismatched_governance_decision_fingerprint_fails_closed() -> None:
    evidence = make_test_evidence()
    rob = assess_research_robustness(evidence)
    qual = qualify_research_evidence(evidence, robustness_assessment=rob)
    assert qual.qualified is True

    # Bad governance decision fingerprint passed with mismatching governance_decision
    with pytest.raises(ValueError, match="does not match"):
        validate_promotion_eligibility(
            evidence,
            governance_decision=qual,
            governance_decision_fingerprint="bad_fingerprint_12345",
        )

    # Bad decision object fingerprint
    fake_qual = ResearchQualificationResult(
        qualified=True,
        status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        evidence_fingerprint=evidence.experiment_fingerprint,
        experiment_fingerprint=evidence.experiment_fingerprint,
        policy_version="tampered_policy_version",
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_promotion_eligibility(
            evidence,
            governance_decision=fake_qual,
            governance_decision_fingerprint="some_expected_fp",
        )


def test_genuinely_qualified_candidate_resolves_and_authorizes(tmp_path: Path) -> None:
    evidence = make_test_evidence(symbol="XAUUSD", timeframe="5m")
    rob = assess_research_robustness(evidence)
    qual = qualify_research_evidence(evidence, robustness_assessment=rob)
    assert qual.qualified is True

    cand_path = save_research_candidate(
        candidate_id="cand_momentum_5m_valid",
        evidence=evidence,
        operational_stability_score=0.85,
        governance_decision=qual,
        base_dir=tmp_path,
    )
    assert cand_path.exists()

    resolved = resolve_promoted_candidate(
        candidate_id="cand_momentum_5m_valid",
        base_dir=tmp_path,
    )
    assert resolved is not None
    assert resolved.candidate_id == "cand_momentum_5m_valid"
    assert resolved.governance_decision_fingerprint == qual.decision_fingerprint

    auth = authorize_production_runtime(
        resolved,
        symbol="XAUUSD",
        timeframe="5m",
        now=datetime.now(timezone.utc),
    )
    assert auth.candidate_id == "cand_momentum_5m_valid"
    assert auth.governance_decision_fingerprint == qual.decision_fingerprint

    readiness = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="cand_momentum_5m_valid",
        research_dir=tmp_path,
    )
    assert readiness["status"] == "READY"


def test_blocked_timeframe_does_not_borrow_candidate_and_independent_timeframes_remain_independent(tmp_path: Path) -> None:
    # 5m candidate is valid and qualified
    ev_5m = make_test_evidence(symbol="XAUUSD", timeframe="5m")
    rob_5m = assess_research_robustness(ev_5m)
    qual_5m = qualify_research_evidence(ev_5m, robustness_assessment=rob_5m)
    save_research_candidate(
        candidate_id="cand_5m_good",
        evidence=ev_5m,
        operational_stability_score=0.88,
        governance_decision=qual_5m,
        base_dir=tmp_path,
    )

    # 15m timeframe has NO candidate or blocked candidate
    readiness_15m = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="15m",
        research_dir=tmp_path,
    )
    assert readiness_15m["status"] == "BLOCKED"

    readiness_5m = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="5m",
        research_dir=tmp_path,
    )
    assert readiness_5m["status"] == "READY"
    assert readiness_5m["candidate_id"] == "cand_5m_good"
