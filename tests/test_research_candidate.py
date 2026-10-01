"""Unit tests for ResearchHypothesis and ResearchCandidate domain model invariants."""

from datetime import datetime, timezone
import pytest
import pandas as pd

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchCandidate,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
)
from src.evaluation.discovery_engine import DiscoveryCriteria, DiscoveryEngine
from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.live_production_decision import PromotedCandidateArtifact, Direction


@pytest.fixture
def sample_scope() -> DatasetScope:
    return DatasetScope(
        dataset_id="test_data_2025",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )


@pytest.fixture
def sample_assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(
        transaction_cost=0.0001,
        slippage=0.0001,
        latency_ms=100.0,
    )


@pytest.fixture
def sample_provenance() -> CodeProvenance:
    return CodeProvenance(
        commit_sha="a1b2c3d4e5f67890",
        repository_status="clean",
        author="Research Team",
    )


@pytest.fixture
def sample_hypothesis(
    sample_scope: DatasetScope,
    sample_assumptions: ExecutionAssumptions,
    sample_provenance: CodeProvenance,
) -> ResearchHypothesis:
    return ResearchHypothesis(
        statement="Momentum breakout on 5m XAUUSD",
        methodology_version="discovery_v1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
        random_seed=42,
    )


@pytest.fixture
def sample_evidence(sample_hypothesis: ResearchHypothesis) -> ResearchEvidence:
    spec = sample_hypothesis.to_experiment_spec()
    partitions = (
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2025-01-01",
            end_date="2025-01-05",
            total_return=0.15,
            max_drawdown=-0.05,
            sharpe_ratio=1.8,
            observations=100,
        ),
        EvidencePartition(
            role=EvidencePartitionRole.VALIDATION,
            start_date="2025-01-05",
            end_date="2025-01-07",
            total_return=0.08,
            max_drawdown=-0.03,
            sharpe_ratio=1.5,
            observations=50,
        ),
        EvidencePartition(
            role=EvidencePartitionRole.OUT_OF_SAMPLE,
            start_date="2025-01-07",
            end_date="2025-01-10",
            total_return=0.10,
            max_drawdown=-0.04,
            sharpe_ratio=1.6,
            observations=50,
        ),
        EvidencePartition(
            role=EvidencePartitionRole.WALK_FORWARD,
            start_date="2025-01-01",
            end_date="2025-01-10",
            total_return=0.12,
            max_drawdown=-0.04,
            sharpe_ratio=1.7,
            observations=5,
            additional_metrics={"positive_window_ratio": 0.8},
        ),
    )
    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=partitions,
        robustness_verdict={"is_robust": True, "passed": True},
        benchmark_comparison={"outperformed_benchmark": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        critique_notes="Qualified research evidence.",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def test_hypothesis_identity_is_stable(
    sample_scope: DatasetScope,
    sample_assumptions: ExecutionAssumptions,
    sample_provenance: CodeProvenance,
):
    """Verify hypothesis_id and fingerprint are deterministic across identical specifications."""
    h1 = ResearchHypothesis(
        statement="Momentum breakout strategy",
        methodology_version="v1",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"window": 10},
    )
    h2 = ResearchHypothesis(
        statement="Momentum breakout strategy",
        methodology_version="v1",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"window": 10},
    )

    assert h1.fingerprint == h2.fingerprint
    assert h1.hypothesis_id == h2.hypothesis_id
    assert h1.hypothesis_id.startswith("hyp_")

    # Modifying parameter must alter fingerprint
    h3 = ResearchHypothesis(
        statement="Momentum breakout strategy",
        methodology_version="v1",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=sample_scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"window": 20},
    )
    assert h3.fingerprint != h1.fingerprint
    assert h3.hypothesis_id != h1.hypothesis_id


def test_same_hypothesis_and_scope_do_not_create_duplicate_candidate(
    sample_assumptions: ExecutionAssumptions,
    sample_provenance: CodeProvenance,
):
    """Verify that discovery engine detects duplicate hypothesis candidates."""
    scope = DatasetScope(
        dataset_id="test_data_2025",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-02",
    )
    cand1 = CandidateSpec(
        generator_name="gen1",
        generator_version="1.0",
        strategy_name="momentum",
        parameters={"window": 10},
        hypothesis_template="Test hypothesis for window=10",
    )

    dates = pd.date_range("2025-01-01", "2025-01-02", freq="5min")
    prices = [2000.0 + i * 0.1 for i in range(len(dates))]
    df = pd.DataFrame({"timestamp": dates, "open": prices, "high": prices, "low": prices, "close": prices})

    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=df,
        candidates=[cand1, cand1],
        dataset_scope=scope,
        execution_assumptions=sample_assumptions,
        code_provenance=sample_provenance,
    )

    assert len(result.research_candidates) == 2
    # The second candidate must carry DUPLICATE_CANDIDATE rejection reason and be REJECTED
    dup_cand = result.research_candidates[1]
    assert RejectionReason.DUPLICATE_CANDIDATE in dup_cand.rejection_reasons or dup_cand.promotion_status == PromotionStatus.REJECTED


def test_research_candidate_requires_evidence(sample_hypothesis: ResearchHypothesis):
    """Verify that candidate with VALIDATED or PROMOTABLE status requires non-None evidence."""
    with pytest.raises(ValueError, match="Validated or Promotable ResearchCandidate requires non-None evidence"):
        ResearchCandidate(
            candidate_id="cand_test_01",
            hypothesis=sample_hypothesis,
            evidence=None,
            validation_status=PromotionStatus.PROMOTABLE,
            promotion_status=PromotionStatus.PROMOTABLE,
        )


def test_research_candidate_requires_provenance(
    sample_hypothesis: ResearchHypothesis,
    sample_evidence: ResearchEvidence,
):
    """Verify has_provenance() checks for non-empty commit_sha."""
    cand = ResearchCandidate(
        candidate_id="cand_test_01",
        hypothesis=sample_hypothesis,
        evidence=sample_evidence,
        validation_status=PromotionStatus.PROMOTABLE,
        promotion_status=PromotionStatus.PROMOTABLE,
    )
    assert cand.has_provenance() is True
    assert cand.has_evidence() is True
    assert cand.has_reproducible_scope() is True
    assert cand.has_validation_state() is True


def test_candidate_cannot_bypass_validation(sample_hypothesis: ResearchHypothesis):
    """Verify unvalidated or proposed candidates fail can_promote()."""
    cand = ResearchCandidate(
        candidate_id="cand_unvalidated",
        hypothesis=sample_hypothesis,
        evidence=None,
        validation_status=PromotionStatus.PROPOSED,
        promotion_status=PromotionStatus.PROPOSED,
    )
    assert cand.can_promote() is False


def test_unvalidated_candidate_cannot_be_promoted(sample_hypothesis: ResearchHypothesis):
    """Verify promoting candidate without evidence raises ValueError."""
    from src.evaluation.stability import CanonicalStabilityEvidence
    cand = ResearchCandidate(
        candidate_id="cand_unvalidated",
        hypothesis=sample_hypothesis,
        evidence=None,
        validation_status=PromotionStatus.PROPOSED,
        promotion_status=PromotionStatus.PROPOSED,
    )
    stab = CanonicalStabilityEvidence("momentum", 0.85)
    with pytest.raises(ValueError, match="Cannot promote ResearchCandidate 'cand_unvalidated': evidence is missing"):
        cand.promote_to_production_artifact(symbol="XAUUSD", timeframe="5m", canonical_stability=stab)


def test_candidate_lineage_reaches_original_experiment(
    sample_hypothesis: ResearchHypothesis,
    sample_evidence: ResearchEvidence,
):
    """Verify candidate lineage connects candidate -> evidence -> experiment spec -> hypothesis."""
    cand = ResearchCandidate(
        candidate_id="cand_lineage_test",
        hypothesis=sample_hypothesis,
        evidence=sample_evidence,
        validation_status=PromotionStatus.PROMOTABLE,
        promotion_status=PromotionStatus.PROMOTABLE,
    )
    assert cand.lineage_reaches_original_experiment() is True


def test_research_candidate_is_not_live_signal(
    sample_hypothesis: ResearchHypothesis,
    sample_evidence: ResearchEvidence,
):
    """Verify is_distinct_from_live_decision() enforces boundary from live signals."""
    cand = ResearchCandidate(
        candidate_id="cand_signal_boundary",
        hypothesis=sample_hypothesis,
        evidence=sample_evidence,
        validation_status=PromotionStatus.PROMOTABLE,
        promotion_status=PromotionStatus.PROMOTABLE,
    )
    assert cand.is_distinct_from_live_decision() is True
    assert not isinstance(cand, PromotedCandidateArtifact)


def test_missing_dataset_scope_fails_closed(
    sample_assumptions: ExecutionAssumptions,
    sample_provenance: CodeProvenance,
):
    """Verify constructing hypothesis with invalid dataset_scope fails closed."""
    with pytest.raises(TypeError, match="dataset_scope must be a DatasetScope instance"):
        ResearchHypothesis(
            statement="Test statement",
            methodology_version="v1",
            strategy_name="momentum",
            strategy_version="1.0",
            dataset_scope="invalid_scope",  # type: ignore
            execution_assumptions=sample_assumptions,
            code_provenance=sample_provenance,
            benchmark_reference="buy_and_hold",
        )


def test_missing_execution_assumptions_fail_closed(
    sample_scope: DatasetScope,
    sample_provenance: CodeProvenance,
):
    """Verify constructing hypothesis with invalid execution_assumptions fails closed."""
    with pytest.raises(TypeError, match="execution_assumptions must be an ExecutionAssumptions instance"):
        ResearchHypothesis(
            statement="Test statement",
            methodology_version="v1",
            strategy_name="momentum",
            strategy_version="1.0",
            dataset_scope=sample_scope,
            execution_assumptions=None,  # type: ignore
            code_provenance=sample_provenance,
            benchmark_reference="buy_and_hold",
        )


def test_missing_evidence_fails_closed(sample_hypothesis: ResearchHypothesis):
    """Verify creating candidate with missing evidence when status is VALIDATED fails closed."""
    with pytest.raises(ValueError, match="Validated or Promotable ResearchCandidate requires non-None evidence"):
        ResearchCandidate(
            candidate_id="cand_no_ev",
            hypothesis=sample_hypothesis,
            evidence=None,
            validation_status=PromotionStatus.VALIDATED,
            promotion_status=PromotionStatus.VALIDATED,
        )


def test_invalid_lifecycle_transition_is_rejected(
    sample_hypothesis: ResearchHypothesis,
    sample_evidence: ResearchEvidence,
):
    """Verify invalid status transition or status types are rejected."""
    with pytest.raises(TypeError, match="validation_status must be a PromotionStatus enum member"):
        ResearchCandidate(
            candidate_id="cand_invalid_status",
            hypothesis=sample_hypothesis,
            evidence=sample_evidence,
            validation_status="INVALID_STATUS",  # type: ignore
            promotion_status=PromotionStatus.PROMOTABLE,
        )

    with pytest.raises(TypeError, match="promotion_status must be a PromotionStatus enum member"):
        ResearchCandidate(
            candidate_id="cand_invalid_status2",
            hypothesis=sample_hypothesis,
            evidence=sample_evidence,
            validation_status=PromotionStatus.PROMOTABLE,
            promotion_status="INVALID_STATUS",  # type: ignore
        )
