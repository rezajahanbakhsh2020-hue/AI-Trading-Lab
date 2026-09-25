"""Focused unit tests for Research Constitution & Experiment Evidence Contract."""

import json
import tempfile
from pathlib import Path
import pytest

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
    compute_experiment_fingerprint,
)
from src.evaluation.research_store import (
    load_research_experiment,
    reconstruct_research_evidence,
    save_research_experiment,
)


def _make_valid_spec(
    hypothesis: str = "Momentum breakout test",
    strategy_name: str = "momentum",
    commit_sha: str = "a1b2c3d4",
    benchmark_reference: str = "BUY_AND_HOLD",
) -> ResearchExperimentSpec:
    dataset_scope = DatasetScope(
        dataset_id="ds_xauusd_1h",
        symbol="XAUUSD",
        timeframe="1h",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    execution_assumptions = ExecutionAssumptions(
        transaction_cost=0.0001,
        slippage=0.0002,
        latency_ms=10.0,
    )
    code_provenance = CodeProvenance(
        commit_sha=commit_sha,
        repository_status="clean",
        author="researcher@lab",
    )
    return ResearchExperimentSpec(
        hypothesis=hypothesis,
        methodology_version="1.0.0",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference=benchmark_reference,
        parameters={"lookback": 20},
        random_seed=42,
    )


def test_valid_experiment_spec_accepted():
    spec = _make_valid_spec()
    assert spec.hypothesis == "Momentum breakout test"
    assert spec.strategy_name == "momentum"
    assert spec.fingerprint is not None
    assert len(spec.fingerprint) == 64


def test_missing_required_benchmark_fails_closed():
    with pytest.raises(ValueError, match="benchmark_reference must be a non-empty string"):
        _make_valid_spec(benchmark_reference="")


def test_invalid_experiment_spec_fails_closed():
    # Empty hypothesis
    with pytest.raises(ValueError, match="hypothesis must be a non-empty string"):
        _make_valid_spec(hypothesis="")

    # Missing commit_sha
    with pytest.raises(ValueError, match="commit_sha must be a non-empty string"):
        _make_valid_spec(commit_sha="")

    # Invalid dataset start/end dates
    with pytest.raises(ValueError, match="cannot be later than end_date"):
        DatasetScope(
            dataset_id="ds1",
            symbol="XAUUSD",
            timeframe="1h",
            start_date="2023-12-31",
            end_date="2023-01-01",
        )

    # Negative execution cost
    with pytest.raises(ValueError, match="transaction_cost cannot be negative"):
        ExecutionAssumptions(
            transaction_cost=-0.01,
            slippage=0.0001,
            latency_ms=0.0,
        )


def test_invalid_partition_date_range_is_rejected():
    with pytest.raises(ValueError, match="cannot be later than end_date"):
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2023-12-31",
            end_date="2023-01-01",
            total_return=0.10,
            max_drawdown=-0.05,
            sharpe_ratio=1.2,
        )


def test_non_finite_partition_metric_is_rejected():
    with pytest.raises(ValueError, match="must be a finite float"):
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2023-01-01",
            end_date="2023-06-30",
            total_return=float("nan"),
            max_drawdown=-0.05,
            sharpe_ratio=1.2,
        )


def test_fingerprint_is_deterministic_and_ignores_timestamps():
    spec1 = _make_valid_spec()
    spec2 = _make_valid_spec()

    assert spec1.fingerprint == spec2.fingerprint

    partition = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-06-30",
        total_return=0.15,
        max_drawdown=-0.05,
        sharpe_ratio=1.8,
    )

    ev1 = ResearchEvidence(
        experiment_fingerprint=spec1.fingerprint,
        spec=spec1,
        partitions=(partition,),
        promotion_status=PromotionStatus.VALIDATED,
        created_at_utc="2023-07-01T00:00:00Z",
    )

    ev2 = ResearchEvidence(
        experiment_fingerprint=spec2.fingerprint,
        spec=spec2,
        partitions=(partition,),
        promotion_status=PromotionStatus.VALIDATED,
        created_at_utc="2023-07-02T12:00:00Z",
    )

    assert ev1.spec.fingerprint == ev2.spec.fingerprint
    assert ev1.experiment_fingerprint == ev2.experiment_fingerprint


def test_evidence_id_changes_when_authoritative_content_changes():
    spec = _make_valid_spec()
    partition = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-06-30",
        total_return=0.15,
        max_drawdown=-0.05,
        sharpe_ratio=1.8,
    )

    ev1 = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(partition,),
        promotion_status=PromotionStatus.VALIDATED,
        critique_notes="Initial pass",
    )

    ev2 = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(partition,),
        promotion_status=PromotionStatus.VALIDATED,
        critique_notes="Updated pass with extra critique notes",
    )

    assert ev1.evidence_id != ev2.evidence_id


def test_provenance_and_evidence_linkage():
    spec = _make_valid_spec()
    partition_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-06-30",
        total_return=0.12,
        max_drawdown=-0.04,
        sharpe_ratio=1.5,
    )
    partition_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2023-07-01",
        end_date="2023-12-31",
        total_return=0.08,
        max_drawdown=-0.03,
        sharpe_ratio=1.2,
    )

    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(partition_is, partition_oos),
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    assert evidence.spec.code_provenance.commit_sha == "a1b2c3d4"
    assert len(evidence.partitions) == 2
    assert evidence.partitions[0].role == EvidencePartitionRole.IN_SAMPLE
    assert evidence.partitions[1].role == EvidencePartitionRole.OUT_OF_SAMPLE


def test_mismatched_fingerprint_raises_error():
    spec = _make_valid_spec()
    fake_fingerprint = "0" * 64

    with pytest.raises(ValueError, match="does not match spec fingerprint"):
        ResearchEvidence(
            experiment_fingerprint=fake_fingerprint,
            spec=spec,
            partitions=(),
            promotion_status=PromotionStatus.EXPERIMENTAL,
        )


def test_rejection_reasons_represented():
    spec = _make_valid_spec()

    with pytest.raises(ValueError, match="REJECTED evidence must specify at least one RejectionReason"):
        ResearchEvidence(
            experiment_fingerprint=spec.fingerprint,
            spec=spec,
            partitions=(),
            promotion_status=PromotionStatus.REJECTED,
            rejection_reasons=(),
        )

    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(),
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(
            RejectionReason.FAILED_OOS,
            RejectionReason.FAILED_ROBUSTNESS,
        ),
        critique_notes="Drawdown in OOS exceeded risk budget.",
    )

    assert evidence.promotion_status == PromotionStatus.REJECTED
    assert RejectionReason.FAILED_OOS in evidence.rejection_reasons
    assert "Drawdown in OOS" in evidence.critique_notes


def test_malformed_persisted_evidence_fails_closed():
    spec = _make_valid_spec()
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(),
        promotion_status=PromotionStatus.PROPOSED,
    )

    raw_dict = evidence.as_dict()
    # Remove required execution assumptions from spec dict
    del raw_dict["spec"]["execution_assumptions"]

    with pytest.raises(ValueError, match="Missing or invalid 'execution_assumptions'"):
        reconstruct_research_evidence(raw_dict)


def test_persistence_idempotent_and_overwrite_protection():
    spec = _make_valid_spec()
    partition = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2023-01-01",
        end_date="2023-12-31",
        total_return=0.10,
        max_drawdown=-0.05,
        sharpe_ratio=1.4,
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(partition,),
        promotion_status=PromotionStatus.VALIDATED,
        created_at_utc="2023-07-01T00:00:00Z",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # First save
        path1 = save_research_experiment(evidence, base_dir=tmpdir)
        assert path1.exists()

        # Idempotent second save of identical evidence
        path2 = save_research_experiment(evidence, base_dir=tmpdir)
        assert path2 == path1

        # Attempt to save conflicting evidence under same fingerprint
        conflicting_evidence = ResearchEvidence(
            experiment_fingerprint=spec.fingerprint,
            spec=spec,
            partitions=(partition,),
            promotion_status=PromotionStatus.VALIDATED,
            critique_notes="Conflicting critique notes",
        )

        with pytest.raises(FileExistsError, match="Cannot overwrite existing research evidence artifact"):
            save_research_experiment(conflicting_evidence, base_dir=tmpdir)
