"""Comprehensive Unit and Behavioral Regression Tests for Research Evidence Integrity Gate,

Temporal Lineage, Source-Order Validation, and Persistence Contracts.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.evidence_integrity import ResearchEvidenceIntegrityGate
from src.evaluation.hypothesis_generator import UnacceptedHypothesisError, accept_hypothesis_for_research
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_qualification import qualify_research_evidence
from src.evaluation.research_runner import (
    run_research_experiment,
    validate_and_prepare_dataset,
)
from src.evaluation.research_store import (
    load_research_experiment,
    save_research_experiment,
)


import numpy as np

def make_test_dataframe(n_rows: int = 100, start_date: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=n_rows, freq="1D")
    np.random.seed(42)
    close = 2000.0 + np.cumsum(np.random.randn(n_rows) * 10.0)
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
    })
    df["return"] = df["close"].pct_change().fillna(0.0)
    return df


def make_test_spec(
    dataset_id: str = "test_ds",
    start_date: str = "2025-01-01",
    end_date: str = "2025-04-10",
) -> ResearchExperimentSpec:
    ds = DatasetScope(
        dataset_id=dataset_id,
        symbol="XAUUSD",
        timeframe="1D",
        start_date=start_date,
        end_date=end_date,
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="a1b2c3d4e5f678901234567890abcdef12345678")
    return ResearchExperimentSpec(
        hypothesis="Testing trend strategy",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"window": 5},
        walk_forward_protocol=WalkForwardProtocol(train_size=30, test_size=20),
    )


# A. Non-monotonic timestamp input fails closed before sorting
def test_non_monotonic_timestamp_column_fails_closed():
    df = make_test_dataframe(50)
    # Swap rows 10 and 20 to make it non-monotonic
    df.iloc[10], df.iloc[20] = df.iloc[20].copy(), df.iloc[10].copy()
    ds = DatasetScope("test_ds", "XAUUSD", "1D", "2025-01-01", "2025-02-19")

    with pytest.raises(ValueError, match="non-monotonic"):
        validate_and_prepare_dataset(df, ds)


# B. Reversed input fails closed before sorting
def test_reversed_timestamp_input_fails_closed():
    df = make_test_dataframe(50)
    df_reversed = df.iloc[::-1].reset_index(drop=True)
    ds = DatasetScope("test_ds", "XAUUSD", "1D", "2025-01-01", "2025-02-19")

    with pytest.raises(ValueError, match="non-monotonic"):
        validate_and_prepare_dataset(df_reversed, ds)


# C. Duplicate timestamps fail closed
def test_duplicate_timestamps_fail_closed():
    df = make_test_dataframe(50)
    df.iloc[10, df.columns.get_loc("timestamp")] = df.iloc[9]["timestamp"]
    ds = DatasetScope("test_ds", "XAUUSD", "1D", "2025-01-01", "2025-02-19")

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_and_prepare_dataset(df, ds)


# D. Already ordered input succeeds
def test_already_ordered_dataset_succeeds():
    df = make_test_dataframe(50)
    ds = DatasetScope("test_ds", "XAUUSD", "1D", "2025-01-01", "2025-02-19")
    prepared = validate_and_prepare_dataset(df, ds)

    assert not prepared.empty
    assert prepared["timestamp"].is_monotonic_increasing
    assert prepared["timestamp"].is_unique


# E. DatetimeIndex and timestamp-column paths have equivalent ordering guarantees
def test_datetime_index_path_ordering_guarantees():
    df = make_test_dataframe(50)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_idx = df.set_index("timestamp")
    ds = DatasetScope("test_ds", "XAUUSD", "1D", "2025-01-01", "2025-02-19")

    prepared_idx = validate_and_prepare_dataset(df_idx, ds)
    assert prepared_idx.index.is_monotonic_increasing
    assert prepared_idx.index.is_unique

    # Reversed DatetimeIndex must fail
    df_idx_rev = df_idx.iloc[::-1]
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_and_prepare_dataset(df_idx_rev, ds)

    # Duplicate DatetimeIndex must fail
    df_idx_dup = df_idx.copy()
    new_idx_vals = df_idx.index.to_series()
    new_idx_vals.iloc[10] = new_idx_vals.iloc[9]
    df_idx_dup.index = pd.DatetimeIndex(new_idx_vals)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_and_prepare_dataset(df_idx_dup, ds)


# F. Exact partition timestamps are populated by new evidence
def test_new_evidence_populates_exact_utc_timestamps():
    df = make_test_dataframe(100)
    spec = make_test_spec()
    evidence = run_research_experiment(spec, df=df)

    assert len(evidence.partitions) >= 3
    for p in evidence.partitions:
        assert p.start_timestamp_utc is not None
        assert p.end_timestamp_utc is not None
        assert "T" in p.start_timestamp_utc
        assert "T" in p.end_timestamp_utc
        assert p.start_timestamp_utc <= p.end_timestamp_utc


# G. Exact timestamps survive JSON persistence
def test_exact_timestamps_survive_json_persistence(tmp_path: Path):
    df = make_test_dataframe(100)
    spec = make_test_spec()
    evidence = run_research_experiment(spec, df=df)

    saved_path = save_research_experiment(evidence, base_dir=tmp_path)
    loaded = load_research_experiment(saved_path)

    for orig_p, loaded_p in zip(evidence.partitions, loaded.partitions):
        assert loaded_p.start_timestamp_utc == orig_p.start_timestamp_utc
        assert loaded_p.end_timestamp_utc == orig_p.end_timestamp_utc


# H. Legacy artifacts without exact timestamps remain loadable
def test_legacy_artifacts_without_exact_timestamps_remain_loadable(tmp_path: Path):
    from tests.test_research_qualification import make_valid_evidence
    evidence = make_valid_evidence()

    # Convert to dict and strip start_timestamp_utc / end_timestamp_utc
    ev_dict = evidence.as_dict()
    for p_data in ev_dict["partitions"]:
        p_data.pop("start_timestamp_utc", None)
        p_data.pop("end_timestamp_utc", None)

    target_dir = tmp_path / evidence.experiment_fingerprint
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / "evidence.json"
    file_path.write_text(json.dumps(ev_dict, indent=2), encoding="utf-8")

    loaded_legacy = load_research_experiment(file_path)
    for p in loaded_legacy.partitions:
        assert p.start_timestamp_utc is None
        assert p.end_timestamp_utc is None

    # Qualification succeeds for legacy artifacts if other policies pass
    from src.evaluation.research_robustness import assess_research_robustness
    qual_res = qualify_research_evidence(loaded_legacy, robustness_assessment=assess_research_robustness(loaded_legacy))
    assert qual_res.qualified is True


# I. Invalid timestamp boundaries fail closed
def test_invalid_timestamp_boundaries_fail_closed():
    with pytest.raises(ValueError, match="cannot be later than"):
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2025-01-01",
            end_date="2025-01-02",
            total_return=0.1,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            observations=50,
            start_timestamp_utc="2025-01-05T00:00:00+00:00",
            end_timestamp_utc="2025-01-01T00:00:00+00:00",  # Reversed!
        )


# J. Partition overlap fails closed
def test_partition_overlap_fails_closed():
    spec = make_test_spec()
    p_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-10",
        total_return=0.1,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=50,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-10T00:00:00+00:00",
    )
    p_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-05",  # Overlaps with IS!
        end_date="2025-01-15",
        total_return=0.1,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2025-01-05T00:00:00+00:00",
        end_timestamp_utc="2025-01-15T00:00:00+00:00",
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(p_is, p_oos),
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    gate_res = ResearchEvidenceIntegrityGate.validate(evidence)
    assert gate_res.valid is False
    assert RejectionReason.FAILED_VALIDATION in gate_res.rejection_reasons


# K. Empty partitions fail closed
def test_empty_partitions_fail_closed():
    spec = make_test_spec()
    p_empty = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-02",
        total_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        observations=0,  # Empty!
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-02T00:00:00+00:00",
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(p_empty,),
        robustness_verdict={"passed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    gate_res = ResearchEvidenceIntegrityGate.validate(evidence, min_observations=1)
    assert gate_res.valid is False
    assert RejectionReason.INSUFFICIENT_DATA in gate_res.rejection_reasons


# L. Walk-forward evidence uses actual test-window span
def test_walk_forward_evidence_uses_actual_test_window_span():
    df = make_test_dataframe(100, start_date="2025-01-01")
    spec = make_test_spec(start_date="2025-01-01", end_date="2025-04-10")
    evidence = run_research_experiment(spec, df=df)

    wf_p = [p for p in evidence.partitions if p.role == EvidencePartitionRole.WALK_FORWARD][0]

    # Full dataset starts at 2025-01-01.
    # With 100 rows, train_size=30, test_size=20:
    # First test window starts at row index 30 (2025-01-31).
    assert wf_p.start_date != "2025-01-01"
    assert wf_p.start_date == "2025-01-31"


# M. Governed lifecycle protection
def test_unaccepted_hypothesis_cannot_enter_runner():
    spec_spec = make_test_spec()
    hypothesis = ResearchHypothesis.from_experiment_spec(spec_spec)
    assert hypothesis.status == HypothesisStatus.GENERATED

    with pytest.raises(UnacceptedHypothesisError, match="is not accepted for research execution"):
        run_research_experiment(hypothesis)

    # Accept hypothesis and execute
    accepted = accept_hypothesis_for_research(hypothesis)
    assert accepted.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH

    df = make_test_dataframe(100)
    evidence = run_research_experiment(accepted, df=df)
    assert isinstance(evidence, ResearchEvidence)
    assert evidence.spec.fingerprint == hypothesis.fingerprint
