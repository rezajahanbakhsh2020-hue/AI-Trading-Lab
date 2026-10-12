"""
Adversarial regression and forensic tests for XAUUSD 30m, 1h, and 4h higher-timeframe
dataset resampling, schema validation, grid alignment, provenance manifests, atomic commit protocol,
research execution, and fail-closed readiness gating.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

import pandas as pd

from src.data.resample_mtf_datasets import (
    DEFAULT_15M_CSV,
    DEFAULT_15M_MANIFEST,
    resample_xauusd_timeframe,
)
from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.live_execution_runtime import verify_timeframe_production_readiness
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import resolve_historical_dataset, run_research_experiment
from src.evaluation.research_store import resolve_promoted_candidate


def test_preserve_existing_15m_artifacts() -> None:
    """Phase 1 Preservation Invariant: Verify 15m CSV and manifest remain 100% unchanged."""
    csv_bytes = DEFAULT_15M_CSV.read_bytes()
    manifest_bytes = DEFAULT_15M_MANIFEST.read_bytes()

    assert hashlib.sha256(csv_bytes).hexdigest() == "45467b12196b1f762a6d78742fee5bca1f69abd744b0e1a9eeffb226ccc1ff56"
    df, meta = verify_dataset_and_manifest_integrity(DEFAULT_15M_CSV, DEFAULT_15M_MANIFEST)
    assert len(df) == 11989
    assert meta["row_count"] == 11989


@pytest.mark.parametrize(
    "timeframe,dataset_id,expected_rows",
    [
        ("30m", "ds_moving_average_30m", 5980),
        ("1h", "ds_moving_average_1h", 2977),
        ("4h", "ds_moving_average_4h", 652),
    ],
)
def test_mtf_dataset_resampling_and_manifest_integrity(
    timeframe: str, dataset_id: str, expected_rows: int, tmp_path: Path
) -> None:
    """Test resampling, grid alignment, and manifest verification for 30m, 1h, 4h."""
    out_dir = tmp_path / f"xauusd_{timeframe}"
    out_p, meta = resample_xauusd_timeframe(
        timeframe=timeframe,
        output_dir=tmp_path,
    )

    assert out_p.exists()
    assert meta["dataset_id"] == dataset_id
    assert meta["row_count"] == expected_rows
    assert meta["validation_summary"]["timestamps_grid_aligned"] is True
    assert meta["validation_summary"]["resampling_group_completeness_valid"] is True

    manifest_p = out_p.parent / out_p.name.replace(".csv", ".manifest.json")
    commit_p = out_p.parent / out_p.name.replace(".csv", ".commit.json")

    df, man = verify_dataset_and_manifest_integrity(out_p, manifest_p, commit_p)
    assert len(df) == expected_rows
    assert man["output_sha256"] == meta["output_sha256"]


@pytest.mark.parametrize("timeframe", ["30m", "1h", "4h"])
def test_mtf_resampling_fault_injection_interrupted_publication(
    timeframe: str, tmp_path: Path
) -> None:
    """Test fault injection during atomic multi-file publication for higher timeframes."""
    # Pre-populate valid initial dataset
    out_p, meta1 = resample_xauusd_timeframe(
        timeframe=timeframe,
        output_dir=tmp_path,
    )

    manifest_p = out_p.parent / out_p.name.replace(".csv", ".manifest.json")
    commit_p = out_p.parent / out_p.name.replace(".csv", ".commit.json")

    initial_csv_bytes = out_p.read_bytes()
    initial_manifest_bytes = manifest_p.read_bytes()

    # Case A: Staging failure before CSV replacement preserves initial files
    with pytest.raises(RuntimeError, match="Atomic dataset persistence failed"):
        resample_xauusd_timeframe(
            timeframe=timeframe,
            output_dir=tmp_path,
            simulate_failure_step="before_csv_replace",
        )

    assert out_p.read_bytes() == initial_csv_bytes
    assert manifest_p.read_bytes() == initial_manifest_bytes

    # Case B: Interrupted publication step after CSV replacement (tampered content)
    out_p.write_text("timestamp,open,high,low,close\n2026-03-26T03:00:00+00:00,100,105,99,102\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Mismatched dataset/manifest generation detected"):
        verify_dataset_and_manifest_integrity(out_p, manifest_p, commit_p)


@pytest.mark.parametrize(
    "timeframe,dataset_id,expected_rows",
    [
        ("30m", "ds_moving_average_30m", 5980),
        ("1h", "ds_moving_average_1h", 2977),
        ("4h", "ds_moving_average_4h", 652),
    ],
)
def test_mtf_dataset_resolution_and_research_experiment(
    timeframe: str, dataset_id: str, expected_rows: int
) -> None:
    """Test resolve_historical_dataset and run_research_experiment for 30m, 1h, 4h."""
    ds = DatasetScope(
        dataset_id=dataset_id,
        symbol="XAUUSD",
        timeframe=timeframe,
        start_date="2026-03-26",
        end_date="2026-09-25",
    )

    resolved = resolve_historical_dataset(ds)
    assert not resolved.empty
    assert len(resolved) == expected_rows

    hyp = ResearchHypothesis(
        statement=f"XAUUSD {timeframe} Baseline Strategy",
        methodology_version="1.0",
        strategy_name="baseline",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=0.0),
        code_provenance=CodeProvenance(commit_sha="dd9988845b7c310db2c4b1387d78524f985381e1"),
        benchmark_reference="buy_and_hold",
        walk_forward_protocol=WalkForwardProtocol(train_size=100, test_size=20),
        parameters={"fast_window": 20, "slow_window": 50},
        status=HypothesisStatus.GENERATED,
    )

    accepted = accept_hypothesis_for_research(hyp)
    evidence = run_research_experiment(spec=accepted)

    assert evidence.experiment_fingerprint == accepted.fingerprint
    assert len(evidence.partitions) > 0


@pytest.mark.parametrize("timeframe", ["30m", "1h", "4h"])
def test_unpromoted_mtf_candidates_fail_closed(timeframe: str) -> None:
    """Test that unpromoted 30m, 1h, and 4h candidates remain BLOCKED in readiness checks."""
    resolved = resolve_promoted_candidate(symbol="XAUUSD", timeframe=timeframe)
    assert resolved is None

    readiness = verify_timeframe_production_readiness(symbol="XAUUSD", timeframe=timeframe)
    assert readiness["status"] == "BLOCKED"
    assert readiness["reason_code"] == "PromotionUnavailable"
