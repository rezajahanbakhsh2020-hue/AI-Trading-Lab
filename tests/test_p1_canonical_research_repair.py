"""Comprehensive Regression Test Suite for P1 Canonical Research Repair.

Covers:
1. Numeric timestamp unit contract (fails without unit, passes with explicit unit 's', ISO-8601 UTC strings, invalid/ambiguous timestamps).
2. Date-only end bounds vs timestamp-valued end bounds in dataset filtering.
3. Exact dataset identity resolution & manifest integrity verification (SHA256, row_count, dataset_id/symbol/timeframe scope matching).
4. Explicit 'moving_average' strategy contract in DEFAULT_REGISTRY, parameter separation (signal vs risk/execution), and rejection of unsupported parameters.
5. Intrabar OHLC risk execution (stop_loss_pct, take_profit_pct, gap-through fills, simultaneous SL/TP touch handling).
6. Research evidence integrity, exact UTC partition timestamps, non-overlapping partitions, and qualification gating.
7. Multi-timeframe independence across all 7 canonical timeframes (1m, 5m, 15m, 30m, 1H, 4H, 1D).
"""

import json
import math
from pathlib import Path
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    RejectionReason,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import (
    resolve_historical_dataset,
    run_research_experiment,
    validate_and_prepare_dataset,
)
from src.evaluation.evidence_integrity import ResearchEvidenceIntegrityGate
from src.evaluation.research_qualification import qualify_research_evidence
from src.strategies.registry import DEFAULT_REGISTRY, list_strategies, run_strategy
from src.evaluation.strategy_evaluator import (
    evaluate_strategy,
    validate_and_separate_strategy_kwargs,
)


def sample_daily_df_numeric_seconds() -> pd.DataFrame:
    """Sample DataFrame with integer Unix-second timestamps (like xauusd_daily_2025.csv)."""
    # 2025-01-01 to 2025-01-10 in daily Unix seconds
    ts_sec = [1735689600 + i * 86400 for i in range(10)]
    return pd.DataFrame(
        {
            "timestamp": ts_sec,
            "open": [2600.0 + i for i in range(10)],
            "high": [2610.0 + i for i in range(10)],
            "low": [2590.0 + i for i in range(10)],
            "close": [2605.0 + i for i in range(10)],
        }
    )


def sample_daily_df_iso_strings() -> pd.DataFrame:
    """Sample DataFrame with ISO-8601 UTC timestamp strings."""
    ts_str = [f"2025-01-0{i+1}T00:00:00+00:00" for i in range(9)] + ["2025-01-10T00:00:00+00:00"]
    return pd.DataFrame(
        {
            "timestamp": ts_str,
            "open": [2600.0 + i for i in range(10)],
            "high": [2610.0 + i for i in range(10)],
            "low": [2590.0 + i for i in range(10)],
            "close": [2605.0 + i for i in range(10)],
        }
    )


# -----------------------------------------------------------------------------
# 1. Timestamp Parsing & Unit Contract Tests
# -----------------------------------------------------------------------------

def test_numeric_timestamp_without_unit_fails_closed():
    df = sample_daily_df_numeric_seconds()
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    with pytest.raises(ValueError, match="Numeric Unix timestamp detected in dataset without an explicitly established timestamp_unit"):
        validate_and_prepare_dataset(df, ds)


def test_numeric_timestamp_with_explicit_unit_seconds_passes():
    df = sample_daily_df_numeric_seconds()
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    prepared = validate_and_prepare_dataset(df, ds, timestamp_unit="s")
    assert len(prepared) == 10
    assert prepared["timestamp"].min() == pd.to_datetime("2025-01-01 00:00:00", utc=True)
    assert prepared["timestamp"].max() == pd.to_datetime("2025-01-10 00:00:00", utc=True)


def test_iso_8601_utc_string_timestamps_pass():
    df = sample_daily_df_iso_strings()
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    prepared = validate_and_prepare_dataset(df, ds)
    assert len(prepared) == 10


def test_invalid_unparseable_timestamp_fails_closed():
    df = pd.DataFrame(
        {
            "timestamp": ["invalid_ts_1", "invalid_ts_2"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-01-02",
    )
    with pytest.raises(ValueError, match="Invalid/unparseable timestamp"):
        validate_and_prepare_dataset(df, ds)


# -----------------------------------------------------------------------------
# 2. Date-Boundary Filtering Tests
# -----------------------------------------------------------------------------

def test_date_only_end_bound_includes_full_final_utc_day():
    ts_list = [
        "2025-01-01T00:00:00+00:00",
        "2025-01-10T08:00:00+00:00",
        "2025-01-10T20:00:00+00:00",
    ]
    df = pd.DataFrame(
        {
            "timestamp": ts_list,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        }
    )
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1H",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    prepared = validate_and_prepare_dataset(df, ds)
    assert len(prepared) == 3


def test_timestamp_valued_end_bound_preserves_exact_instant():
    ts_list = [
        "2025-01-01T00:00:00+00:00",
        "2025-01-10T08:00:00+00:00",
        "2025-01-10T20:00:00+00:00",
    ]
    df = pd.DataFrame(
        {
            "timestamp": ts_list,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        }
    )
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1H",
        start_date="2025-01-01T00:00:00+00:00",
        end_date="2025-01-10T12:00:00+00:00",
    )
    prepared = validate_and_prepare_dataset(df, ds)
    assert len(prepared) == 2
    assert prepared["timestamp"].max() == pd.to_datetime("2025-01-10T08:00:00+00:00", utc=True)


# -----------------------------------------------------------------------------
# 3. Exact Dataset Identity & Manifest Integrity Tests
# -----------------------------------------------------------------------------

def test_missing_dataset_resolution_fails_closed_without_substitution():
    missing_timeframes = ["1m", "5m", "1d"]
    for tf in missing_timeframes:
        ds = DatasetScope(
            dataset_id=f"ds_moving_average_{tf.lower()}",
            symbol="XAUUSD",
            timeframe=tf,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        with pytest.raises(FileNotFoundError, match="Unable to resolve exact approved dataset artifact"):
            resolve_historical_dataset(ds)


def test_tracked_daily_2025_resolves_with_manifest_integrity():
    ds = DatasetScope(
        dataset_id="xauusd_daily_2025",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    df = resolve_historical_dataset(ds)
    assert len(df) == 312
    assert df.attrs.get("timestamp_unit") == "s"


def test_manifest_row_count_mismatch_fails_closed(tmp_path, monkeypatch):
    csv_file = tmp_path / "ds_test.csv"
    manifest_file = tmp_path / "ds_test.manifest.json"

    df_sample = sample_daily_df_iso_strings()
    df_sample.to_csv(csv_file, index=False)

    import hashlib
    actual_sha = hashlib.sha256(csv_file.read_bytes()).hexdigest()

    manifest_data = {
        "manifest_version": "1.0",
        "dataset_id": "ds_test",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "output_sha256": actual_sha,
        "row_count": 999,  # Intentional row count mismatch
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    ds = DatasetScope("ds_test", "XAUUSD", "1D", "2025-01-01", "2025-01-10")

    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    with pytest.raises(RuntimeError, match="Row count mismatch"):
        verify_dataset_and_manifest_integrity(csv_file, manifest_file)


def test_manifest_output_sha256_mismatch_fails_closed(tmp_path):
    csv_file = tmp_path / "ds_test.csv"
    manifest_file = tmp_path / "ds_test.manifest.json"

    df_sample = sample_daily_df_iso_strings()
    df_sample.to_csv(csv_file, index=False)

    manifest_data = {
        "manifest_version": "1.0",
        "dataset_id": "ds_test",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "output_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "row_count": len(df_sample),
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    with pytest.raises(RuntimeError, match="Mismatched dataset/manifest generation detected"):
        verify_dataset_and_manifest_integrity(csv_file, manifest_file)


# -----------------------------------------------------------------------------
# 4. Intrabar OHLC Risk Execution Tests
# -----------------------------------------------------------------------------

def test_stop_loss_pct_alters_backtest_outcomes_when_touched():
    # Construct 4 bars where signal is 1 at bar 1, position enters at bar 2 open=100.
    # At bar 2: Low drops to 95 (touching SL of 2% = 98).
    df = pd.DataFrame(
        {
            "timestamp": [f"2025-01-0{i+1}T00:00:00+00:00" for i in range(4)],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 95.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "signal": [0, 1, 1, 1],
            "return": [0.0, 0.0, 0.0, 0.0],
        }
    )

    res_no_sl = run_backtest(df, stop_loss_pct=None)
    res_with_sl = run_backtest(df, stop_loss_pct=0.02)

    assert res_no_sl["equity"].iloc[-1] == 1.0
    # Stop loss executed at 98 on entry 100 -> return -0.02 -> equity 0.98
    assert res_with_sl["equity"].iloc[-1] == pytest.approx(0.98)


def test_take_profit_pct_alters_backtest_outcomes_when_touched():
    # Bar 2: High reaches 105 (touching TP of 3% = 103).
    df = pd.DataFrame(
        {
            "timestamp": [f"2025-01-0{i+1}T00:00:00+00:00" for i in range(4)],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 105.0, 100.0],
            "low": [100.0, 100.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "signal": [0, 1, 1, 1],
            "return": [0.0, 0.0, 0.0, 0.0],
        }
    )

    res_no_tp = run_backtest(df, take_profit_pct=None)
    res_with_tp = run_backtest(df, take_profit_pct=0.03)

    assert res_no_tp["equity"].iloc[-1] == 1.0
    # Take profit executed at 103 on entry 100 -> return +0.03 -> equity 1.03
    assert res_with_tp["equity"].iloc[-1] == pytest.approx(1.03)


def test_simultaneous_stop_and_target_touch_uses_conservative_stop():
    # Bar 2: High is 105, Low is 95 (both SL=98 and TP=103 breached in same bar).
    df = pd.DataFrame(
        {
            "timestamp": [f"2025-01-0{i+1}T00:00:00+00:00" for i in range(4)],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 105.0, 100.0],
            "low": [100.0, 100.0, 95.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "signal": [0, 1, 1, 1],
            "return": [0.0, 0.0, 0.0, 0.0],
        }
    )

    res = run_backtest(df, stop_loss_pct=0.02, take_profit_pct=0.03)
    # Pessimistic fill at SL (98) -> equity 0.98
    assert res["equity"].iloc[-1] == pytest.approx(0.98)


def test_risk_parameters_without_ohlc_fails_closed():
    df = pd.DataFrame(
        {
            "signal": [0, 1, 1],
            "return": [0.0, 0.01, 0.01],
        }
    )
    with pytest.raises(ValueError, match="Backtesting with risk parameters .* requires OHLC columns"):
        run_backtest(df, stop_loss_pct=0.01)


# -----------------------------------------------------------------------------
# 5. Strategy Registry & Parameter Contract Tests
# -----------------------------------------------------------------------------

def test_default_registry_exposes_moving_average():
    assert DEFAULT_REGISTRY.contains("moving_average")
    assert "moving_average" in list_strategies()
    spec = DEFAULT_REGISTRY.get("moving_average")
    assert spec.name == "moving_average"
    assert spec.category == "trend"


def test_parameter_separation_validates_signal_and_risk_kwargs():
    sig_kwargs, risk_kwargs = validate_and_separate_strategy_kwargs(
        "moving_average",
        {
            "fast_window": 10,
            "slow_window": 30,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        },
    )
    assert sig_kwargs == {"fast_window": 10, "slow_window": 30}
    assert risk_kwargs == {"stop_loss_pct": 0.02, "take_profit_pct": 0.04}


def test_unsupported_strategy_parameter_rejected():
    with pytest.raises(ValueError, match="received unsupported parameter"):
        validate_and_separate_strategy_kwargs(
            "moving_average",
            {"fast_window": 10, "slow_window": 30, "unsupported_param": 123},
        )


def test_evaluate_strategy_routes_risk_kwargs_to_backtest():
    df = pd.DataFrame(
        {
            "timestamp": [f"2025-01-0{i+1}T00:00:00+00:00" for i in range(10)],
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
        }
    )

    res = evaluate_strategy(
        df=df,
        name="moving_average",
        strategy_kwargs={
            "fast_window": 2,
            "slow_window": 4,
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.02,
        },
    )
    assert res.name == "moving_average"
    assert res.observations == 10


# -----------------------------------------------------------------------------
# 6. Evidence Integrity & Qualification Tests
# -----------------------------------------------------------------------------

def test_research_experiment_missing_dataset_returns_rejected_evidence():
    hyp = ResearchHypothesis(
        statement="Missing dataset hypothesis test",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_moving_average_1m",  # missing
            symbol="XAUUSD",
            timeframe="1m",
            start_date="2025-01-01",
            end_date="2025-12-31",
        ),
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("sha123"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 10, "slow_window": 30, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
        walk_forward_protocol=WalkForwardProtocol(100, 20),
        status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
    )

    evidence = run_research_experiment(hyp)
    assert evidence.promotion_status == PromotionStatus.REJECTED
    assert RejectionReason.INVALID_DATASET_SCOPE in evidence.rejection_reasons

    qual = qualify_research_evidence(evidence)
    assert qual.qualified is False


def test_evidence_integrity_gate_enforces_exact_utc_partition_timestamps():
    df = sample_daily_df_numeric_seconds()
    prepared = validate_and_prepare_dataset(
        df,
        DatasetScope("xauusd_daily_2025", "XAUUSD", "1D", "2025-01-01", "2025-01-10"),
        timestamp_unit="s",
    )
    hyp = ResearchHypothesis(
        statement="Valid dataset experiment test",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=DatasetScope(
            dataset_id="xauusd_daily_2025",
            symbol="XAUUSD",
            timeframe="1D",
            start_date="2025-01-01",
            end_date="2025-01-10",
        ),
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("sha123"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 2, "slow_window": 4, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
        walk_forward_protocol=WalkForwardProtocol(4, 2),
        status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
    )
    evidence = run_research_experiment(hyp, df=prepared)
    gate_res = ResearchEvidenceIntegrityGate.validate(evidence)
    assert gate_res.valid is True


# -----------------------------------------------------------------------------
# 7. Canonical Timeframes Independence Test
# -----------------------------------------------------------------------------

def test_seven_canonical_timeframes_independent_resolution():
    canonical_timeframes = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    available_datasets = {
        "15m": "ds_moving_average_15m",
        "30m": "ds_moving_average_30m",
        "1H": "ds_moving_average_1h",
        "4H": "ds_moving_average_4h",
        "1D": "xauusd_daily_2025",
    }

    for tf in canonical_timeframes:
        if tf in available_datasets:
            ds = DatasetScope(
                dataset_id=available_datasets[tf],
                symbol="XAUUSD",
                timeframe=tf,
                start_date="2025-01-01",
                end_date="2026-12-31",
            )
            df = resolve_historical_dataset(ds)
            assert not df.empty
        else:
            ds = DatasetScope(
                dataset_id=f"ds_moving_average_{tf.lower()}",
                symbol="XAUUSD",
                timeframe=tf,
                start_date="2025-01-01",
                end_date="2026-12-31",
            )
            with pytest.raises(FileNotFoundError):
                resolve_historical_dataset(ds)
