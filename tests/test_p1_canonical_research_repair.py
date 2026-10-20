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
    _calculate_max_drawdown,
)
from src.evaluation.evidence_integrity import ResearchEvidenceIntegrityGate
from src.evaluation.research_qualification import qualify_research_evidence
from src.strategies.registry import DEFAULT_REGISTRY, list_strategies, run_strategy
from src.evaluation.strategy_evaluator import (
    evaluate_strategy,
    validate_and_separate_strategy_kwargs,
)


def test_walk_forward_drawdown_includes_initial_capital():
    # A loss on the first OOS period is a drawdown from starting equity 1.0.
    assert _calculate_max_drawdown([-0.05, 0.02]) == pytest.approx(-0.05)
    assert _calculate_max_drawdown([0.10, -0.05]) == pytest.approx(-0.05)
    assert _calculate_max_drawdown([]) == 0.0


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


def test_manifest_missing_all_three_identity_fields_rejected(tmp_path):
    csv_file = tmp_path / "ds_test.csv"
    manifest_file = tmp_path / "ds_test.manifest.json"

    df_sample = sample_daily_df_iso_strings()
    df_sample.to_csv(csv_file, index=False)

    import hashlib
    actual_sha = hashlib.sha256(csv_file.read_bytes()).hexdigest()

    manifest_data = {
        "manifest_version": "1.0",
        "output_sha256": actual_sha,
        "row_count": len(df_sample),
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    with pytest.raises(ValueError, match="missing required identity field"):
        verify_dataset_and_manifest_integrity(csv_file, manifest_file)


def test_manifest_identity_fields_missing_individually_rejected(tmp_path):
    csv_file = tmp_path / "ds_test.csv"
    manifest_file = tmp_path / "ds_test.manifest.json"

    df_sample = sample_daily_df_iso_strings()
    df_sample.to_csv(csv_file, index=False)

    import hashlib
    actual_sha = hashlib.sha256(csv_file.read_bytes()).hexdigest()

    base_manifest = {
        "manifest_version": "1.0",
        "dataset_id": "ds_test",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "output_sha256": actual_sha,
        "row_count": len(df_sample),
    }

    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity

    for field in ("dataset_id", "symbol", "timeframe"):
        bad_manifest = dict(base_manifest)
        del bad_manifest[field]
        manifest_file.write_text(json.dumps(bad_manifest), encoding="utf-8")

        with pytest.raises(ValueError, match=f"missing required identity field '{field}'"):
            verify_dataset_and_manifest_integrity(csv_file, manifest_file)


def test_manifest_identity_fields_null_blank_whitespace_rejected(tmp_path):
    csv_file = tmp_path / "ds_test.csv"
    manifest_file = tmp_path / "ds_test.manifest.json"

    df_sample = sample_daily_df_iso_strings()
    df_sample.to_csv(csv_file, index=False)

    import hashlib
    actual_sha = hashlib.sha256(csv_file.read_bytes()).hexdigest()

    base_manifest = {
        "manifest_version": "1.0",
        "dataset_id": "ds_test",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "output_sha256": actual_sha,
        "row_count": len(df_sample),
    }

    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity

    invalid_values = [None, 123, "", "   ", "\t\n"]

    for field in ("dataset_id", "symbol", "timeframe"):
        for inv in invalid_values:
            bad_manifest = dict(base_manifest)
            bad_manifest[field] = inv
            manifest_file.write_text(json.dumps(bad_manifest), encoding="utf-8")

            with pytest.raises((TypeError, ValueError)):
                verify_dataset_and_manifest_integrity(csv_file, manifest_file)


def test_incorrect_dataset_id_symbol_or_timeframe_rejected(tmp_path):
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
        "row_count": len(df_sample),
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    from src.evaluation.research_runner import DatasetScope, run_research_experiment, ResearchHypothesis, ExecutionAssumptions, CodeProvenance, WalkForwardProtocol, HypothesisStatus

    # Test via research experiment execution or dataset scope resolution with mismatched scope
    bad_scopes = [
        DatasetScope("wrong_id", "XAUUSD", "1D", "2025-01-01", "2025-01-10"),
        DatasetScope("ds_test", "EURUSD", "1D", "2025-01-01", "2025-01-10"),
        DatasetScope("ds_test", "XAUUSD", "1H", "2025-01-01", "2025-01-10"),
    ]

    for scope in bad_scopes:
        hyp = ResearchHypothesis(
            statement="Test scope mismatch",
            methodology_version="1.0",
            strategy_name="moving_average",
            strategy_version="1.0",
            dataset_scope=scope,
            execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
            code_provenance=CodeProvenance("sha123"),
            benchmark_reference="buy_and_hold",
            parameters={"fast_window": 2, "slow_window": 4},
            walk_forward_protocol=WalkForwardProtocol(4, 2),
            status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
        )

        with pytest.raises(ValueError):
            # _load_and_verify_csv or run_research_experiment with custom df_path raises ValueError
            from src.evaluation.research_runner import validate_and_prepare_dataset
            # Calling verify via research_runner
            from src.evaluation.research_runner import resolve_historical_dataset
            # Let's test _load_and_verify_csv directly or through path resolution mock
            from unittest.mock import patch
            with patch("src.evaluation.research_runner.Path.exists", return_value=True), \
                 patch("src.evaluation.research_runner.Path.is_file", return_value=True), \
                 patch("src.data.ingest_xauusd_15m.verify_dataset_and_manifest_integrity", return_value=(df_sample, manifest_data)):
                resolve_historical_dataset(scope)


def test_missing_manifest_on_csv_rejected(tmp_path):
    csv_file = tmp_path / "ds_no_manifest.csv"
    df_sample = sample_daily_df_iso_strings()
    df_sample.to_csv(csv_file, index=False)

    from src.evaluation.research_runner import DatasetScope, resolve_historical_dataset
    from unittest.mock import patch

    scope_match = DatasetScope("ds_no_manifest", "XAUUSD", "1D", "2025-01-01", "2025-01-10")
    scope_mismatch = DatasetScope("ds_no_manifest", "EURUSD", "1H", "2025-01-01", "2025-01-10")

    for scope in (scope_match, scope_mismatch):
        def mock_exists(self):
            # Return True for the CSV file itself so it is found, but False for the .manifest.json
            if self.name.endswith(".manifest.json"):
                return False
            return True

        with patch("src.evaluation.research_runner.Path.exists", autospec=True, side_effect=mock_exists), \
             patch("src.evaluation.research_runner.Path.is_file", return_value=True):
            with pytest.raises(FileNotFoundError, match="Dataset manifest missing for CSV artifact"):
                resolve_historical_dataset(scope)


def test_commit_marker_identity_mismatch_rejected(tmp_path):
    csv_file = tmp_path / "ds_test.csv"
    manifest_file = tmp_path / "ds_test.manifest.json"
    commit_file = tmp_path / "ds_test.commit.json"

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
        "row_count": len(df_sample),
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Commit marker with mismatched dataset_id
    commit_data = {
        "commit_version": "1.0",
        "dataset_id": "mismatched_id",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "output_sha256": actual_sha,
    }
    commit_file.write_text(json.dumps(commit_data), encoding="utf-8")

    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    with pytest.raises(ValueError, match="conflicts with manifest"):
        verify_dataset_and_manifest_integrity(csv_file, manifest_file, commit_file)


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
    with pytest.raises(ValueError, match="Backtesting with risk parameters requires OHLC columns"):
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
# 7. Canonical Timeframes Independence & End-to-End Proof Tests
# -----------------------------------------------------------------------------

def test_committed_render_daily_candidate_resolves_and_authorizes_from_checkout():
    """The exact Render-mapped candidate must resolve from committed repository artifacts."""
    from src.evaluation.research_store import resolve_promoted_candidate
    from src.evaluation.live_production_decision import authorize_production_runtime

    research_dir = Path("results/research_experiments")
    candidate_id = "cand_moving_average_1d_verified_20261020"
    promoted = resolve_promoted_candidate(
        candidate_id=candidate_id,
        symbol="XAUUSD",
        timeframe="1D",
        base_dir=research_dir,
    )
    assert promoted is not None
    assert promoted.candidate_id == candidate_id
    assert promoted.symbol == "XAUUSD"
    assert promoted.timeframe == "1D"
    assert not isinstance(promoted.operational_stability_score, bool)
    assert math.isfinite(float(promoted.operational_stability_score))
    authorization = authorize_production_runtime(
        promoted, symbol="XAUUSD", timeframe="1D"
    )
    assert authorization.candidate_id == candidate_id
    assert authorization.governance_decision_fingerprint


def test_e2e_1d_moving_average_signal_generation_and_publication(tmp_path):
    """E2E proof: manifest-verified fixture -> real research qualification -> authorization -> Contract v1 publisher."""
    import hashlib
    import json
    import numpy as np
    from datetime import datetime, timezone
    from unittest.mock import patch, MagicMock

    from src.evaluation.research_constitution import (
        ResearchHypothesis,
        ExecutionAssumptions,
        CodeProvenance,
        WalkForwardProtocol,
        HypothesisStatus,
    )
    from src.evaluation.research_runner import run_research_experiment, validate_and_prepare_dataset
    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.stability import CanonicalStabilityEvidence, calculate_stability_score
    from src.evaluation.research_store import persist_promoted_candidate_binding
    from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig
    from src.integration.project2_publisher import Project2Publisher

    res_dir = tmp_path / "research"
    store_path = tmp_path / "decision_history.json"
    snapshot_path = tmp_path / "live_snapshot.json"
    csv_path = tmp_path / "fixture_xauusd_daily.csv"
    manifest_path = tmp_path / "fixture_xauusd_daily.manifest.json"

    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D", tz="UTC")
    idx = np.arange(len(dates), dtype=float)
    close = 2000.0 * np.exp(0.0012 * idx + 0.001 * np.sin(idx * 0.07))
    open_prices = np.concatenate(([close[0] / np.exp(0.0012)], close[:-1]))
    fixture_df = pd.DataFrame({
        "timestamp": dates.astype(str),
        "open": open_prices,
        "high": np.maximum(open_prices, close) * 1.0015,
        "low": np.minimum(open_prices, close) * 0.9985,
        "close": close,
    })
    fixture_df.to_csv(csv_path, index=False)
    output_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps({
        "manifest_version": "1.0",
        "dataset_id": "fixture_xauusd_daily",
        "symbol": "XAUUSD",
        "timeframe": "1D",
        "output_sha256": output_sha,
        "row_count": len(fixture_df),
        "columns": list(fixture_df.columns),
        "timestamp_unit": "iso8601",
    }, indent=2), encoding="utf-8")

    ds = DatasetScope(
        dataset_id="fixture_xauusd_daily",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2024-01-01",
        end_date="2025-12-31",
    )
    raw_df, manifest = verify_dataset_and_manifest_integrity(csv_path, manifest_path)
    assert manifest["dataset_id"] == ds.dataset_id
    assert manifest["symbol"] == ds.symbol
    assert manifest["timeframe"] == ds.timeframe
    df = validate_and_prepare_dataset(raw_df, ds)

    hyp = ResearchHypothesis(
        statement="Moving Average 5/20 on manifest-verified 1D integration fixture",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("7b34bdf4f81d40abd4274ac41ebfe9f2b6222255"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 5, "slow_window": 20, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        walk_forward_protocol=WalkForwardProtocol(train_size=150, test_size=30),
        status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
    )
    evidence = run_research_experiment(hyp, df=df, persist_evidence=True, base_dir=res_dir)
    robustness = assess_research_robustness(evidence)
    assert evidence.promotion_status.value in {"PROMOTABLE", "VALIDATED"}, (
        f"Canonical research failed: status={evidence.promotion_status.value}; "
        f"rejections={[r.value for r in evidence.rejection_reasons]}; robustness={robustness!r}"
    )

    stability_inputs = pd.DataFrame([
        {
            "strategy": evidence.spec.strategy_name,
            "total_return": part.total_return,
            "max_drawdown": part.max_drawdown,
            "sharpe_ratio": part.sharpe_ratio,
            "positive_window_rate": 1.0 if part.total_return > 0 else 0.0,
        }
        for part in evidence.partitions
    ])
    stability_ranking = calculate_stability_score(stability_inputs)
    assert not stability_ranking.empty
    measured_stability = CanonicalStabilityEvidence(
        strategy_name=evidence.spec.strategy_name,
        stability_score=float(stability_ranking.iloc[0]["stability_score"]),
    )
    persist_promoted_candidate_binding(
        candidate_id="cand_moving_average_1d",
        evidence=evidence,
        canonical_stability=measured_stability,
        robustness_assessment=robustness,
        base_dir=res_dir,
    )

    ref_now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    publisher = Project2Publisher(
        publish_url="http://localhost:8000/api/v1/integration/project1/ingest",
        api_key="test_api_key",
        enabled=True,
        max_age_seconds=86400 * 30,
    )
    cfg = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="1D",
        candidate_id="cand_moving_average_1d",
        research_dir=res_dir,
    )
    delivered_payloads = []

    def mock_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        delivered_payloads.append(payload)
        pub_id = payload.get("event_id") or payload.get("publication_id")
        resp = MagicMock()
        resp.status = 200
        resp.getcode.return_value = 200
        resp.__enter__.return_value = resp
        resp.read.return_value = json.dumps({"status": "INGESTED", "event_id": pub_id}).encode("utf-8")
        return resp

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="1D",
        max_age_seconds=86400 * 30,
        publisher=publisher,
        production_config=cfg,
        store_path=store_path,
        snapshot_path=snapshot_path,
        research_dir=res_dir,
    )
    with patch("urllib.request.urlopen", side_effect=mock_urlopen) as publisher_transport:
        result = runtime.run_once(market_data=df, reference_now=ref_now)

    assert publisher_transport.call_count == 1
    assert len(delivered_payloads) == 1
    payload = delivered_payloads[0]
    assert payload["contract_version"] == "1.0"
    assert payload["instrument"]["symbol"] == "XAUUSD"
    assert payload["instrument"]["interval"] == "1D"
    assert payload["signal"]["decision"] == "BUY"
    assert payload["signal"]["candidate_id"] == "cand_moving_average_1d"
    assert result["blocked"] is False
    assert result["decision"] == "BUY"
    assert result["publish_result"]["published"] is True
    assert result["publish_result"]["status"] == "PUBLISHED"
    assert result["publish_result"]["delivery_status"] == "DELIVERED"
    assert store_path.exists()

def test_unauthorized_runtime_publishes_nothing(tmp_path):
    """Negative Control: Unauthorized/unpromoted timeframe publishes nothing."""
    from datetime import datetime, timezone
    from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig
    from src.integration.project2_publisher import Project2Publisher

    store_path = tmp_path / "decision_history.json"
    snapshot_path = tmp_path / "live_snapshot.json"

    df_5m = sample_daily_df_iso_strings()

    publisher = Project2Publisher(
        publish_url="http://localhost:8000/api/v1/integration/project1/ingest",
        api_key="test_api_key",
        enabled=True,
    )
    cfg = ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m")

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=publisher,
        production_config=cfg,
        store_path=store_path,
        snapshot_path=snapshot_path,
    )

    res = runtime.run_once(market_data=df_5m)

    assert res["blocked"] is True
    assert res["publish_result"]["published"] is False
    assert res["publish_result"]["status"] == "SKIPPED_BLOCKED"


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

def test_real_manifest_backed_15m_research_qualification(tmp_path):
    """Evaluate real repository data using canonical research and unchanged policy."""
    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    from src.evaluation.research_constitution import (
        ResearchHypothesis,
        ExecutionAssumptions,
        CodeProvenance,
        WalkForwardProtocol,
        HypothesisStatus,
    )
    from src.evaluation.research_runner import run_research_experiment, validate_and_prepare_dataset
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.research_qualification import qualify_research_evidence

    data_dir = Path("data/raw/xauusd_15m")
    raw_df, manifest = verify_dataset_and_manifest_integrity(
        data_dir / "ds_moving_average_15m.csv",
        data_dir / "ds_moving_average_15m.manifest.json",
        data_dir / "ds_moving_average_15m.commit.json",
    )
    scope = DatasetScope(
        dataset_id=manifest["dataset_id"],
        symbol=manifest["symbol"],
        timeframe=manifest["timeframe"],
        start_date=manifest["start_timestamp_utc"][:10],
        end_date=manifest["end_timestamp_utc"][:10],
    )
    df = validate_and_prepare_dataset(raw_df, scope)
    hypothesis = ResearchHypothesis(
        statement="Moving Average 20/50 on actual manifest-backed XAUUSD 15m data",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=scope,
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("7b34bdf4f81d40abd4274ac41ebfe9f2b6222255"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 20, "slow_window": 50, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
        walk_forward_protocol=WalkForwardProtocol(train_size=1000, test_size=250),
        status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
    )
    evidence = run_research_experiment(hypothesis, df=df, persist_evidence=True, base_dir=tmp_path / "research")
    robustness = assess_research_robustness(evidence)
    qualification = qualify_research_evidence(evidence, robustness_assessment=robustness)
    # The checked-in 15m sample currently fails real robustness criteria.
    # Keep this as a fail-closed regression: never manufacture promotion for it.
    assert not qualification.qualified
    assert "FAILED_PARAMETER_SENSITIVITY" in {r.value for r in qualification.rejection_reasons}
    assert "FAILED_STATISTICAL_VALIDATION" in {r.value for r in qualification.rejection_reasons}

def test_real_manifest_backed_daily_candidate_qualifies_without_policy_relaxation(tmp_path):
    """Qualify real 2025 data, create a genuine binding, preflight, authorize, and publish."""
    import numpy as np
    from datetime import datetime, timezone
    from unittest.mock import patch, MagicMock
    from src.data.ingest_xauusd_15m import verify_dataset_and_manifest_integrity
    from src.evaluation.research_constitution import (
        ResearchHypothesis,
        ExecutionAssumptions,
        CodeProvenance,
        WalkForwardProtocol,
        HypothesisStatus,
    )
    from src.evaluation.research_runner import run_research_experiment, validate_and_prepare_dataset
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.stability import CanonicalStabilityEvidence, calculate_stability_score
    from src.evaluation.research_store import (
        save_research_experiment,
        persist_promoted_candidate_binding,
        resolve_promoted_candidate,
    )
    from src.evaluation.live_production_decision import authorize_production_runtime
    from src.evaluation.live_execution_runtime import (
        ContinuousLiveRuntime,
        LiveExecutionRuntime,
        ProductionRuntimeConfig,
        verify_timeframe_production_readiness,
    )
    from src.integration.project2_publisher import Project2Publisher

    csv_path = Path("data/raw/xauusd_daily_2025.csv")
    manifest_path = Path("data/raw/xauusd_daily_2025.manifest.json")
    raw_df, manifest = verify_dataset_and_manifest_integrity(csv_path, manifest_path)
    scope = DatasetScope(
        dataset_id=manifest["dataset_id"],
        symbol=manifest["symbol"],
        timeframe=manifest["timeframe"],
        start_date=manifest["start_timestamp_utc"][:10],
        end_date=manifest["end_timestamp_utc"][:10],
    )
    df = validate_and_prepare_dataset(raw_df, scope, timestamp_unit=manifest.get("timestamp_unit"))
    hypothesis = ResearchHypothesis(
        statement="Moving Average 20/50 with explicit wider risk barriers on verified XAUUSD daily data",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=scope,
        execution_assumptions=ExecutionAssumptions(0.0001, 0.0001, 10.0),
        code_provenance=CodeProvenance("fe1ce94340a2bba95a81f89045e3557fee2e88f5"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 20, "slow_window": 50, "stop_loss_pct": 0.05, "take_profit_pct": 0.10},
        walk_forward_protocol=WalkForwardProtocol(train_size=150, test_size=30),
        status=HypothesisStatus.ACCEPTED_FOR_RESEARCH,
    )
    evidence = run_research_experiment(
        hypothesis,
        df=df,
        persist_evidence=True,
        base_dir=tmp_path / "research",
    )
    robustness = assess_research_robustness(evidence)
    qualification = qualify_research_evidence(evidence, robustness_assessment=robustness)
    assert qualification.qualified, (
        f"Daily candidate did not qualify under unchanged production policy: "
        f"status={evidence.promotion_status.value}; "
        f"rejections={[r.value for r in evidence.rejection_reasons]}; "
        f"notes={qualification.qualification_notes}; robustness={robustness!r}"
    )

    candidate_id = "cand_moving_average_1d_verified_20261020"
    research_dir = Path("results/research_experiments")
    binding_path = research_dir / "by_candidate" / candidate_id / "candidate.json"
    # Persist the same canonical evidence/binding that passed qualification.
    save_research_experiment(evidence, base_dir=research_dir)
    stability_inputs = pd.DataFrame([
        {
            "strategy": evidence.spec.strategy_name,
            "total_return": part.total_return,
            "max_drawdown": part.max_drawdown,
            "sharpe_ratio": part.sharpe_ratio,
            "positive_window_rate": 1.0 if part.total_return > 0 else 0.0,
        }
        for part in evidence.partitions
    ])
    stability_ranking = calculate_stability_score(stability_inputs)
    assert not stability_ranking.empty
    measured_stability = CanonicalStabilityEvidence(
        strategy_name=evidence.spec.strategy_name,
        stability_score=float(stability_ranking.iloc[0]["stability_score"]),
    )
    persist_promoted_candidate_binding(
        candidate_id=candidate_id,
        evidence=evidence,
        canonical_stability=measured_stability,
        robustness_assessment=robustness,
        base_dir=research_dir,
    )

    # Resolve the persisted binding and execute the real authorization gate.
    promoted = resolve_promoted_candidate(
        candidate_id=candidate_id,
        symbol="XAUUSD",
        timeframe="1D",
        base_dir=research_dir,
    )
    assert promoted is not None
    authorization = authorize_production_runtime(promoted, symbol="XAUUSD", timeframe="1D")
    assert authorization.authorization_fingerprint
    assert not isinstance(authorization.operational_stability_score, bool)
    assert math.isfinite(float(authorization.operational_stability_score))

    now = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    max_age_seconds = 3 * 24 * 60 * 60  # Three days accommodates a completed daily candle in this integration fixture.
    publisher = Project2Publisher(
        publish_url="http://localhost:8000/api/v1/integration/project1/ingest",
        api_key="test_api_key",
        enabled=True,
        max_age_seconds=max_age_seconds,
    )
    readiness = verify_timeframe_production_readiness(
        symbol="XAUUSD",
        timeframe="1D",
        candidate_id=candidate_id,
        research_dir=research_dir,
        publisher=publisher,
        reference_now=now,
        check_provider_data=False,
    )
    assert readiness["status"] == "READY", f"Persisted daily candidate not READY: {readiness}"

    continuous = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1D"],
        poll_interval=0,
        publish=True,
        persist=False,
        max_age_seconds=max_age_seconds,
        research_dir=research_dir,
        store_path=tmp_path / "continuous_decisions.json",
        snapshot_path=tmp_path / "continuous_snapshot.json",
        publisher=publisher,
        candidate_ids={"1D": candidate_id},
        market_data_loaders={"1D": lambda: runtime_market_data},
        clock=lambda: now,
        sleep_fn=lambda _: None,
    )
    startup = continuous.verify_startup_readiness()
    assert startup["1D"]["status"] == "READY", f"Continuous startup preflight failed: {startup}"

    cfg = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="1D",
        candidate_id=candidate_id,
        research_dir=research_dir,
    )
    delivered_payloads = []

    def mock_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        delivered_payloads.append(payload)
        pub_id = payload.get("event_id") or payload.get("publication_id")
        resp = MagicMock()
        resp.status = 200
        resp.getcode.return_value = 200
        resp.__enter__.return_value = resp
        resp.read.return_value = json.dumps({"status": "INGESTED", "event_id": pub_id}).encode("utf-8")
        return resp

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="1D",
        max_age_seconds=max_age_seconds,
        publisher=publisher,
        production_config=cfg,
        store_path=tmp_path / "daily_decisions.json",
        snapshot_path=tmp_path / "daily_snapshot.json",
        research_dir=research_dir,
    )
    # The research evidence above comes from the real manifest-backed 2025
    # dataset. Use a separate deterministic, completed daily candle fixture
    # only for the live evaluation because the 2025 dataset correctly emits NO TRADE.
    runtime_dates = pd.date_range(
        end=(pd.Timestamp(now).normalize() - pd.Timedelta(days=1)),
        periods=100,
        freq="D",
    )
    runtime_idx = np.arange(len(runtime_dates), dtype=float)
    runtime_close = 2200.0 * np.exp(0.002 * runtime_idx + 0.0005 * np.sin(runtime_idx * 0.1))
    runtime_open = np.concatenate(([runtime_close[0] / np.exp(0.002)], runtime_close[:-1]))
    runtime_market_data = pd.DataFrame({
        "timestamp": runtime_dates,
        "open": runtime_open,
        "high": np.maximum(runtime_open, runtime_close) * 1.0015,
        "low": np.minimum(runtime_open, runtime_close) * 0.9985,
        "close": runtime_close,
    })
    with patch("urllib.request.urlopen", side_effect=mock_urlopen) as transport:
        result = runtime.run_once(market_data=runtime_market_data, reference_now=now, publish=True)

    assert transport.call_count == 1
    assert len(delivered_payloads) == 1
    payload = delivered_payloads[0]
    assert payload["contract_version"] == "1.0"
    assert payload["instrument"]["symbol"] == "XAUUSD"
    assert payload["instrument"]["interval"] == "1D"
    assert payload["signal"]["decision"] == "BUY"
    assert payload["signal"]["candidate_id"] == candidate_id
    assert result["blocked"] is False
    assert result["decision"] == "BUY"
    assert result["publish_result"]["published"] is True
    assert result["publish_result"]["status"] == "PUBLISHED"
    assert result["publish_result"]["delivery_status"] == "DELIVERED"
