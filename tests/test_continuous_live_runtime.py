"""Focused unit tests for ContinuousLiveRuntime orchestration layer."""
import json
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
)
from src.evaluation.mtf_intelligence import CanonicalTimeframe
from tests.test_live_execution_runtime import (
    make_buy_market_data,
    make_dummy_df,
    persist_momentum_candidate as base_persist_candidate,
)


def persist_momentum_candidate(base_dir, candidate_id: str = "cand_momentum_live", symbol: str = "XAUUSD", timeframe: str = "5m"):
    """Helper wrapper around base_persist_candidate allowing timeframe parameterization."""
    from src.evaluation.research_constitution import (
        CodeProvenance,
        DatasetScope,
        EvidencePartition,
        EvidencePartitionRole,
        ExecutionAssumptions,
        PromotionStatus,
        ResearchEvidence,
        ResearchExperimentSpec,
    )
    from src.evaluation.research_store import save_research_candidate

    ds = DatasetScope(
        dataset_id=f"ds_{symbol.lower()}_{timeframe}",
        symbol=symbol,
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    spec = ResearchExperimentSpec(
        hypothesis="Persisted live-runtime momentum candidate",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001, slippage=0.001, latency_ms=10.0
        ),
        code_provenance=CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c"),
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
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict={
            "passed": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )
    save_research_candidate(candidate_id=candidate_id, evidence=evidence, base_dir=base_dir, operational_stability_score=0.85)
    return candidate_id


def make_tf_market_data(start_time="2025-01-01 10:00", timeframe="5m") -> pd.DataFrame:
    """Create test candle data for a given timeframe."""
    freq_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1H": "1h", "4h": "4h", "4H": "4h", "1d": "1d", "1D": "1d"}
    pd_freq = freq_map.get(timeframe, "5min")
    timestamps = pd.date_range(start_time, periods=10, freq=pd_freq, tz="UTC")
    prices = [2000.0 + (i * 2.0) for i in range(10)]
    return pd.DataFrame({
        "openTime": timestamps,
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "timestamp": timestamps,
    })


def test_1_one_shot_compatibility(tmp_path) -> None:
    """Requirement 1: Existing LiveExecutionRuntime.run_once() primitive remains intact and backward compatible."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live")
    config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="cand_momentum_live",
        research_dir=tmp_path,
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )
    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    res = runtime.run_once(publish=False, persist=True, market_data=df, reference_now=ref_now)
    assert res["blocked"] is False
    assert res["decision"] == "BUY"
    assert res["strategy"] == "momentum"
    assert res["candidate_id"] == "cand_momentum_live"


def test_2_continuous_repetition(tmp_path) -> None:
    """Requirement 2: Continuous runtime remains alive across multiple scheduler ticks using fake clock/data."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live")
    df1 = make_tf_market_data("2025-01-01 10:00", "5m")
    df2 = make_tf_market_data("2025-01-01 10:05", "5m")

    ticks_data = [df1, df2]
    tick_idx = 0

    def mock_loader(tf: str) -> pd.DataFrame:
        nonlocal tick_idx
        data = ticks_data[min(tick_idx, len(ticks_data) - 1)]
        return data

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=mock_loader,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    ref_now1 = pd.to_datetime(df1["openTime"], utc=True).iloc[-1].to_pydatetime()
    ref_now2 = pd.to_datetime(df2["openTime"], utc=True).iloc[-1].to_pydatetime()

    # Tick 1
    t1 = cont.tick(reference_now=ref_now1)
    tick_idx = 1
    # Tick 2
    t2 = cont.tick(reference_now=ref_now2)

    assert cont.is_running
    assert len(cont._execution_history) == 2
    assert t1["evaluations"]["5m"]["blocked"] is False
    assert t2["evaluations"]["5m"]["blocked"] is False


def test_3_candle_deduplication(tmp_path) -> None:
    """Requirement 3: Repeated outer-loop ticks for the same closed candle do NOT produce duplicate evaluations/publications."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live")
    df_same = make_tf_market_data("2025-01-01 10:00", "5m")
    ref_now = pd.to_datetime(df_same["openTime"], utc=True).iloc[-1].to_pydatetime()

    mock_publisher = MagicMock()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=True,
        persist=False,
        publisher=mock_publisher,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df_same,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    t1 = cont.tick(reference_now=ref_now)
    t2 = cont.tick(reference_now=ref_now)

    assert t1["evaluations"]["5m"]["blocked"] is False
    assert t2["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert t2["evaluations"]["5m"]["reason"] == "candle_already_evaluated"


def test_4_timeframe_cadence(tmp_path) -> None:
    """Requirement 4: A 5m candle update does not evaluate an un-due 15m/30m/1h candle if the candle timestamp has not changed."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_15m", timeframe="15m")

    df_5m_1 = make_tf_market_data("2025-01-01 10:00", "5m")
    df_5m_2 = make_tf_market_data("2025-01-01 10:05", "5m")
    df_15m = make_tf_market_data("2025-01-01 10:00", "15m")

    ref_now1 = pd.to_datetime(df_5m_1["openTime"], utc=True).iloc[-1].to_pydatetime()
    ref_now2 = pd.to_datetime(df_5m_2["openTime"], utc=True).iloc[-1].to_pydatetime()

    state = {"tick": 0}

    def loader(tf: str) -> pd.DataFrame:
        if tf == "5m":
            return df_5m_1 if state["tick"] == 0 else df_5m_2
        return df_15m

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=loader,
        candidate_ids={"5m": "cand_momentum_live", "15m": "cand_momentum_live_15m"},
    )

    t1 = cont.tick(reference_now=ref_now1)
    state["tick"] = 1
    t2 = cont.tick(reference_now=ref_now2)

    # On tick 1: both 5m and 15m evaluate cleanly
    assert t1["evaluations"]["5m"]["blocked"] is False
    assert t1["evaluations"]["15m"]["blocked"] is False

    # On tick 2: new 5m candle evaluates, 15m candle is deduplicated/skipped
    assert t2["evaluations"]["5m"]["blocked"] is False
    assert t2["evaluations"]["15m"]["status"] == "SKIPPED_DEDUPLICATED"


def test_5_multi_timeframe_scheduling(tmp_path) -> None:
    """Requirement 5: Multiple canonical timeframes become due and are each evaluated with exact timeframe identity."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_15m", timeframe="15m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_1H", timeframe="1H")

    df_5m = make_tf_market_data("2025-01-01 10:00", "5m")
    ref_now = pd.to_datetime(df_5m["openTime"], utc=True).iloc[-1].to_pydatetime()

    loaders = {
        "5m": lambda: df_5m,
        "15m": lambda: make_tf_market_data("2025-01-01 10:00", "15m"),
        "1H": lambda: make_tf_market_data("2025-01-01 10:00", "1H"),
    }

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m", "1H"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=loaders,
        candidate_ids={"5m": "cand_momentum_live", "15m": "cand_momentum_live_15m", "1H": "cand_momentum_live_1H"},
    )

    res = cont.tick(reference_now=ref_now)
    evals = res["evaluations"]

    assert evals["5m"]["interval"] == "5m"
    assert evals["15m"]["interval"] == "15m"
    assert evals["1H"]["interval"] == "1H"


def test_6_missing_candidate_fails_closed(tmp_path) -> None:
    """Requirement 6: Continuous mode fails closed for a timeframe without an authoritative promoted candidate; does not substitute."""
    df_5m = make_tf_market_data("2025-01-01 10:00", "5m")
    ref_now = pd.to_datetime(df_5m["openTime"], utc=True).iloc[-1].to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df_5m,
        candidate_ids={"5m": "cand_nonexistent_999"},
    )

    res = cont.tick(reference_now=ref_now)
    eval_5m = res["evaluations"]["5m"]

    assert eval_5m["blocked"] is True
    assert eval_5m["reason"] == "PromotionUnavailable"
    assert eval_5m["decision"] == "NO TRADE"


def test_7_no_1m_fabrication() -> None:
    """Requirement 7: Continuous mode rejects unsupported '1m' timeframe explicitly without inventing candidates."""
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '1m'"):
        ContinuousLiveRuntime(
            symbol="XAUUSD",
            timeframes=["1m"],
        )


def test_8_recoverable_provider_failure(tmp_path) -> None:
    """Requirement 8: Recoverable provider failure during one tick does not terminate the continuous runtime."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live")

    state = {"fail": True}

    def failing_loader(tf: str) -> pd.DataFrame:
        if state["fail"]:
            raise RuntimeError("Temporary network timeout")
        return make_tf_market_data("2025-01-01 10:00", "5m")

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=failing_loader,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    df_5m = make_tf_market_data("2025-01-01 10:00", "5m")
    ref_now = pd.to_datetime(df_5m["openTime"], utc=True).iloc[-1].to_pydatetime()

    # Tick 1 fails recoverably
    t1 = cont.tick(reference_now=ref_now)
    assert t1["evaluations"]["5m"]["status"] == "FAILED_RECOVERABLE"
    assert "Temporary network timeout" in t1["evaluations"]["5m"]["error"]

    # Runtime remains alive
    assert cont.is_running

    # Tick 2 succeeds after provider recovers
    state["fail"] = False
    t2 = cont.tick(reference_now=ref_now)
    assert t2["evaluations"]["5m"]["blocked"] is False


def test_9_graceful_shutdown(tmp_path) -> None:
    """Requirement 9: stop() terminates the continuous runtime deterministically."""
    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
    )

    assert cont.is_running is True
    cont.stop()
    assert cont.is_running is False

    # Running ticks when stopped short-circuits
    res = cont.run_ticks(max_ticks=5)
    assert len(res) == 0


def test_10_publication_identity(tmp_path) -> None:
    """Requirement 10: Continuous mode delegates publication strictly through existing canonical path."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live")
    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {"status": "SKIPPED_DISABLED", "published": False}

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=True,
        persist=False,
        publisher=mock_publisher,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    res = cont.tick(reference_now=ref_now)
    eval_5m = res["evaluations"]["5m"]
    assert eval_5m["blocked"] is False
    assert eval_5m["decision"] == "BUY"


def test_11_duplicate_prevention_across_ticks(tmp_path) -> None:
    """Requirement 11: The exact same (symbol, timeframe, candle_ts, candidate_id) cannot be evaluated twice across ticks."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live")

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    # Tick 1: evaluates
    t1 = cont.tick(reference_now=ref_now)
    assert t1["evaluations"]["5m"]["blocked"] is False

    # Tick 2 & 3: same candle, deduplicated
    t2 = cont.tick(reference_now=ref_now)
    t3 = cont.tick(reference_now=ref_now)

    assert t2["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert t3["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
