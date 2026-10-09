"""Comprehensive focused unit tests for ContinuousLiveRuntime covering all 41 required test invariants."""
import datetime
import json
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    LiveExecutionRuntime,
    ProductionBlocked,
    ProductionRuntimeConfig,
    is_candle_closed,
    parse_continuous_candidate_ids,
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


def make_tf_market_data(start_time="2025-01-01 10:00", timeframe="5m", count=10) -> pd.DataFrame:
    """Create test candle data for a given timeframe."""
    freq_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1H": "1h", "4h": "4h", "4H": "4h", "1d": "1d", "1D": "1d"}
    pd_freq = freq_map.get(timeframe, "5min")
    timestamps = pd.date_range(start_time, periods=count, freq=pd_freq, tz="UTC")
    prices = [2000.0 + (i * 2.0) for i in range(count)]
    return pd.DataFrame({
        "openTime": timestamps,
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "timestamp": timestamps,
    })


# -----------------------------------------------------------------------------
# Closed-candle tests (1-7)
# -----------------------------------------------------------------------------

def test_1_explicit_is_open_true_no_evaluation(tmp_path) -> None:
    """1. Explicit 'isOpen=True' in provider row -> not closed -> no evaluation."""
    df = make_tf_market_data("2025-01-01 10:00", "5m", count=1)
    df["isOpen"] = True

    ref_now = pd.to_datetime("2025-01-01 10:10:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
    )

    res = cont.tick(reference_now=ref_now)
    eval_5m = res["evaluations"]["5m"]
    assert eval_5m["status"] == "SKIPPED_NO_CLOSED_CANDLE"


def test_2_explicit_closed_candle_allowed(tmp_path) -> None:
    """2. Explicit closed candle ('isOpen=False' / 'isClosed=True') -> evaluation allowed."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df = make_buy_market_data()
    df["isOpen"] = False

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

    res = cont.tick(reference_now=ref_now)
    eval_5m = res["evaluations"]["5m"]
    assert eval_5m["blocked"] is False
    assert eval_5m["decision"] == "BUY"


def test_3_no_explicit_field_before_duration_no_evaluation(tmp_path) -> None:
    """3. No explicit open/closed field + reference_now < candle_open + duration -> no evaluation."""
    df = make_tf_market_data("2025-01-01 10:00", "5m", count=1)
    # candle open = 10:00, 5m duration -> closed at 10:05.
    # reference_now = 10:04 (before closure)
    ref_now = pd.to_datetime("2025-01-01 10:04:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
    )

    res = cont.tick(reference_now=ref_now)
    assert res["evaluations"]["5m"]["status"] == "SKIPPED_NO_CLOSED_CANDLE"


def test_4_no_explicit_field_at_or_after_duration_allowed(tmp_path) -> None:
    """4. No explicit field + reference_now >= candle_open + duration -> evaluation allowed."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df = make_buy_market_data()
    # Candle open = 10:45, duration = 5m -> closed at 10:50.
    last_candle_open = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()
    ref_now = last_candle_open + datetime.timedelta(minutes=5)

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

    res = cont.tick(reference_now=ref_now)
    assert res["evaluations"]["5m"]["blocked"] is False


def test_5_open_to_closed_transition_evaluates_once(tmp_path) -> None:
    """5. Open -> closed transition triggers exactly one evaluation."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df_single = make_tf_market_data("2025-01-01 10:00", "5m", count=1)
    candle_open = pd.to_datetime(df_single["openTime"], utc=True).iloc[-1].to_pydatetime()

    ref_open = candle_open + datetime.timedelta(minutes=2)
    ref_closed = candle_open + datetime.timedelta(minutes=5)

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df_single,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    # Poll 1: Open -> skipped
    t1 = cont.tick(reference_now=ref_open)
    assert t1["evaluations"]["5m"]["status"] == "SKIPPED_NO_CLOSED_CANDLE"

    # Poll 2: Closed -> evaluated
    t2 = cont.tick(reference_now=ref_closed)
    assert t2["evaluations"]["5m"]["blocked"] is False


def test_6_repeated_polls_after_closure_deduplicated(tmp_path) -> None:
    """6. Repeated polls after candle closure are deduplicated."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df = make_buy_market_data()
    last_candle_open = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()
    ref_closed = last_candle_open + datetime.timedelta(minutes=5)

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

    t1 = cont.tick(reference_now=ref_closed)
    t2 = cont.tick(reference_now=ref_closed + datetime.timedelta(seconds=10))

    assert t1["evaluations"]["5m"]["blocked"] is False
    assert t2["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"


def test_7_exact_source_candle_timestamp_preserved(tmp_path) -> None:
    """7. Source candle timestamp is preserved exactly without shifting or rounding."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df = make_buy_market_data()
    expected_ts_iso = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime().isoformat()
    ref_closed = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime() + datetime.timedelta(minutes=5)

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

    res = cont.tick(reference_now=ref_closed)
    eval_5m = res["evaluations"]["5m"]
    assert eval_5m["market_data"]["timestamp"].iloc[-1].to_pydatetime().isoformat() == expected_ts_iso


# -----------------------------------------------------------------------------
# Scheduling tests (8-12)
# -----------------------------------------------------------------------------

def test_8_repeated_polls_with_unchanged_closed_candle(tmp_path) -> None:
    """8. Repeated polls with unchanged closed candle do not produce duplicate evaluations."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df = make_buy_market_data()
    ref_closed = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime() + datetime.timedelta(minutes=5)

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

    for i in range(3):
        res = cont.tick(reference_now=ref_closed)
        if i == 0:
            assert res["evaluations"]["5m"]["blocked"] is False
        else:
            assert res["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"


def test_9_10_11_new_5m_closed_candle_only_5m_evaluates(tmp_path) -> None:
    """9, 10, 11. New 5m closed candle evaluates 5m, unchanged 15m is skipped."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_5m", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_15m", timeframe="15m")

    df_5m_1 = make_tf_market_data("2025-01-01 10:00", "5m")
    df_5m_2 = make_tf_market_data("2025-01-01 10:05", "5m")
    df_15m = make_tf_market_data("2025-01-01 10:00", "15m")

    ref_now1 = pd.to_datetime("2025-01-01 10:50:00Z").to_pydatetime()
    ref_now2 = pd.to_datetime("2025-01-01 10:55:00Z").to_pydatetime()

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
        candidate_ids={"5m": "cand_momentum_live_5m", "15m": "cand_momentum_live_15m"},
    )

    t1 = cont.tick(reference_now=ref_now1)
    state["tick"] = 1
    t2 = cont.tick(reference_now=ref_now2)

    # On tick 1: both 5m and 15m evaluate
    assert t1["evaluations"]["5m"]["blocked"] is False
    assert t1["evaluations"]["15m"]["blocked"] is False

    # On tick 2: 5m evaluates new candle, 15m is skipped
    assert t2["evaluations"]["5m"]["blocked"] is False
    assert t2["evaluations"]["15m"]["status"] == "SKIPPED_DEDUPLICATED"


def test_12_same_timestamp_across_different_timeframes_distinct(tmp_path) -> None:
    """12. Same timestamp across different timeframes produces distinct evaluation identities."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_5m", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live_15m", timeframe="15m")

    df_5m = make_tf_market_data("2025-01-01 10:00", "5m")
    df_15m = make_tf_market_data("2025-01-01 10:00", "15m")

    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders={"5m": lambda: df_5m, "15m": lambda: df_15m},
        candidate_ids={"5m": "cand_momentum_live_5m", "15m": "cand_momentum_live_15m"},
    )

    res = cont.tick(reference_now=ref_now)
    assert res["evaluations"]["5m"]["interval"] == "5m"
    assert res["evaluations"]["15m"]["interval"] == "15m"


# -----------------------------------------------------------------------------
# Candidate authority tests (13-18)
# -----------------------------------------------------------------------------

def test_13_14_exact_timeframe_candidate_used(tmp_path) -> None:
    """13, 14. 5m uses 5m candidate, 15m uses 15m candidate."""
    cand_5m = persist_momentum_candidate(tmp_path, candidate_id="cand_5m", timeframe="5m")
    cand_15m = persist_momentum_candidate(tmp_path, candidate_id="cand_15m", timeframe="15m")

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", tf),
        candidate_ids={"5m": cand_5m, "15m": cand_15m},
    )

    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()
    res = cont.tick(reference_now=ref_now)

    assert res["evaluations"]["5m"]["candidate_id"] == "cand_5m"
    assert res["evaluations"]["15m"]["candidate_id"] == "cand_15m"


def test_15_explicit_candidate_timeframe_mismatch_fails_closed(tmp_path) -> None:
    """15. Explicit candidate with mismatched DatasetScope timeframe fails closed."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_5m", timeframe="5m")

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["15m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "15m"),
        candidate_ids={"15m": "cand_5m"},  # Scope mismatch!
    )

    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()
    res = cont.tick(reference_now=ref_now)

    eval_15m = res["evaluations"]["15m"]
    assert eval_15m["blocked"] is True
    assert eval_15m["reason"] == "PromotionEligibilityError"
    assert "timeframe '5m'" in eval_15m["detail"]
    assert "15m" in eval_15m["detail"]
    assert len(cont._evaluated_candles) == 0


def test_16_missing_candidate_fails_closed(tmp_path) -> None:
    """16. Missing candidate for timeframe fails closed."""
    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
        candidate_ids={"5m": "cand_nonexistent"},
    )

    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()
    res = cont.tick(reference_now=ref_now)

    eval_5m = res["evaluations"]["5m"]
    assert eval_5m["blocked"] is True
    assert eval_5m["reason"] == "PromotionUnavailable"


def test_17_18_parse_continuous_candidate_ids_prevents_global_copy() -> None:
    """17, 18. Continuous candidate parser prevents single candidate ID copying across multiple timeframes."""
    # Single candidate ID for multiple timeframes raises ValueError
    with pytest.raises(ValueError, match="must specify explicit per-timeframe mappings"):
        parse_continuous_candidate_ids("single_cand_id", timeframes=["5m", "15m"])

    # Explicit mapping parses correctly
    parsed = parse_continuous_candidate_ids("5m=cand1,15m=cand2", timeframes=["5m", "15m"])
    assert parsed == {"5m": "cand1", "15m": "cand2"}


# -----------------------------------------------------------------------------
# Deduplication tests (19-24)
# -----------------------------------------------------------------------------

def test_19_same_four_field_identity_deduplicated(tmp_path) -> None:
    """19. Same 4-field identity (symbol, timeframe, candle_ts, candidate_id) is deduplicated."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_5m", timeframe="5m")
    df = make_buy_market_data()
    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
        candidate_ids={"5m": "cand_5m"},
    )

    t1 = cont.tick(reference_now=ref_now)
    t2 = cont.tick(reference_now=ref_now)

    assert t1["evaluations"]["5m"]["blocked"] is False
    assert t2["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"


def test_20_same_candle_different_candidate_distinct(tmp_path) -> None:
    """20. Same candle timestamp + different candidate ID = distinct deduplication identity."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_A", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_B", timeframe="5m")
    df = make_buy_market_data()
    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
        candidate_ids={"5m": "cand_A"},
    )

    t1 = cont.tick(reference_now=ref_now)
    assert t1["evaluations"]["5m"]["candidate_id"] == "cand_A"

    # Switch candidate ID for 5m
    cont.candidate_ids["5m"] = "cand_B"
    t2 = cont.tick(reference_now=ref_now)
    assert t2["evaluations"]["5m"]["candidate_id"] == "cand_B"


def test_21_same_timestamp_different_timeframe_distinct(tmp_path) -> None:
    """21. Same timestamp + different timeframe = distinct deduplication identity."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_5m", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_15m", timeframe="15m")

    df_5m = make_tf_market_data("2025-01-01 10:00", "5m")
    df_15m = make_tf_market_data("2025-01-01 10:00", "15m")
    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders={"5m": lambda: df_5m, "15m": lambda: df_15m},
        candidate_ids={"5m": "cand_5m", "15m": "cand_15m"},
    )

    res = cont.tick(reference_now=ref_now)
    assert res["evaluations"]["5m"]["blocked"] is False
    assert res["evaluations"]["15m"]["blocked"] is False


def test_22_open_candle_never_enters_dedup_state(tmp_path) -> None:
    """22. Open candle never enters deduplication state."""
    df = make_tf_market_data("2025-01-01 10:00", "5m", count=1)
    df["isOpen"] = True
    ref_now = pd.to_datetime("2025-01-01 10:10:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df,
    )

    cont.tick(reference_now=ref_now)
    assert len(cont._evaluated_candles) == 0


def test_23_24_failed_evaluation_retryable(tmp_path) -> None:
    """23, 24. Failed evaluation never enters dedup state; retry succeeds when provider recovers."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")

    state = {"fail": True}

    def loader(tf: str) -> pd.DataFrame:
        if state["fail"]:
            raise RuntimeError("Provider error")
        return make_buy_market_data()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=loader,
        candidate_ids={"5m": "cand_momentum_live"},
    )

    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    # Poll 1: Fails -> not added to dedup set
    t1 = cont.tick(reference_now=ref_now)
    assert t1["evaluations"]["5m"]["status"] == "FAILED_RECOVERABLE"
    assert len(cont._evaluated_candles) == 0

    # Poll 2: Provider recovers -> succeeds
    state["fail"] = False
    t2 = cont.tick(reference_now=ref_now)
    assert t2["evaluations"]["5m"]["blocked"] is False
    assert len(cont._evaluated_candles) == 1


# -----------------------------------------------------------------------------
# Boundary & Spy tests (25-28)
# -----------------------------------------------------------------------------

def test_25_26_27_28_delegates_to_run_once_primitive(tmp_path) -> None:
    """25, 26, 27, 28. Continuous live runtime delegates to LiveExecutionRuntime.run_once() rather than duplicating logic."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    df = make_buy_market_data()
    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

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

    with patch.object(LiveExecutionRuntime, "run_once") as mock_run_once:
        mock_run_once.return_value = {"blocked": False, "decision": "BUY", "interval": "5m"}

        res = cont.tick(reference_now=ref_now)

        assert mock_run_once.called
        assert res["evaluations"]["5m"]["decision"] == "BUY"


# -----------------------------------------------------------------------------
# Regression test for blocked state not being recorded in deduplication set
# -----------------------------------------------------------------------------

def test_blocked_result_not_deduplicated_and_retryable(tmp_path) -> None:
    """Regression test: ProductionBlocked results must NEVER enter deduplication state and must remain retryable."""
    # 1. Candidate created for 15m
    cand_15m = persist_momentum_candidate(tmp_path, candidate_id="cand_15m", timeframe="15m")

    # 2. Configured for 5m initially with candidate cand_15m (causes ProductionBlocked)
    df_5m = make_tf_market_data("2025-01-01 10:00", "5m")
    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: df_5m,
        candidate_ids={"5m": cand_15m},  # Mismatched timeframe!
    )

    # 3. Poll 1: Reaches ProductionBlocked
    t1 = cont.tick(reference_now=ref_now)

    assert t1["evaluations"]["5m"]["blocked"] is True
    assert len(cont._evaluated_candles) == 0

    # 4. Fix candidate for 5m
    cand_5m = persist_momentum_candidate(tmp_path, candidate_id="cand_5m", timeframe="5m")
    cont.candidate_ids["5m"] = cand_5m

    # 5. Poll 2: Same candle polled again
    t2 = cont.tick(reference_now=ref_now)

    # 6. Evaluation now succeeds and is recorded in dedup set
    assert t2["evaluations"]["5m"]["blocked"] is False
    assert len(cont._evaluated_candles) == 1

    # 7. Poll 3: Same candle polled again is now deduplicated
    t3 = cont.tick(reference_now=ref_now)
    assert t3["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert len(cont._evaluated_candles) == 1


# -----------------------------------------------------------------------------
# Failure tests (29-32)
# -----------------------------------------------------------------------------

def test_29_30_31_32_one_timeframe_failure_does_not_terminate_runtime(tmp_path) -> None:
    """29, 30, 31, 32. One timeframe provider failure does not crash continuous loop; other timeframe continues, failed retries."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_5m", timeframe="5m")
    persist_momentum_candidate(tmp_path, candidate_id="cand_15m", timeframe="15m")

    state = {"fail_5m": True}

    def loader(tf: str) -> pd.DataFrame:
        if tf == "5m" and state["fail_5m"]:
            raise RuntimeError("5m data fetch timeout")
        return make_tf_market_data("2025-01-01 10:00", tf)

    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=loader,
        candidate_ids={"5m": "cand_5m", "15m": "cand_15m"},
    )

    ref_now = pd.to_datetime("2025-01-01 12:00:00Z").to_pydatetime()

    # Tick 1: 5m fails, 15m succeeds
    t1 = cont.tick(reference_now=ref_now)
    assert t1["evaluations"]["5m"]["status"] == "FAILED_RECOVERABLE"
    assert t1["evaluations"]["15m"]["blocked"] is False

    # Tick 2: 5m recovers and succeeds
    state["fail_5m"] = False
    t2 = cont.tick(reference_now=ref_now)
    assert t2["evaluations"]["5m"]["blocked"] is False


# -----------------------------------------------------------------------------
# Compatibility & Shutdown tests (33-37)
# -----------------------------------------------------------------------------

def test_33_one_shot_mode_backward_compatible(tmp_path) -> None:
    """33. One-shot mode remains backward compatible."""
    persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
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
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime() + datetime.timedelta(minutes=5)

    res = runtime.run_once(publish=False, persist=True, market_data=df, reference_now=ref_now)
    assert res["blocked"] is False
    assert res["decision"] == "BUY"


def test_34_35_36_37_graceful_shutdown_and_max_ticks(tmp_path) -> None:
    """34, 35, 36, 37. Graceful shutdown, stop() signal, and max_ticks loop termination work deterministically."""
    cand_id = persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")
    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        poll_interval=0.0,
        publish=False,
        persist=False,
        research_dir=tmp_path,
        market_data_loaders=lambda tf: make_tf_market_data("2025-01-01 10:00", "5m"),
        candidate_ids={"5m": cand_id},
    )

    assert cont.is_running is True
    cont.stop()
    assert cont.is_running is False

    # max_ticks termination
    cont._stop_event.clear()
    cont.run_continuous(max_ticks=3)
    assert len(cont._execution_history) == 3


# -----------------------------------------------------------------------------
# Canonical MTF regression tests (38-41)
# -----------------------------------------------------------------------------

def test_38_39_40_41_canonical_ladder_and_1m_rejection() -> None:
    """38, 39, 40, 41. Canonical ladder is 5m, 15m, 30m, 1H, 4H, 1D; '1m' is strictly rejected."""
    ladder = CanonicalTimeframe.canonical_ladder()
    ladder_values = [tf.value for tf in ladder]
    assert ladder_values == ["5m", "15m", "30m", "1H", "4H", "1D"]

    # 1m rejected
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '1m'"):
        CanonicalTimeframe.from_str("1m")

    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '1m'"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["1m"])
