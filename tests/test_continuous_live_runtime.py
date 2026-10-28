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


def make_fresh_tf_data(timeframe: str, ref_now: datetime.datetime, count: int = 10) -> pd.DataFrame:
    """Hermetic fresh candle slice for a timeframe, mirroring real provider shape.

    The last row is the currently OPEN candle (opens at ref_now). The second-to-last
    candle opens one duration earlier and is therefore CLOSED by ref_now, and every
    closed candle's age is within duration + max_age tolerance."""
    duration = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1H": 3600, "4H": 14400, "1D": 86400,
    }[timeframe]
    dates = [ref_now - datetime.timedelta(seconds=duration * i) for i in range(count - 1, -1, -1)]
    prices = [2000.0 + (i * 2.0) for i in range(count)]
    return pd.DataFrame({
        "openTime": [d.isoformat() for d in dates],
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "timestamp": dates,
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
    """15. Explicit candidate with mismatched DatasetScope timeframe fails closed.

    Fails closed for ACTIONS (no authorized candidate, no trade executed), while
    still producing the timeframe's own truthful NO_TRADE market observation with
    the eligibility blocker disclosed."""
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
    # No authorized action: the mismatched candidate must never act on this timeframe.
    assert eval_15m.get("observation") is True
    assert eval_15m.get("blocked") is False
    assert eval_15m.get("decision") == "NO TRADE"
    assert eval_15m.get("candidate_id") is None
    assert eval_15m.get("publish_result", {}).get("status") == "SKIPPED_DISABLED"
    # The eligibility blocker is honestly disclosed, naming the misconfigured scope.
    assert eval_15m.get("reason") == "PromotionEligibilityError"
    assert "timeframe '5m'" in eval_15m.get("detail", "")
    assert "15m" in eval_15m.get("detail", "")


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
    # Fail-closed: no authorized candidate AND no trade action. The freshest
    # truth (stale market data failed the freshness/closure gate for this
    # candle) is reported at top level, with the missing candidate blocker
    # disclosed in blocked_state.
    assert eval_5m["blocked"] is True
    assert eval_5m["reason"] == "stale_market_data"
    assert eval_5m["blocked_state"]["reason"] == "PromotionUnavailable"


def test_candidate_mapping_parser_exact_duplicate_key_rejected() -> None:
    """1. Exact duplicate key in explicit mapping must raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate candidate mapping key for canonical timeframe '5m'"):
        parse_continuous_candidate_ids("5m=cand_A,5m=cand_B", timeframes=["5m"])


def test_candidate_mapping_parser_canonical_duplicate_key_rejected() -> None:
    """2. Duplicate key after canonicalization (e.g. 1h and 1H) must raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate candidate mapping key for canonical timeframe '1H'"):
        parse_continuous_candidate_ids("1h=cand_A,1H=cand_B", timeframes=["1H"])


def test_candidate_mapping_parser_out_of_scope_key_rejected() -> None:
    """3. Explicit mapping key outside configured timeframe scope must raise ValueError."""
    with pytest.raises(ValueError, match="outside the configured timeframe scope"):
        parse_continuous_candidate_ids("5m=cand_5m,1D=cand_1d", timeframes=["5m"])


def test_candidate_mapping_parser_empty_middle_segment_rejected() -> None:
    """4. Empty middle segment in comma-separated mapping must raise ValueError."""
    with pytest.raises(ValueError, match="Contains empty comma-separated segment"):
        parse_continuous_candidate_ids("5m=cand_5m,,1D=cand_1d", timeframes=["5m", "1D"])


def test_candidate_mapping_parser_leading_trailing_empty_segment_rejected() -> None:
    """5. Leading or trailing empty segment in non-empty mapping must raise ValueError."""
    with pytest.raises(ValueError, match="Contains empty comma-separated segment"):
        parse_continuous_candidate_ids(",5m=cand_5m", timeframes=["5m"])

    with pytest.raises(ValueError, match="Contains empty comma-separated segment"):
        parse_continuous_candidate_ids("5m=cand_5m,", timeframes=["5m"])


def test_candidate_mapping_parser_duplicate_configured_timeframe_rejected() -> None:
    """6. Configured timeframes containing aliases that normalize to duplicate canonical identity must raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate canonical timeframe identity '1H'"):
        parse_continuous_candidate_ids("1H=cand_1h", timeframes=["1h", "1H"])


def test_candidate_mapping_parser_single_candidate_multi_timeframe_rejected() -> None:
    """7. Plain candidate ID supplied for multiple timeframes must raise ValueError."""
    with pytest.raises(ValueError, match="must specify explicit per-timeframe mappings"):
        parse_continuous_candidate_ids("cand_single", timeframes=["5m", "15m"])


def test_candidate_mapping_parser_unsupported_timeframe_fails_closed() -> None:
    """8. Unsupported timeframe names in mapping or configured timeframes fail closed."""
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '2m'"):
        parse_continuous_candidate_ids("2m=cand_2m", timeframes=["5m"])

    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '2m'"):
        parse_continuous_candidate_ids("5m=cand_5m", timeframes=["2m"])


def test_candidate_mapping_parser_valid_explicit_mapping() -> None:
    """9. Valid explicit mapping matching configured timeframes is accepted."""
    parsed = parse_continuous_candidate_ids("5m=cand_5m,15m=cand_15m", timeframes=["5m", "15m"])
    assert parsed == {"5m": "cand_5m", "15m": "cand_15m"}


def test_candidate_mapping_parser_independent_1m_and_5m_mapping() -> None:
    """10. Independent 1m and 5m mapping is accepted."""
    parsed = parse_continuous_candidate_ids("1m=cand_1m,5m=cand_5m", timeframes=["1m", "5m"])
    assert parsed == {"1m": "cand_1m", "5m": "cand_5m"}


def test_candidate_mapping_parser_valid_partial_mapping_seven_timeframes() -> None:
    """11. Valid partial mapping across seven configured timeframes is accepted."""
    seven_tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    parsed = parse_continuous_candidate_ids("5m=cand_5m,1D=cand_1d", timeframes=seven_tfs)
    assert parsed == {"5m": "cand_5m", "1D": "cand_1d"}


def test_candidate_mapping_parser_render_yaml_exact_mapping_accepted() -> None:
    """12. Exact mapping specified in render.yaml is accepted."""
    seven_tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    parsed = parse_continuous_candidate_ids("5m=cand_moving_average_5m,1D=cand_moving_average_1d", timeframes=seven_tfs)
    assert parsed == {"5m": "cand_moving_average_5m", "1D": "cand_moving_average_1d"}


def test_candidate_mapping_parser_single_candidate_single_timeframe_accepted() -> None:
    """13. Plain candidate ID with exactly one configured timeframe is accepted."""
    parsed = parse_continuous_candidate_ids("cand_single", timeframes=["5m"])
    assert parsed == {"5m": "cand_single"}


def test_candidate_mapping_parser_whitespace_tolerance() -> None:
    """14. Surrounding whitespace around valid components is supported cleanly."""
    parsed = parse_continuous_candidate_ids(" 5m = cand_5m , 1D = cand_1d ", timeframes=["5m", "1D"])
    assert parsed == {"5m": "cand_5m", "1D": "cand_1d"}


def test_candidate_mapping_parser_empty_or_none_compatible() -> None:
    """15. None or empty argument string returns empty dict."""
    assert parse_continuous_candidate_ids(None, timeframes=["5m"]) == {}
    assert parse_continuous_candidate_ids("", timeframes=["5m"]) == {}
    assert parse_continuous_candidate_ids("   ", timeframes=["5m"]) == {}


def test_continuous_live_runtime_constructor_empty_timeframes_rejected() -> None:
    """Boundary B: ContinuousLiveRuntime constructor with empty timeframes sequence must raise ValueError."""
    with pytest.raises(ValueError, match="ContinuousLiveRuntime requires at least one configured timeframe"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=[])

    with pytest.raises(ValueError, match="Invalid timeframes string"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=" , ")


def test_continuous_live_runtime_constructor_direct_candidate_ids_out_of_scope_rejected() -> None:
    """Boundary A1: Direct candidate_ids injection with key outside configured timeframe set must raise ValueError."""
    with pytest.raises(ValueError, match="outside the configured timeframe scope"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], candidate_ids={"1D": "cand_1d"})


def test_continuous_live_runtime_constructor_direct_candidate_ids_canonical_duplicates_rejected() -> None:
    """Boundary A2: Direct candidate_ids injection with duplicate canonical keys (1h and 1H) must raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate candidate mapping key for canonical timeframe '1H'"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["1H"], candidate_ids={"1h": "cand_a", "1H": "cand_b"})


def test_continuous_live_runtime_constructor_direct_candidate_ids_invalid_types_or_empty_rejected() -> None:
    """Boundary A3: Direct candidate_ids injection with non-string, empty, or whitespace keys/values must raise ValueError."""
    with pytest.raises(ValueError, match="candidate IDs must be non-empty strings"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], candidate_ids={"5m": ""})

    with pytest.raises(ValueError, match="candidate IDs must be non-empty strings"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], candidate_ids={"5m": "   "})

    with pytest.raises(ValueError, match="mapping keys must be non-empty strings"):
        ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"], candidate_ids={"": "cand_5m"})  # type: ignore


def test_continuous_live_runtime_constructor_direct_candidate_ids_valid_partial_accepted() -> None:
    """Boundary A4: Direct candidate_ids injection with valid partial mapping is accepted and normalized."""
    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1m", "5m", "15m", "30m", "1H", "4H", "1D"],
        candidate_ids={"5m": "cand_5m", "1d": "cand_1d"},
    )
    assert cont.candidate_ids == {"5m": "cand_5m", "1D": "cand_1d"}


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

def test_38_39_40_41_canonical_ladder_includes_seven_canonical_timeframes() -> None:
    """Canonical ladder matches the seven timeframe product contract including 1m."""
    ladder = CanonicalTimeframe.canonical_ladder()
    ladder_values = [tf.value for tf in ladder]
    assert ladder_values == ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    assert ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["5m"]).timeframes == ("5m",)
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '2m'"):
        CanonicalTimeframe.from_str("2m")


def test_independent_startup_readiness_partial_blocked_runs_healthy_timeframe(tmp_path: Path) -> None:
    """Test A: One BLOCKED timeframe (missing candidate) must not prevent healthy READY timeframes from starting.

    Verifies:
    1. Both 5m (READY) and 15m (BLOCKED: missing candidate) are evaluated in readiness check.
    2. 5m is READY, 15m is BLOCKED with reason code PromotionUnavailable.
    3. Startup readiness preflight passes because at least 1 timeframe (5m) is READY.
    4. Continuous loop tick evaluates 5m normally and emits BLOCKED status for 15m without aborting tick.
    5. 15m cannot borrow 5m candidate or publish signals.
    """
    from unittest.mock import patch
    from src.evaluation.research_store import save_research_candidate
    from tests.test_timeframe_identity_invariants import _make_test_evidence

    # Create candidate artifact ONLY for 5m
    save_research_candidate(
        candidate_id="cand_5m_only",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.88,
        base_dir=tmp_path,
    )

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        research_dir=tmp_path,
        candidate_ids={
            "5m": "cand_5m_only",
            "15m": "missing_15m_cand",
        },
        publish=False,
    )

    readiness = cont_runtime.verify_startup_readiness()
    assert readiness["5m"]["status"] == "READY"
    assert readiness["5m"]["candidate_id"] == "cand_5m_only"

    assert readiness["15m"]["status"] == "BLOCKED"
    assert readiness["15m"]["reason_code"] == "PromotionUnavailable"

    # Execute tick and verify 5m runs healthy while 15m remains BLOCKED safely
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    dates = [now_dt - datetime.timedelta(minutes=5 * i) for i in range(10)]
    dates.reverse()
    df_5m = pd.DataFrame({
        "openTime": [d.isoformat() for d in dates],
        "open": [2000.0] * 10,
        "high": [2005.0] * 10,
        "low": [1995.0] * 10,
        "close": [2002.0] * 10,
    })

    with patch.object(cont_runtime, "_acquire_market_data", return_value=df_5m):
        tick_res = cont_runtime.tick(reference_now=now_dt)

        eval_5m = tick_res["evaluations"]["5m"]
        eval_15m = tick_res["evaluations"]["15m"]

        assert eval_5m["blocked"] is False
        assert eval_5m["candidate_id"] == "cand_5m_only"

        assert eval_15m.get("observation") is True
        assert eval_15m.get("blocked") is False
        assert eval_15m.get("decision") == "NO TRADE"
        assert eval_15m.get("candidate_id") is None
        assert eval_15m.get("reason") == "PromotionUnavailable"
        assert eval_15m.get("publish_result", {}).get("status") == "SKIPPED_DISABLED"


def test_blocked_timeframe_does_not_suppress_other_timeframes_publication(tmp_path: Path) -> None:
    """Acceptance: a missing/unqualified candidate on one timeframe must not suppress that
    timeframe's own evaluation outcome nor prevent another timeframe's valid decision from
    reaching the P2 publisher. Publication is per-timeframe independent."""
    from unittest.mock import MagicMock
    from src.evaluation.research_store import save_research_candidate
    from tests.test_timeframe_identity_invariants import _make_test_evidence

    save_research_candidate(
        candidate_id="cand_5m_pub",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.88,
        base_dir=tmp_path,
    )

    mock_publisher = MagicMock()
    mock_publisher.enabled = True
    mock_publisher.publish_url = "https://project2.example/api/v1/integration/project1/ingest"
    mock_publisher.api_key = "test_api_key"
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
        "delivery_status": "DELIVERED",
    }

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        research_dir=tmp_path,
        store_path=tmp_path / "decisions.json",
        snapshot_path=tmp_path / "snapshots.json",
        candidate_ids={"5m": "cand_5m_pub", "15m": "missing_15m_cand"},
        publish=True,
        publisher=mock_publisher,
        max_age_seconds=3600.0,
    )

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    dates = [now_dt - datetime.timedelta(minutes=5 * i) for i in range(100)]
    dates.reverse()
    prices = [2000.0 + (i * 2.0) for i in range(100)]
    df_5m = pd.DataFrame({
        "openTime": [d.isoformat() for d in dates],
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
    })

    def loader(tf: str) -> pd.DataFrame:
        return df_5m

    cont_runtime.verify_startup_readiness()
    tick_res = cont_runtime.tick(reference_now=now_dt)

    # 15m is independently evaluated on its own real candle slice; with no
    # candidate of its own it fails closed for ACTIONS and publishes its own
    # truthful NO_TRADE market observation (never claiming the 5m candidate).
    eval_15m = tick_res["evaluations"]["15m"]
    assert eval_15m.get("observation") is True
    assert eval_15m.get("blocked") is False
    assert eval_15m.get("decision") == "NO TRADE"
    assert eval_15m.get("candidate_id") is None
    assert eval_15m.get("reason") == "PromotionUnavailable"
    assert eval_15m.get("publish_result", {}).get("published") is True
    assert eval_15m.get("publication", {}).get("signal", {}).get("decision") == "NO TRADE"
    assert eval_15m.get("publication", {}).get("provenance", {}).get("authorized") is False

    # 5m evaluated independently and its valid decision was delivered to P2
    # even without the 15m candidate.
    eval_5m = tick_res["evaluations"]["5m"]
    assert eval_5m["blocked"] is False
    assert eval_5m["symbol"] == "XAUUSD"
    assert eval_5m["interval"] == "5m"
    assert eval_5m["candidate_id"] == "cand_5m_pub"
    assert eval_5m["decision"] in ("BUY", "SELL", "NO TRADE")
    assert eval_5m["publish_result"]["published"] is True
    assert eval_5m["publish_result"]["status"] == "PUBLISHED"

    # Both timeframes reached the single publisher path independently.
    assert mock_publisher.publish.call_count == 2


def test_all_seven_configured_timeframes_independently_attempted(tmp_path: Path) -> None:
    """Acceptance: all seven canonical timeframes are configured and independently attempted;
    candidate-less timeframes are evaluated by the direct strategy WITHOUT any candidate or
    promotion artifact (no candidate blockers, no borrowed data), while an explicitly
    configured-but-missing candidate still fails closed for that timeframe alone."""
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1m", "5m", "15m", "30m", "1H", "4H", "1D"],
        poll_interval=0.0,
        research_dir=tmp_path,
        store_path=tmp_path / "decisions.json",
        snapshot_path=tmp_path / "snapshots.json",
        candidate_ids={"1D": "missing_1d"},
        publish=False,
        persist=False,
        market_data_loaders=lambda tf: make_fresh_tf_data(tf, now_dt),
    )

    readiness = cont_runtime.verify_startup_readiness()
    assert set(readiness.keys()) == {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}

    # Candidate-less timeframes are READY without any promotion artifact.
    for tf in ("1m", "5m", "15m", "30m", "1H", "4H"):
        assert readiness[tf]["status"] == "READY", tf
        assert readiness[tf]["candidate_id"] is None, tf
        assert readiness[tf]["reason_code"] == "READY", tf

    # 1D explicitly maps a candidate ID that does not exist -> promoted-candidate
    # path fails closed for that timeframe alone.
    assert readiness["1D"]["status"] == "BLOCKED"
    assert readiness["1D"]["reason_code"] == "PromotionUnavailable"

    tick_res = cont_runtime.tick(reference_now=now_dt)
    evaluations = tick_res["evaluations"]
    assert set(evaluations.keys()) == {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}

    for tf in ("1m", "5m", "15m", "30m", "1H", "4H"):
        # Each candidate-less timeframe produces its own direct-strategy
        # decision derived ONLY from its own data (no candidate consulted),
        # with publication disabled in this test.
        ev = evaluations[tf]
        assert ev.get("direct_signal") is True, tf
        assert ev.get("blocked") is False, tf
        assert ev.get("decision") in ("BUY", "SELL", "NO TRADE"), tf
        assert ev.get("candidate_id") is None, tf
        assert ev.get("interval") == tf, tf
        assert ev.get("publish_result", {}).get("status") == "SKIPPED_DISABLED", tf
        # Deterministic outcome for the monotonically rising fresh fixture with
        # 10 candles: insufficient history for the slow MA -> truthful NO TRADE.
        assert ev.get("decision") == "NO TRADE", tf
        assert ev.get("reason") == "insufficient_history", tf
        assert ev.get("trade_levels") is None, tf

    # 1D keeps its own truthful candidate-gated observation: no authorized
    # candidate for that timeframe, never borrowing another frame's decision.
    ev_1d = evaluations["1D"]
    assert ev_1d.get("observation") is True
    assert ev_1d.get("blocked") is False
    assert ev_1d.get("decision") == "NO TRADE"
    assert ev_1d.get("candidate_id") is None
    assert ev_1d.get("reason") == "PromotionUnavailable"
    assert ev_1d.get("publish_result", {}).get("status") == "SKIPPED_DISABLED"

    # The runtime is still alive and each frame remains independently recoverable.
    assert cont_runtime.is_running


def test_stale_timeframe_readiness_cache_invalidation_on_failed_preflight(tmp_path: Path) -> None:
    """Verify that a failed preflight invalidates any previous cached _timeframe_readiness matrix."""
    from unittest.mock import patch
    from src.evaluation.research_store import save_research_candidate
    from tests.test_timeframe_identity_invariants import _make_test_evidence

    # 1. Create candidate artifact for 5m
    save_research_candidate(
        candidate_id="cand_5m_cache",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.88,
        base_dir=tmp_path,
    )

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "15m"],
        research_dir=tmp_path,
        candidate_ids={
            "5m": "cand_5m_cache",
            "15m": "missing_15m",
        },
        publish=False,
    )

    # First preflight: 5m is READY -> cached
    first_res = cont_runtime.verify_startup_readiness()
    assert first_res["5m"]["status"] == "READY"
    assert cont_runtime._timeframe_readiness["5m"]["status"] == "READY"

    # Second preflight: mock verify_timeframe_production_readiness to return BLOCKED for all
    blocked_mock = {
        "timeframe": "5m",
        "status": "BLOCKED",
        "reason_code": "PromotionUnavailable",
        "detail": "Mocked failure for stale cache test",
        "candidate_id": "cand_5m_cache",
    }
    with patch("src.evaluation.live_execution_runtime.verify_timeframe_production_readiness", return_value=blocked_mock):
        # Candidate unavailability must NOT abort the worker: it is cached as per-timeframe
        # BLOCKED state and reported independently, while the process keeps running.
        second_res = cont_runtime.verify_startup_readiness()
        assert second_res["5m"]["status"] == "BLOCKED"
        assert second_res["15m"]["status"] == "BLOCKED"

    # Assert that stale readiness cache from first preflight was cleared and replaced
    assert cont_runtime._timeframe_readiness["5m"]["status"] == "BLOCKED"
    assert cont_runtime._timeframe_readiness["15m"]["status"] == "BLOCKED"


def test_all_blocked_timeframes_do_not_abort_worker(tmp_path: Path) -> None:
    """A worker with 0 READY timeframes must NOT crash: each timeframe independently
    reports its own truthful outcome on preflight and on every tick."""
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1m", "15m"],
        research_dir=tmp_path,
        store_path=tmp_path / "decisions.json",
        snapshot_path=tmp_path / "snapshots.json",
        candidate_ids={
            "1m": "missing_1m",
            "15m": "missing_15m",
        },
        publish=False,
        persist=False,
        market_data_loaders=lambda tf: make_fresh_tf_data(tf, now_dt),
    )

    # Preflight must NOT raise: candidate unavailability is per-timeframe.
    readiness = cont_runtime.verify_startup_readiness()
    assert readiness["1m"]["status"] == "BLOCKED"
    assert readiness["1m"]["reason_code"] == "PromotionUnavailable"
    assert readiness["15m"]["status"] == "BLOCKED"
    assert readiness["15m"]["reason_code"] == "PromotionUnavailable"

    # Continuous ticks proceed and each timeframe produces its own truthful
    # NO_TRADE observation with the missing-candidate blocker disclosed
    # (publication disabled in this test).
    tick_res = cont_runtime.tick(reference_now=now_dt)
    assert tick_res["evaluations"]["1m"].get("observation") is True
    assert tick_res["evaluations"]["1m"].get("blocked") is False
    assert tick_res["evaluations"]["1m"].get("decision") == "NO TRADE"
    assert tick_res["evaluations"]["1m"].get("reason") == "PromotionUnavailable"
    assert tick_res["evaluations"]["15m"].get("observation") is True
    assert tick_res["evaluations"]["15m"].get("blocked") is False
    assert tick_res["evaluations"]["15m"].get("decision") == "NO TRADE"
    assert tick_res["evaluations"]["15m"].get("reason") == "PromotionUnavailable"

    # The rejection remains for publish-mode misuse: publishing requested while the
    # publisher configuration is unusable is a process-level failure.
    unusable = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1D"],
        research_dir=tmp_path,
        candidate_ids={"1D": "missing_1d"},
        publish=True,
    )
    from src.evaluation.research_store import PromotionUnavailable
    with pytest.raises(PromotionUnavailable, match="unusable"):
        unusable.verify_startup_readiness()


def test_independent_startup_readiness_seven_timeframe_ladder(tmp_path: Path) -> None:
    """Test E: Canonical seven-timeframe ladder readiness preflight matrix."""
    from src.evaluation.research_store import save_research_candidate
    from tests.test_timeframe_identity_invariants import _make_test_evidence

    # Only 5m and 1D candidate artifacts created
    save_research_candidate(
        candidate_id="cand_5m",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )
    save_research_candidate(
        candidate_id="cand_1d",
        evidence=_make_test_evidence("1D"),
        operational_stability_score=0.90,
        base_dir=tmp_path,
    )

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=CanonicalTimeframe.canonical_ladder(),
        research_dir=tmp_path,
        candidate_ids={
            "5m": "cand_5m",
            "1D": "cand_1d",
        },
        publish=False,
    )

    readiness = cont_runtime.verify_startup_readiness()
    assert len(readiness) == 7

    # Explicitly configured candidates remain gated by promotion artifacts.
    assert readiness["5m"]["status"] == "READY"
    assert readiness["1D"]["status"] == "READY"

    # Candidate-less timeframes are direct-strategy frames: READY without any
    # promotion artifact and never borrowing another timeframe's candidate.
    for direct_tf in ("1m", "15m", "30m", "1H", "4H"):
        assert readiness[direct_tf]["status"] == "READY"
        assert readiness[direct_tf]["candidate_id"] is None
        assert readiness[direct_tf]["reason_code"] == "READY"


def test_timeframe_scoped_persistence_path_isolation(tmp_path: Path) -> None:
    """Verify that ContinuousLiveRuntime writes snapshots and store history to distinct timeframe-scoped paths."""
    from unittest.mock import patch
    from src.evaluation.research_store import save_research_candidate
    from tests.test_timeframe_identity_invariants import _make_test_evidence

    # Create candidate artifacts for 1m and 5m
    save_research_candidate(
        candidate_id="cand_1m",
        evidence=_make_test_evidence("1m"),
        operational_stability_score=0.80,
        base_dir=tmp_path,
    )
    save_research_candidate(
        candidate_id="cand_5m",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )

    base_store = tmp_path / "live" / "decision_history.json"
    base_snapshot = tmp_path / "live" / "latest_execution.json"

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1m", "5m"],
        research_dir=tmp_path,
        store_path=base_store,
        snapshot_path=base_snapshot,
        candidate_ids={
            "1m": "cand_1m",
            "5m": "cand_5m",
        },
        publish=False,
    )

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    dates = [now_dt - datetime.timedelta(minutes=i) for i in range(10)]
    dates.reverse()
    df_1m = pd.DataFrame({
        "openTime": [d.isoformat() for d in dates],
        "open": [2000.0] * 10,
        "high": [2005.0] * 10,
        "low": [1995.0] * 10,
        "close": [2002.0] * 10,
    })

    with patch.object(cont_runtime, "_acquire_market_data", return_value=df_1m):
        tick_res = cont_runtime.tick(reference_now=now_dt)

        # Confirm 1m and 5m were evaluated independently
        eval_1m = tick_res["evaluations"]["1m"]
        eval_5m = tick_res["evaluations"]["5m"]

        assert eval_1m["blocked"] is False
        assert eval_1m["candidate_id"] == "cand_1m"
        assert eval_1m["interval"] == "1m"

        assert eval_5m["blocked"] is False
        assert eval_5m["candidate_id"] == "cand_5m"
        assert eval_5m["interval"] == "5m"

        # Verify distinct timeframe-scoped snapshot files exist
        snap_1m_file = tmp_path / "live" / "1m" / "latest_execution.json"
        snap_5m_file = tmp_path / "live" / "5m" / "latest_execution.json"

        assert snap_1m_file.exists()
        assert snap_5m_file.exists()

        snap_1m = json.loads(snap_1m_file.read_text(encoding="utf-8"))
        snap_5m = json.loads(snap_5m_file.read_text(encoding="utf-8"))

        assert snap_1m["candidate_id"] == "cand_1m"
        assert snap_1m["interval"] == "1m"

        assert snap_5m["candidate_id"] == "cand_5m"
        assert snap_5m["interval"] == "5m"

        # Verify distinct timeframe-scoped decision history store files exist
        store_1m_file = tmp_path / "live" / "1m" / "decision_history.json"
        store_5m_file = tmp_path / "live" / "5m" / "decision_history.json"

        assert store_1m_file.exists()
        assert store_5m_file.exists()
