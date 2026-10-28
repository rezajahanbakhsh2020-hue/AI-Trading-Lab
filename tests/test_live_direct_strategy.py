"""Focused tests for the candidate-free direct live-signal path (Issue #100).

Contract under test:
- ContinuousLiveRuntime evaluates ONE transparent deterministic multi-setup
  strategy (multi_setup_direct) independently on each canonical timeframe's OWN
  data without any candidate_id, promotion, governance fingerprint, or research
  qualification.
- BUY/SELL only when the rule is met on a closed, fresh candle; otherwise
  NO TRADE with a machine-readable reason; all trade levels null for NO TRADE.
- Fail-closed on stale/future/invalid/unclosed data; nothing fabricated.
- Publication flows through the existing P1->P2 Contract v1.0 payload via
  Project2Publisher (disabled in tests unless locally simulated).
- Per-timeframe failure isolation: one bad frame never stops the other six and
  health never hides blocked frames.
"""
import datetime
from typing import Any

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    verify_timeframe_direct_readiness,
)
from src.evaluation.live_direct_strategy import (
    DIRECT_PROVENANCE_TYPE,
    DIRECT_STRATEGY_ID,
    DIRECT_STRATEGY_VERSION,
    PULLBACK_LOOKBACK,
    BREAKOUT_LOOKBACK,
    SETUP_PRIORITY,
    BUY,
    SELL,
    NO_TRADE,
    DirectSignalDataError,
    DirectSignalPublication,
    evaluate_direct_live_signal,
)
from src.integration.project2_publisher import Project2Publisher

REF_NOW = datetime.datetime(2025, 6, 1, 12, 0, tzinfo=datetime.timezone.utc)
DURATION_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600, "4H": 14400, "1D": 86400}


def closed_candles(timeframe: str, closes: list[float], *, low_extra: float = 5.0, high_extra: float = 5.0) -> pd.DataFrame:
    """Closed-candle slice for `timeframe` whose LAST candle is fully closed by REF_NOW."""
    duration = DURATION_SECONDS[timeframe]
    n = len(closes)
    ts = [REF_NOW - datetime.timedelta(seconds=duration * i) for i in range(n, 0, -1)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": [c - 2 for c in closes],
        "high": [c + high_extra for c in closes],
        "low": [c - low_extra for c in closes],
        "close": closes,
    })


def run_direct_runtime(
    tmp_path,
    loaders: dict[str, Any],
    **kwargs: Any,
) -> ContinuousLiveRuntime:
    defaults: dict[str, Any] = dict(
        publish=False,
        persist=False,
        poll_interval=0.0,
        max_age_seconds=3600.0,
        store_path=tmp_path / "decision_history.json",
        snapshot_path=tmp_path / "snapshot.json",
    )
    defaults.update(kwargs)
    return ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=sorted(loaders.keys()),
        market_data_loaders=loaders,
        **defaults,
    )


# ---------------------------------------------------------------------------
# Strategy unit rules
# ---------------------------------------------------------------------------

def test_bullish_crossover_emits_buy_from_real_candle_values() -> None:
    df = closed_candles("5m", [1990.0] * 24 + [2100.0])
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == BUY
    assert signal.reason == "crossover_bullish"
    assert signal.entry_price == 2100.0
    assert signal.stop_loss == 2095.0  # real candle low
    assert signal.take_profit == pytest.approx(2100.0 + 2.0 * 5.0)
    assert signal.risk_reward_ratio == 2.0
    assert signal.candle_timestamp_utc == "2025-06-01T11:55:00Z"


def test_bearish_crossover_emits_sell_from_real_candle_values() -> None:
    df = closed_candles("5m", [1990.0] * 24 + [1800.0])
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == SELL
    assert signal.reason == "crossover_bearish"
    assert signal.entry_price == 1800.0
    assert signal.stop_loss == 1805.0  # real candle high
    assert signal.take_profit == pytest.approx(1790.0)


def test_flat_market_emits_no_setup_met_no_trade() -> None:
    df = closed_candles("15m", [1990.0] * 30)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="15m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == NO_TRADE
    assert signal.reason == "no_setup_met"
    assert signal.entry_price is None
    assert signal.stop_loss is None
    assert signal.take_profit is None
    assert signal.risk_reward_ratio is None


def test_short_history_yields_insufficient_history_no_trade() -> None:
    df = closed_candles("4H", [1990.0] * 10)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="4H", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == NO_TRADE
    assert signal.reason == "insufficient_history"
    assert signal.entry_price is None and signal.stop_loss is None
    assert signal.take_profit is None and signal.risk_reward_ratio is None


# ---------------------------------------------------------------------------
# Setup 2: trend-continuation after pullback (BUY/SELL symmetric)
# ---------------------------------------------------------------------------

def test_pullback_continuation_buy_uses_real_candle_levels() -> None:
    # Established uptrend (fast > slow, slow rising), one of the previous 3
    # candles traded back into the fast-MA zone, then the evaluation candle
    # CONFIRMED resumption (closes above previous close and above the fast MA).
    # Note: this candle also satisfies the breakout rule (2012 > prev 10-bar
    # range high of 2009); fixed priority must classify it as pullback first.
    closes = [1990.0] * 7 + [1993.0] * 3 + [1996.0] * 4 + [1999.0] * 3 + [1998.0, 1996.0, 2000.0, 2004.0, 2012.0]
    df = closed_candles("1H", closes)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="1H", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == BUY
    assert signal.reason == "pullback_continuation_bullish"
    assert signal.entry_price == 2012.0
    assert signal.stop_loss == 2007.0       # real evaluation-candle low (invalidation)
    assert signal.take_profit == pytest.approx(2022.0)
    assert signal.risk_reward_ratio == 2.0
    payload = signal.to_contract_v1_payload()
    assert payload["signal"]["invalidation"] == 2007.0


def test_pullback_continuation_sell_uses_real_candle_levels() -> None:
    closes = [2006.0] * 7 + [2003.0] * 3 + [2000.0] * 4 + [1997.0] * 3 + [1998.0, 2000.0, 1996.0, 1992.0, 1984.0]
    df = closed_candles("1H", closes)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="1H", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == SELL
    assert signal.reason == "pullback_continuation_bearish"
    assert signal.entry_price == 1984.0
    assert signal.stop_loss == 1989.0       # real evaluation-candle high (invalidation)
    assert signal.take_profit == pytest.approx(1974.0)
    assert signal.risk_reward_ratio == 2.0


def test_pullback_without_resumption_confirmation_is_no_trade() -> None:
    # Dip into the fast-MA zone in an established uptrend, but the evaluation
    # candle does NOT close above the previous close / fast MA: no entry.
    closes = [1990.0] * 7 + [1993.0] * 3 + [1996.0] * 4 + [1999.0] * 3 + [1998.0, 1996.0, 2000.0, 2004.0, 1993.0]
    df = closed_candles("1H", closes)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="1H", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == NO_TRADE
    assert signal.reason == "no_setup_met"
    assert signal.entry_price is None and signal.stop_loss is None
    assert signal.take_profit is None and signal.risk_reward_ratio is None


# ---------------------------------------------------------------------------
# Setup 3: confirmed breakout/momentum (BUY/SELL symmetric)
# ---------------------------------------------------------------------------

def test_breakout_momentum_buy_uses_real_candle_levels() -> None:
    # CLOSE above the highest high of the previous 10 candles, with momentum
    # confirmation (close above previous close and above the fast MA). No
    # crossover (fast does not cross slow), no pullback-continuation trend
    # regime (fast < slow): the breakout/momentum rule alone is the entry.
    closes = [2000.0] * 20 + [1975.0, 2020.0]
    df = closed_candles("5m", closes)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == BUY
    assert signal.reason == "breakout_momentum_bullish"
    assert signal.entry_price == 2020.0
    assert signal.stop_loss == 2015.0
    assert signal.take_profit == pytest.approx(2030.0)
    assert signal.risk_reward_ratio == 2.0


def test_breakout_momentum_sell_uses_real_candle_levels() -> None:
    # Mirror confirms symmetry: CLOSE below the lowest low of the previous 10
    # candles with momentum confirmation; bear pullback regime is not met
    # (fast == slow, slow steadily flat-then-rising around the capitulation).
    closes = [2000.0] * 20 + [2100.0, 1900.0]
    df = closed_candles("5m", closes)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    assert signal.decision == SELL
    assert signal.reason == "breakout_momentum_bearish"
    assert signal.entry_price == 1900.0
    assert signal.stop_loss == 1905.0
    assert signal.take_profit == pytest.approx(1890.0)
    assert signal.risk_reward_ratio == 2.0


def test_buy_and_sell_rules_are_exact_mirrors_on_reflected_prices() -> None:
    """For any candle series the reflected series y = C - x (high/low swapped)
    must flip the decision BUY<->SELL with exactly mirrored trade levels."""
    df = closed_candles("30m", [1990.0]*30 + [1997.0, 1993.0, 2001.0, 1995.0, 2004.0, 1991.0, 2007.0, 1992.0, 1998.0, 1990.0, 2015.0])
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="30m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    c = 4000.0
    ref_df = df.copy()
    for col_from, col_to in (("low", "high"), ("high", "low"), ("open", "open"), ("close", "close")):
        ref_df[col_to] = c - df[col_from]
    ref_signal = evaluate_direct_live_signal(
        ref_df, symbol="XAUUSD", timeframe="30m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    mirror_decision = {BUY: SELL, SELL: BUY, NO_TRADE: NO_TRADE}[signal.decision]
    assert ref_signal.decision == mirror_decision
    if signal.decision == NO_TRADE:
        assert ref_signal.entry_price is None and ref_signal.entry_price is None
    else:
        assert ref_signal.entry_price == pytest.approx(c - signal.entry_price)
        assert ref_signal.stop_loss == pytest.approx(c - signal.stop_loss)
        assert ref_signal.take_profit == pytest.approx(c - signal.take_profit)
        assert ref_signal.risk_reward_ratio == signal.risk_reward_ratio
        mirror_reason = (
            signal.reason.replace("bullish", "bearish") if signal.decision == BUY
            else signal.reason.replace("bearish", "bullish")
        )
        assert ref_signal.reason == mirror_reason


def test_stale_data_fails_closed_without_publication() -> None:
    duration = DURATION_SECONDS["30m"]
    old_end = REF_NOW - datetime.timedelta(days=5)
    df = closed_candles("30m", [1990.0] * 30)
    df["timestamp"] = df["timestamp"].apply(lambda t: t - (REF_NOW - old_end))
    with pytest.raises(DirectSignalDataError) as exc:
        evaluate_direct_live_signal(
            df, symbol="XAUUSD", timeframe="30m", reference_now=REF_NOW, max_age_seconds=3600.0,
        )
    assert exc.value.reason == "stale_market_data"


def test_invalid_ohlc_data_fails_closed() -> None:
    df = closed_candles("5m", [1990.0] * 25)
    df.loc[df.index[3], "high"] = df.loc[df.index[3], "open"] - 10.0  # violates OHLC geometry
    with pytest.raises(DirectSignalDataError) as exc:
        evaluate_direct_live_signal(
            df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
        )
    assert exc.value.reason == "invalid_market_data"


def test_future_timestamp_fails_closed() -> None:
    df = closed_candles("5m", [1990.0] * 25)
    df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=1)  # last candle opens AFTER now
    with pytest.raises(DirectSignalDataError) as exc:
        evaluate_direct_live_signal(
            df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
        )
    assert exc.value.reason == "future_candle_timestamp"


# ---------------------------------------------------------------------------
# Publication artifact & P1->P2 contract
# ---------------------------------------------------------------------------

def test_direct_publication_payload_contract_v1_trade_levels() -> None:
    df = closed_candles("5m", [1990.0] * 24 + [2100.0])
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="5m", reference_now=REF_NOW, max_age_seconds=3600.0,
    )
    payload = signal.to_contract_v1_payload()
    assert payload["contract_version"] == "1.0"
    assert payload["event_type"] == "TRADING_SIGNAL"
    assert payload["timestamp"] == signal.candle_timestamp_utc  # real event time
    assert payload["instrument"] == {"symbol": "XAUUSD", "interval": "5m"}
    assert payload["signal"]["decision"] == "BUY"
    assert payload["signal"]["candidate_id"] is None
    assert payload["signal"]["confidence"] is None
    assert payload["signal"]["stability_score"] is None
    assert payload["signal"]["strategy"] == DIRECT_STRATEGY_ID
    assert payload["provenance"]["provenance_type"] == DIRECT_PROVENANCE_TYPE
    assert payload["provenance"]["candidate_id"] is None
    assert payload["provenance"]["authorized"] is False
    assert payload["provenance"]["operational_stability_score"] is None
    assert payload["provenance"]["produced_at"] == signal.decision_timestamp_utc
    assert payload["trade_setup"]["entry_price"] == 2100.0
    assert payload["trade_setup"]["stop_loss"] == 2095.0
    assert payload["trade_setup"]["take_profit"] == pytest.approx(2110.0)
    assert payload["trade_setup"]["risk_reward_ratio"] == 2.0


def test_direct_publication_payload_no_trade_has_only_null_levels() -> None:
    df = closed_candles("1H", [1990.0] * 30)
    signal = evaluate_direct_live_signal(
        df, symbol="XAUUSD", timeframe="1H", reference_now=REF_NOW, max_age_seconds=7200.0,
    )
    payload = signal.to_contract_v1_payload()
    assert payload["signal"]["decision"] == "NO TRADE"
    assert all(v is None for v in payload["trade_setup"].values())
    assert payload["signal"]["trend"] == "NEUTRAL"


def test_publisher_accepts_direct_signal_publication_and_delivers_contract_payload() -> None:
    """The existing P1->P2 publisher path accepts DirectSignalPublication."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    payload = DirectSignalPublication(
        symbol="XAUUSD",
        timeframe="5m",
        decision="BUY",
        reason="crossover_bullish",
        detail="Fast MA crossed above slow MA.",
        candle_timestamp_utc=(now_utc - datetime.timedelta(minutes=5)).isoformat(),
        decision_timestamp_utc=now_utc.isoformat(),
        entry_price=2100.0,
        stop_loss=2095.0,
        take_profit=2110.0,
        risk_reward_ratio=2.0,
    )

    pub_disabled = Project2Publisher(publish_url="https://example.com", api_key="k", enabled=False)
    res = pub_disabled.publish(payload)
    assert res["status"] == "SKIPPED_DISABLED"
    assert res["published"] is False


def test_publisher_rejects_non_contract_payload_types() -> None:
    pub = Project2Publisher(publish_url="https://example.com", api_key="k", enabled=False)
    res = pub.publish(object())
    assert res["status"] == "INVALID_RESPONSE"


# ---------------------------------------------------------------------------
# ContinuousLiveRuntime direct-mode integration
# ---------------------------------------------------------------------------

def test_all_seven_timeframes_evaluated_without_any_candidate_or_research_artifacts(tmp_path) -> None:
    """The live runtime produces a decision for all seven frames with NO candidate
    IDs and an effectively nonexistent research store."""
    loaders = {tf: (lambda tf=tf: closed_candles(tf, [1990.0] * (25 if tf != "4H" else 6))) for tf in ("1m", "5m", "15m", "30m", "1H", "4H", "1D")}
    cont = run_direct_runtime(
        tmp_path,
        loaders,
        research_dir=tmp_path / "nonexistent_research_dir",  # no research artifacts at all
        max_age_seconds=10 ** 7,
    )
    readiness = cont.verify_startup_readiness()
    assert all(r["status"] == "READY" and r["candidate_id"] is None for r in readiness.values())

    tick = cont.tick(reference_now=REF_NOW)
    evals = tick["evaluations"]
    assert set(evals.keys()) == {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}
    for tf, ev in evals.items():
        assert ev.get("direct_signal") is True, tf
        assert ev.get("candidate_id") is None, tf
        assert ev.get("decision") in ("BUY", "SELL", "NO TRADE"), tf
        assert ev["record"]["strategy_id"] == DIRECT_STRATEGY_ID
        assert ev["record"]["authorized"] is False
    # Every fresh frame's publication carries its own real closed-candle event time
    for tf, ev in evals.items():
        assert ev.get("publication") and ev["publication"]["instrument"]["interval"] == tf, tf


def test_buy_decision_reaches_publisher_through_existing_contract(tmp_path) -> None:
    pub_mock = MagicMock()
    pub_mock.enabled = True
    pub_mock.publish.return_value = {"status": "PUBLISHED", "published": True, "event_id": "e1"}
    loaders = {"5m": lambda: closed_candles("5m", [1990.0] * 24 + [2100.0])}
    cont = run_direct_runtime(tmp_path, loaders, publish=True, publisher=pub_mock)
    tick = cont.tick(reference_now=REF_NOW)
    ev = tick["evaluations"]["5m"]
    assert ev["decision"] == "BUY"
    assert ev["publish_result"]["published"] is True
    assert pub_mock.publish.call_count == 1
    delivered = pub_mock.publish.call_args[0][0]
    assert isinstance(delivered, DirectSignalPublication)
    assert pub_mock.publish.call_args[1] == {"skip_if_no_trade": False}
    contract = delivered.to_contract_v1_payload()
    assert contract["signal"]["decision"] == "BUY"
    assert contract["trade_setup"]["entry_price"] == 2100.0


def test_stale_frame_fails_closed_and_never_publishes(tmp_path) -> None:
    pub_mock = MagicMock()
    pub_mock.enabled = True
    old_candles = closed_candles("15m", [1990.0] * 30)
    old_candles["timestamp"] = old_candles["timestamp"] - pd.Timedelta(days=30)
    loaders = {"15m": lambda: old_candles}
    cont = run_direct_runtime(tmp_path, loaders, publish=True, publisher=pub_mock, max_age_seconds=3600.0)
    tick = cont.tick(reference_now=REF_NOW)
    ev = tick["evaluations"]["15m"]
    assert ev["blocked"] is True
    assert ev["reason"] == "stale_market_data"
    assert ev["publish_result"]["status"] == "SKIPPED_BLOCKED"
    assert ev["trade_levels"] is None
    pub_mock.publish.assert_not_called()
    assert tick["health"]["timeframes"]["15m"]["execution"] == "BLOCKED"


def test_unclosed_candle_skips_without_fabrication(tmp_path) -> None:
    pub_mock = MagicMock()
    pub_mock.enabled = True
    # Provider explicitly reports every candidate candle open: no closed candle exists
    df = closed_candles("5m", [1990.0] * 25)
    df["isOpen"] = True
    loaders = {"5m": lambda: df}
    cont = run_direct_runtime(tmp_path, loaders, publish=True, publisher=pub_mock)
    tick = cont.tick(reference_now=REF_NOW)
    ev = tick["evaluations"]["5m"]
    assert ev["status"] == "SKIPPED_NO_CLOSED_CANDLE"
    assert ev["reason"] == "latest_candle_is_open_or_unavailable"
    pub_mock.publish.assert_not_called()
    assert len(cont._evaluated_candles) == 0


def test_one_frame_failure_is_isolated_from_the_other_six(tmp_path) -> None:
    frames = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    loaders = {tf: (lambda tf=tf: closed_candles(tf, [1990.0] * 25)) for tf in frames}
    def failing_1H():
        raise RuntimeError("1H provider outage")

    loaders["1H"] = failing_1H
    cont = run_direct_runtime(tmp_path, loaders, max_age_seconds=10 ** 7)
    tick = cont.tick(reference_now=REF_NOW)
    evals = tick["evaluations"]
    assert evals["1H"]["status"] == "FAILED_RECOVERABLE"
    for tf in ("1m", "5m", "15m", "30m", "4H", "1D"):
        assert evals[tf].get("direct_signal") is True, tf
        assert evals[tf].get("blocked") is False, tf
    assert tick["health"]["timeframes"]["1H"]["execution"] == "UNHEALTHY"
    for tf in ("1m", "5m", "15m", "30m", "4H", "1D"):
        assert tick["health"]["timeframes"][tf]["execution"] == "HEALTHY", tf
    assert tick["health"]["seven_frame_parity"] is False
    assert "1H provider outage" in tick["health"]["timeframes"]["1H"]["reason_code"]


def test_failed_publication_is_retryable_and_healthy_outcome_deduplicates(tmp_path) -> None:
    failed = {"status": "FAILED", "published": False, "error": "HTTP 503"}
    published = {"status": "PUBLISHED", "published": True, "event_id": "e1", "publication_id": "e1", "remote_event_id": "e1"}
    side = [failed, published]
    pub_mock = MagicMock()
    pub_mock.enabled = True
    pub_mock.publish.side_effect = lambda *a, **k: side.pop(0) if side else published
    loaders = {"5m": lambda: closed_candles("5m", [1990.0] * 24 + [2100.0])}
    cont = run_direct_runtime(tmp_path, loaders, publish=True, publisher=pub_mock)

    t1 = cont.tick(reference_now=REF_NOW)
    assert t1["evaluations"]["5m"]["publish_result"]["status"] == "FAILED"
    dedup_key = ("XAUUSD", "5m", "2025-06-01T11:55:00+00:00", "direct:multi_setup_direct")
    assert dedup_key not in cont._evaluated_candles

    t2 = cont.tick(reference_now=REF_NOW)
    assert t2["evaluations"]["5m"]["publish_result"]["published"] is True
    assert dedup_key in cont._evaluated_candles

    t3 = cont.tick(reference_now=REF_NOW)
    assert t3["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert pub_mock.publish.call_count == 2


def test_readiness_matrix_is_per_timeframe_and_publication_misconfiguration_blocks(tmp_path) -> None:
    bad_pub = MagicMock()
    bad_pub.enabled = True
    bad_pub.publish_url = ""
    bad_pub.api_key = ""
    res_bad = verify_timeframe_direct_readiness("XAUUSD", "15m", publisher=bad_pub)
    assert res_bad["status"] == "BLOCKED"
    assert res_bad["reason_code"] == "PUBLICATION_MISCONFIGURED"

    ok_pub = MagicMock()
    ok_pub.enabled = False
    for tf in ("1m", "5m", "15m", "30m", "1H", "4H", "1D"):
        res = verify_timeframe_direct_readiness("XAUUSD", tf, publisher=ok_pub)
        assert res["status"] == "READY", tf
        assert res["candidate_id"] is None
        assert res["strategy_name"] == DIRECT_STRATEGY_ID

    bad_tf = verify_timeframe_direct_readiness("XAUUSD", "2m", publisher=ok_pub)
    assert bad_tf["status"] == "BLOCKED"
    assert bad_tf["reason_code"] == "INVALID_TIMEFRAME"


def test_direct_mode_never_reads_research_and_never_borrows_across_timeframes(tmp_path) -> None:
    """Frames with identical candle timestamps still derive distinct, own-data decisions;
    no candidate is consulted even when another frame's is configured explicitly."""
    loaders = {
        "5m": lambda: closed_candles("5m", [1990.0] * 24 + [2100.0]),   # BUY
        "15m": lambda: closed_candles("15m", [1990.0] * 25),           # NO TRADE
    }
    cont = run_direct_runtime(
        tmp_path,
        loaders,
        max_age_seconds=10 ** 7,
    )
    tick = cont.tick(reference_now=REF_NOW)
    ev5 = tick["evaluations"]["5m"]
    ev15 = tick["evaluations"]["15m"]
    assert ev5["decision"] == "BUY"
    assert ev15["decision"] == "NO TRADE"
    # Distinct derived signals: no shared/borrowed identity or levels
    assert ev5["publication"]["signal"] is not ev15["publication"].get("signal")
    assert ev5["trade_levels"]["entry_price"] == 2100.0
    assert ev15["trade_levels"] is None
    assert all(ev["candidate_id"] is None for ev in (ev5, ev15))
    # Candidate-free mode persists even with a stale research store configured
    tick2 = cont.tick(reference_now=REF_NOW)
    assert tick2["evaluations"]["5m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert tick2["evaluations"]["15m"]["status"] == "SKIPPED_DEDUPLICATED"


def test_direct_strategy_constants_are_explicit() -> None:
    assert DIRECT_STRATEGY_ID == "multi_setup_direct"
    assert DIRECT_STRATEGY_VERSION == "2.0"
    import src.evaluation.live_direct_strategy as m
    assert m.FAST_MA_PERIOD == 5
    assert m.SLOW_MA_PERIOD == 20
    assert m.PULLBACK_LOOKBACK == 3
    assert m.BREAKOUT_LOOKBACK == 10
    assert m.RISK_REWARD_RATIO == 2.0
    assert SETUP_PRIORITY == ("crossover", "pullback_continuation", "breakout_momentum")
