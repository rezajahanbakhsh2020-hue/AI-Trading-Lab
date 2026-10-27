"""Focused regression tests: a missing/unqualified candidate for one timeframe must NOT
suppress that timeframe's market evaluation or publication, nor affect any other timeframe.

Contract under test (owner directive):
- All seven canonical timeframes run independently in the continuous worker.
- Candidate resolution must not be a hard prerequisite for market evaluation and
  publishing: a timeframe without an authorized candidate still evaluates its real
  market-data slice and publishes its own first-class NO_TRADE observation with real
  timestamps and explicit provenance.
- BUY/SELL remain impossible without a qualified candidate (no fabricated signals).
- Stale/unclosed data fail closed: no observation publication for that cycle.
- Publication failures are not deduplicated and retry on the next tick.
"""

import datetime
import json
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.evaluation.live_execution_runtime import ContinuousLiveRuntime
from src.evaluation.live_market_observation import MarketObservationPublication
from src.integration.project2_publisher import Project2Publisher
from tests.test_continuous_live_runtime import persist_momentum_candidate
from tests.test_timeframe_readiness_worker_health import make_real_run_once_success


TF_DURATION_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600, "4H": 14400, "1D": 86400}
TF_PANDAS_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1H": "1h", "4H": "4h", "1D": "1d"}


def candles_ending_before(tf: str, *, last_open: datetime.datetime, count: int = 10) -> pd.DataFrame:
    """Build a candle slice for `tf` whose FINAL candle opens at `last_open` (UTC).

    Timestamps are provider candle opens; candle duration equals the exact canonical
    duration so closed/fresh boundaries are deterministic per timeframe.
    """
    assert tf in TF_DURATION_SECONDS
    opens = pd.date_range(end=last_open, periods=count, freq=TF_PANDAS_FREQ[tf], tz="UTC")
    prices = [2000.0 + (i * 2.0) for i in range(count)]
    return pd.DataFrame({
        "timestamp": opens,
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
    })


REF_NOW = datetime.datetime(2025, 1, 1, 12, 30, tzinfo=datetime.timezone.utc)


def _fake_published(event_id: str = "evt_obs_1") -> dict:
    return {
        "status": "PUBLISHED",
        "published": True,
        "http_code": 200,
        "event_id": event_id,
        "publication_id": event_id,
        "remote_event_id": event_id,
        "attempts": 1,
    }


class SequenceLoader:
    """Per-timeframe market-data loader returning prepared frames in sequence."""

    def __init__(self, *frames: pd.DataFrame) -> None:
        self._frames = list(frames) or [pd.DataFrame()]

    def __call__(self) -> pd.DataFrame:
        if len(self._frames) > 1:
            return self._frames.pop(0)
        return self._frames[0]


def _runtime(
    tmp_path,
    loaders: dict[str, Any],
    *,
    candidate_ids: dict[str, str] | None = None,
    publisher: Any | None = None,
    publish: bool = True,
    persist: bool = False,
) -> ContinuousLiveRuntime:
    return ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=sorted(loaders.keys()),
        poll_interval=0.0,
        publish=publish,
        persist=persist,
        research_dir=tmp_path,
        store_path=tmp_path / "live_history.json",
        snapshot_path=tmp_path / "latest_execution.json",
        publisher=publisher,
        candidate_ids=candidate_ids or {},
        market_data_loaders=loaders,
    )


def _enabled_publisher(events: list[dict] | None = None) -> MagicMock:
    pub = MagicMock()
    pub.enabled = True
    if events is None:
        pub.publish.return_value = _fake_published()
    else:
        pub.publish.side_effect = events
    return pub


# ---------------------------------------------------------------------------
# MarketObservationPublication artifact unit tests
# ---------------------------------------------------------------------------

def _observation_kwargs() -> dict:
    return dict(
        symbol="XAUUSD",
        timeframe="5m",
        reason="PromotionUnavailable",
        detail="No persisted promoted candidate matches candidate_id=None.",
        candle_timestamp_utc="2025-01-01T12:25:00+00:00",
        observation_timestamp_utc="2025-01-01T12:40:00+00:00",
    )


def test_observation_publication_builds_truthful_payload() -> None:
    obs = MarketObservationPublication(**_observation_kwargs())
    payload = obs.to_contract_v1_payload()

    assert payload["contract_version"] == "1.0"
    assert payload["event_type"] == "TRADING_SIGNAL"
    assert payload["timestamp"] == "2025-01-01T12:25:00Z"
    assert payload["instrument"] == {"symbol": "XAUUSD", "interval": "5m"}
    assert payload["signal"]["decision"] == "NO TRADE"
    assert payload["signal"]["signal_label"] == "NO TRADE"
    assert payload["signal"]["trend"] == "NEUTRAL"
    assert payload["signal"]["strategy"] is None
    assert payload["signal"]["candidate_id"] is None
    assert payload["signal"]["stability_score"] is None
    assert payload["trade_setup"] == {
        "entry_price": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "tp3": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "trailing_stop": None,
    }
    provenance = payload["provenance"]
    assert provenance["provenance_type"] == "market_observation_no_trade"
    assert provenance["authorized"] is False
    assert provenance["authorization_state"] == "no_authorized_candidate_for_timeframe"
    assert provenance["blocked_reason_code"] == "PromotionUnavailable"
    assert provenance["produced_at"] == "2025-01-01T12:40:00Z"
    assert obs.decision == "NO TRADE"


def test_observation_publication_id_is_deterministic() -> None:
    kwargs = _observation_kwargs()
    assert MarketObservationPublication(**kwargs).observation_id == MarketObservationPublication(**kwargs).observation_id
    changed = dict(kwargs, candle_timestamp_utc="2025-01-01T12:30:00+00:00")
    assert MarketObservationPublication(**changed).observation_id != MarketObservationPublication(**kwargs).observation_id


def test_observation_publication_is_immutable_and_always_no_trade() -> None:
    obs = MarketObservationPublication(**_observation_kwargs())
    with pytest.raises(Exception):
        obs.decision = "BUY"  # type: ignore[misc]
    assert obs.decision == "NO TRADE"


@pytest.mark.parametrize("bad_kwargs", [
    dict(candle_timestamp_utc="2025-01-01T12:25:00"),  # naive
    dict(observation_timestamp_utc="not-a-ts"),
    dict(reason=""),
    dict(detail="   "),
    dict(symbol=""),
])
def test_observation_publication_fails_closed_on_invalid_input(bad_kwargs) -> None:
    base = _observation_kwargs()
    base.update(bad_kwargs)
    with pytest.raises(ValueError):
        MarketObservationPublication(**base)


def test_publisher_accepts_observation_payload_over_http(monkeypatch) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    obs = MarketObservationPublication(
        symbol="XAUUSD",
        timeframe="5m",
        reason="PromotionUnavailable",
        detail="d",
        candle_timestamp_utc=(now - datetime.timedelta(minutes=5)).isoformat(),
        observation_timestamp_utc=now.isoformat(),
    )
    captured: dict[str, Any] = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def getcode(self):
            return 200

        def geturl(self):
            return "http://localhost/ingest"

        def read(self):
            captured["body"] = json.dumps({"status": "INGESTED", "event_id": obs.observation_id}).encode("utf-8")
            return captured["body"]

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    pub = Project2Publisher(
        publish_url="http://localhost/ingest",
        api_key="secret_key",
        enabled=True,
    )
    res = pub.publish(obs, skip_if_no_trade=False)

    assert res["status"] == "PUBLISHED"
    assert res["published"] is True
    assert res["event_id"] == obs.observation_id
    body = captured["payload"]
    assert body["signal"]["decision"] == "NO TRADE"
    assert body["timestamp"] == obs.candle_timestamp_utc
    assert body["provenance"]["authorized"] is False


def test_publisher_observation_skipped_when_no_trade_skip_configured() -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    obs = MarketObservationPublication(
        symbol="XAUUSD",
        timeframe="5m",
        reason="PromotionUnavailable",
        detail="d",
        candle_timestamp_utc=(now - datetime.timedelta(minutes=5)).isoformat(),
        observation_timestamp_utc=now.isoformat(),
    )
    pub = Project2Publisher(publish_url="http://localhost/ingest", api_key="k", enabled=True)
    res = pub.publish(obs, skip_if_no_trade=True)
    assert res == {
        "status": "SKIPPED_NO_TRADE",
        "published": False,
        "reason": "Decision is NO TRADE and skip_if_no_trade=True",
    }


def test_publisher_rejects_raw_dict_payload() -> None:
    pub = Project2Publisher(publish_url="http://localhost/ingest", api_key="k", enabled=True)
    res = pub.publish({"signal": {"decision": "NO TRADE"}})
    assert res["status"] == "INVALID_RESPONSE"


# ---------------------------------------------------------------------------
# ContinuousLiveRuntime tick-level isolation tests
# ---------------------------------------------------------------------------

def test_missing_candidate_timeframe_evaluates_and_publishes_no_trade_observation(tmp_path) -> None:
    """A timeframe with no candidate mapping still gets its own market-data slice,
    evaluation result, and publication outcome in the same tick."""
    pub_mock = _enabled_publisher()
    loaders = {"15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 10, tzinfo=datetime.timezone.utc)))}
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)
    tick_summary = cont.tick(reference_now=REF_NOW)

    res = tick_summary["evaluations"]["15m"]
    assert res.get("observation") is True
    assert res.get("blocked") is False
    assert res.get("decision") == "NO TRADE"
    assert res.get("status") == "PUBLISHED"
    assert res.get("reason") == "PromotionUnavailable"
    assert res.get("candidate_id") is None
    assert res.get("candle_timestamp") == "2025-01-01T12:10:00+00:00"
    assert res.get("decision_timestamp") == "2025-01-01T12:30:00Z"
    assert res["publish_result"]["published"] is True
    assert res["record"]["decision"] == "NO TRADE"
    assert res["record"]["candidate_id"] is None
    assert res["record"]["authorized"] is False
    assert res["publication"]["signal"]["decision"] == "NO TRADE"
    assert res["publication"]["timestamp"] == "2025-01-01T12:10:00Z"
    assert res["publication"]["provenance"]["authorized"] is False

    # The publisher received the immutable observation artifact through the single
    # publisher path, with the worker's skip_if_no_trade policy.
    assert pub_mock.publish.call_count == 1
    delivered = pub_mock.publish.call_args[0][0]
    assert isinstance(delivered, MarketObservationPublication)
    assert pub_mock.publish.call_args[1] == {"skip_if_no_trade": False}

    assert tick_summary["health"]["status"] == "WORKER_HEALTHY"
    assert tick_summary["health"]["execution_healthy"] is True


def test_unqualified_timeframe_does_not_prevent_qualified_timeframe_evaluation_and_publication(tmp_path) -> None:
    """A candidate-less 15m cannot stop the fully authorized 5m from evaluating and publishing."""
    cand_5m = persist_momentum_candidate(tmp_path, candidate_id="cand_momentum_live", timeframe="5m")

    pub_mock = _enabled_publisher()
    loaders = {
        "5m": SequenceLoader(candles_ending_before("5m", last_open=datetime.datetime(2025, 1, 1, 12, 25, tzinfo=datetime.timezone.utc))),
        "15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 10, tzinfo=datetime.timezone.utc))),
    }
    cont = _runtime(tmp_path, loaders, candidate_ids={"5m": cand_5m}, publisher=pub_mock, persist=True)
    tick_summary = cont.tick(reference_now=REF_NOW)

    evals = tick_summary["evaluations"]
    res_5m = evals["5m"]
    res_15m = evals["15m"]

    # 5m evaluated on its own slice through the full authorized path and published independently
    assert res_5m.get("blocked") is False
    assert res_5m.get("candidate_id") == cand_5m
    assert res_5m.get("symbol") == "XAUUSD"
    assert res_5m.get("interval") == "5m"
    assert isinstance(res_5m.get("publish_result"), dict)

    # 15m truthful NO_TRADE observation, independently produced and published
    assert res_15m.get("observation") is True
    assert res_15m.get("blocked") is False
    assert res_15m.get("decision") == "NO TRADE"
    assert res_15m.get("candidate_id") is None
    assert res_15m.get("publish_result", {}).get("published") is True

    # Both timeframes reached the single publisher path
    assert pub_mock.publish.call_count == 2

    assert tick_summary["health"]["status"] == "WORKER_HEALTHY"
    assert tick_summary["health"]["execution_healthy"] is True
    diag = tick_summary["health"]["timeframes"]["15m"]
    assert diag["execution"] == "HEALTHY"
    assert diag["readiness"] in ("UNKNOWN", "BLOCKED")
    # The 15m readiness blocker remains honestly reported
    assert "PromotionUnavailable" in str(diag["reason_code"])


def test_unqualified_timeframe_with_stale_data_fails_closed_without_publication(tmp_path) -> None:
    """Stale market data must never become a releasable observation: blocked and unpublished."""
    pub_mock = _enabled_publisher()
    # Last candle closes 12:00 -> ~30 minutes stale relative to REF_NOW for the 15m slice.
    loaders = {"15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 11, 45, tzinfo=datetime.timezone.utc)))}
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)
    tick_summary = cont.tick(reference_now=REF_NOW)

    res = tick_summary["evaluations"]["15m"]
    assert res.get("blocked") is True
    assert res.get("status") == "BLOCKED"
    assert res.get("reason") == "stale_market_data"
    assert res.get("observation") is None
    assert res.get("blocked_state", {}).get("reason") == "PromotionUnavailable"
    assert res.get("blocked_state", {}).get("candidate_id") is None
    assert res.get("blocked_state", {}).get("market_data_fresh") is False
    pub_mock.publish.assert_not_called()

    assert tick_summary["health"]["status"] == "WORKER_DEGRADED"


def test_unqualified_timeframe_without_any_closed_candle_skips_without_publication(tmp_path) -> None:
    """No closed candle at all -> benign per-timeframe skip, no fabricated evaluation or publication."""
    pub_mock = _enabled_publisher()
    # Single candle that opened 4 minutes ago: not yet closed for a 15m timeframe.
    loaders = {"15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 15, tzinfo=datetime.timezone.utc), count=1))}
    early_ref = datetime.datetime(2025, 1, 1, 12, 19, tzinfo=datetime.timezone.utc)
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)
    tick_summary = cont.tick(reference_now=early_ref)

    res = tick_summary["evaluations"]["15m"]
    assert res.get("status") == "SKIPPED_NO_CLOSED_CANDLE"
    assert res.get("reason") == "latest_candle_is_open_or_unavailable"
    pub_mock.publish.assert_not_called()


def test_observation_publication_failure_retries_next_tick_and_then_deduplicates(tmp_path) -> None:
    """Failed observation publication is NOT deduplicated; healthy outcome is."""
    failed_pub = {"status": "FAILED", "published": False, "error": "HTTP 503"}
    pub_mock = _enabled_publisher(events=[
        failed_pub,
        _fake_published(),
        _fake_published(),
    ])
    loaders = {"15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 10, tzinfo=datetime.timezone.utc)))}
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)

    tick1 = cont.tick(reference_now=REF_NOW)
    res1 = tick1["evaluations"]["15m"]
    assert res1["publish_result"]["status"] == "FAILED"
    assert tick1["health"]["status"] == "WORKER_DEGRADED"

    dedup_key = ("XAUUSD", "15m", "2025-01-01T12:10:00+00:00", None)
    assert dedup_key not in cont._evaluated_candles

    tick2 = cont.tick(reference_now=REF_NOW)
    res2 = tick2["evaluations"]["15m"]
    assert res2["publish_result"]["published"] is True
    assert tick2["health"]["status"] == "WORKER_HEALTHY"
    assert dedup_key in cont._evaluated_candles
    assert pub_mock.publish.call_count == 2

    tick3 = cont.tick(reference_now=REF_NOW)
    assert tick3["evaluations"]["15m"]["status"] == "SKIPPED_DEDUPLICATED"
    assert tick3["evaluations"]["15m"]["reason"] == "candle_already_evaluated"
    assert tick3["health"]["status"] == "WORKER_HEALTHY"
    assert pub_mock.publish.call_count == 2


def test_observation_dedup_key_is_per_timeframe_and_candle(tmp_path) -> None:
    """A deduplicated observation does not suppress a different candle."""
    pub_mock = _enabled_publisher()
    loader_1 = candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 10, tzinfo=datetime.timezone.utc))
    loader_2 = candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 20, tzinfo=datetime.timezone.utc))
    loaders = {"15m": SequenceLoader(loader_1, loader_2)}
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)

    cont.tick(reference_now=REF_NOW)  # first candle (open 12:10) closed 12:25
    cont.tick(reference_now=datetime.datetime(2025, 1, 1, 12, 40, tzinfo=datetime.timezone.utc))  # second candle (open 12:20) closes 12:35

    dedup_keys = cont._evaluated_candles
    assert ("XAUUSD", "15m", "2025-01-01T12:10:00+00:00", None) in dedup_keys
    assert ("XAUUSD", "15m", "2025-01-01T12:20:00+00:00", None) in dedup_keys
    assert len(dedup_keys) == 2


def test_multiple_candidate_less_timeframes_are_independently_attempted(tmp_path) -> None:
    """All configured timeframes produce their own evaluation and publication attempts."""
    pub_mock = _enabled_publisher()
    loaders = {
        "15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 10, tzinfo=datetime.timezone.utc))),
        "30m": SequenceLoader(candles_ending_before("30m", last_open=datetime.datetime(2025, 1, 1, 12, 0, tzinfo=datetime.timezone.utc))),
    }
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)
    tick_summary = cont.tick(reference_now=REF_NOW)

    assert tick_summary["evaluations"]["15m"].get("observation") is True
    assert tick_summary["evaluations"]["30m"].get("observation") is True
    pub_mock.publish.call_count == 2
    candle_keys = [key for key in cont._evaluated_candles]
    assert {k[1] for k in candle_keys} == {"15m", "30m"}
    assert tick_summary["health"]["status"] == "WORKER_HEALTHY"


def test_startup_preflight_blocked_timeframes_are_still_attempted_each_tick(tmp_path) -> None:
    """A BLOCKED startup readiness preflight result must not suppress tick-level
    evaluation or publication for that timeframe."""
    # No candidate at all -> all configured timeframes BLOCKED in preflight.
    pub_mock = _enabled_publisher()
    loaders = {"15m": SequenceLoader(candles_ending_before("15m", last_open=datetime.datetime(2025, 1, 1, 12, 10, tzinfo=datetime.timezone.utc)))}
    cont = _runtime(tmp_path, loaders, publisher=pub_mock)
    cont._timeframe_readiness = {
        "15m": {"status": "BLOCKED", "reason_code": "PromotionUnavailable", "candidate_id": None},
    }

    tick_summary = cont.tick(reference_now=REF_NOW)
    res = tick_summary["evaluations"]["15m"]
    assert res.get("observation") is True
    assert res.get("blocked") is False
    assert res.get("publish_result", {}).get("published") is True


# ---------------------------------------------------------------------------
# Health classifier: observation result validation
# ---------------------------------------------------------------------------

def _health_runtime():
    now_dt = REF_NOW
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=["15m"], clock=lambda: now_dt)
    runtime._started_at_utc = now_dt.isoformat()
    runtime._last_tick_at_utc = now_dt.isoformat()
    return runtime


def _observation_result(published=True, pub_status="PUBLISHED", decision="NO TRADE", **overrides):
    pub_res: dict[str, Any] = {"status": pub_status, "published": published}
    if published is False and pub_status == "SKIPPED_DISABLED":
        pub_res["reason"] = "Publisher disabled in configuration"
    res: dict[str, Any] = {
        "observation": True,
        "blocked": False,
        "status": pub_status,
        "decision": decision,
        "reason": "PromotionUnavailable",
        "detail": "No authorized production candidate for XAUUSD 15m.",
        "symbol": "XAUUSD",
        "interval": "15m",
        "timeframe": "15m",
        "candle_timestamp": "2025-01-01T12:10:00+00:00",
        "decision_timestamp": REF_NOW.isoformat(),
        "record": {"observation_id": "obs123", "decision": "NO TRADE", "symbol": "XAUUSD"},
        "publication": {"contract_version": "1.0", "signal": {"decision": "NO TRADE"}},
        "publish_result": pub_res,
    }
    res.update(overrides)
    return res


def test_health_accepts_published_observation_result() -> None:
    runtime = _health_runtime()
    h = runtime.get_health_status(reference_now=REF_NOW, current_evaluations={"15m": _observation_result()})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True


def test_health_accepts_clean_skip_observation_results() -> None:
    runtime = _health_runtime()
    h = runtime.get_health_status(
        reference_now=REF_NOW,
        current_evaluations={"15m": _observation_result(published=False, pub_status="SKIPPED_DISABLED")},
    )
    assert h["status"] == "WORKER_HEALTHY"


@pytest.mark.parametrize("pub_status", ["FAILED", "REJECTED", "AUTH_FAILED", "TIMED_OUT", "INVALID_RESPONSE", "SKIPPED_BLOCKED"])
def test_health_rejects_observation_results_with_publication_failures(pub_status) -> None:
    runtime = _health_runtime()
    h = runtime.get_health_status(
        reference_now=REF_NOW,
        current_evaluations={"15m": _observation_result(published=False, pub_status=pub_status)},
    )
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_health_rejects_observation_with_trade_decision() -> None:
    runtime = _health_runtime()
    h = runtime.get_health_status(
        reference_now=REF_NOW,
        current_evaluations={"15m": _observation_result(decision="BUY")},
    )
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_health_rejects_observation_with_missing_publication_evidence() -> None:
    runtime = _health_runtime()
    h = runtime.get_health_status(
        reference_now=REF_NOW,
        current_evaluations={"15m": _observation_result(publication=None)},
    )
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_health_rejects_observation_with_contradictory_publication_flags() -> None:
    runtime = _health_runtime()
    h = runtime.get_health_status(
        reference_now=REF_NOW,
        current_evaluations={"15m": _observation_result(published=True, pub_status="SKIPPED_DISABLED")},
    )
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_health_rejects_observation_that_claims_candidate_identity() -> None:
    runtime = _health_runtime()
    runtime._timeframe_readiness = {
        "15m": {"status": "READY", "candidate_id": "cand_15m", "reason_code": "READY"},
    }
    h = runtime.get_health_status(
        reference_now=REF_NOW,
        current_evaluations={"15m": _observation_result(candidate_id="cand_15m")},
    )
    # An observation must never carry candidate identity or authorization.
    assert h["status"] == "WORKER_DEGRADED"
    assert h["execution_healthy"] is False


def test_observation_results_do_not_break_seven_frame_parity_semantics() -> None:
    all_tfs = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
    runtime = ContinuousLiveRuntime(symbol="XAUUSD", timeframes=all_tfs, clock=lambda: REF_NOW)
    runtime._started_at_utc = REF_NOW.isoformat()
    runtime._last_tick_at_utc = REF_NOW.isoformat()

    evals = {tf: _observation_result() for tf in all_tfs}
    h = runtime.get_health_status(reference_now=REF_NOW, current_evaluations=evals)

    # Observations are per-timeframe healthy operations, but candidate-less readiness
    # means seven-frame CANDIDATE parity stays False by design.
    assert h["seven_frame_parity"] is False
    assert h["execution_healthy"] is True


def test_real_run_once_results_remain_healthy_after_observation_support() -> None:
    """Existing canonical evaluation health semantics are unchanged."""
    runtime = _health_runtime()
    res = make_real_run_once_success()
    h = runtime.get_health_status(reference_now=REF_NOW, current_evaluations={"5m": res})
    assert h["status"] == "WORKER_HEALTHY"
    assert h["execution_healthy"] is True
