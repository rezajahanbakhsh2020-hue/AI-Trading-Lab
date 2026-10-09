import datetime
import pandas as pd

from src.evaluation.live_execution_runtime import validate_market_data_freshness


def _frame(ts: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "timestamp": pd.Timestamp(ts),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
        }]
    )


def _now(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_15m_signal_is_not_blocked_by_300_second_legacy_ttl():
    result = validate_market_data_freshness(
        _frame("2026-10-08T18:15:00Z"),
        timeframe="15m",
        max_age_seconds=300,
        reference_now=_now("2026-10-08T18:30:00Z"),
    )
    assert result["fresh"] is True
    assert result["stale"] is False


def test_1h_signal_is_not_blocked_by_300_second_legacy_ttl():
    result = validate_market_data_freshness(
        _frame("2026-10-08T17:00:00Z"),
        timeframe="1H",
        max_age_seconds=300,
        reference_now=_now("2026-10-08T18:00:00Z"),
    )
    assert result["fresh"] is True
    assert result["stale"] is False


def test_4h_signal_is_not_blocked_by_300_second_legacy_ttl():
    result = validate_market_data_freshness(
        _frame("2026-10-08T14:00:00Z"),
        timeframe="4H",
        max_age_seconds=300,
        reference_now=_now("2026-10-08T18:00:00Z"),
    )
    assert result["fresh"] is True
    assert result["stale"] is False


def test_1d_signal_is_not_blocked_by_300_second_legacy_ttl():
    result = validate_market_data_freshness(
        _frame("2026-10-08T00:00:00Z"),
        timeframe="1D",
        max_age_seconds=300,
        reference_now=_now("2026-10-09T00:00:00Z"),
    )
    assert result["fresh"] is True
    assert result["stale"] is False


def test_unclosed_candle_is_rejected_and_old_closed_candle_is_accepted():
    unclosed = validate_market_data_freshness(
        _frame("2026-10-08T17:30:00Z"),
        timeframe="1H",
        reference_now=_now("2026-10-08T18:00:00Z"),
    )
    assert unclosed["fresh"] is False
    assert unclosed["reason"] == "unclosed_market_data"

    result = validate_market_data_freshness(
        _frame("2026-10-08T10:00:00Z"),
        timeframe="1H",
        reference_now=_now("2026-10-08T18:00:00Z"),
    )
    assert result["fresh"] is True
    assert result["reason"] == "fresh"


def test_all_canonical_timeframes_use_their_exact_candle_close_boundary():
    cases = [
        ("5m", 5 * 60),
        ("15m", 15 * 60),
        ("30m", 30 * 60),
        ("1H", 60 * 60),
        ("4H", 4 * 60 * 60),
        ("1D", 24 * 60 * 60),
    ]
    opened = _now("2026-10-01T00:00:00Z")
    for timeframe, duration in cases:
        candle = _frame(opened.isoformat())
        before_close = validate_market_data_freshness(
            candle,
            timeframe=timeframe,
            reference_now=opened + datetime.timedelta(seconds=duration - 1),
        )
        at_close = validate_market_data_freshness(
            candle,
            timeframe=timeframe,
            reference_now=opened + datetime.timedelta(seconds=duration),
        )
        assert before_close["fresh"] is False, timeframe
        assert before_close["reason"] == "unclosed_market_data", timeframe
        assert at_close["fresh"] is True, timeframe
        assert at_close["timeframe"] == timeframe, timeframe


def test_provider_open_candle_state_overrides_elapsed_time():
    frame = _frame("2026-10-01T00:00:00Z")
    frame["isOpen"] = True
    result = validate_market_data_freshness(
        frame,
        timeframe="5m",
        reference_now=_now("2026-10-01T01:00:00Z"),
    )
    assert result["fresh"] is False
    assert result["reason"] == "unclosed_market_data"
