import pandas as pd
import pytest

from src.evaluation.live_trade_display import build_live_trade_display
from src.visualization.live_trade_overlay import (
    build_live_trade_overlay,
)


@pytest.fixture(autouse=True)
def mock_candidate_resolution(monkeypatch):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="5m", base_dir=None, **kwargs):
        strat = strategy_id or "momentum"
        tf = timeframe or "5m"
        cid = candidate_id or f"cand_{strat}_{tf}"
        ev = make_promoted_evidence(timeframe=tf, symbol=symbol, strategy_name=strat)
        rob = assess_research_robustness(ev)
        gov = qualify_research_evidence(ev, robustness_assessment=rob)
        params = dict(ev.spec.parameters)
        return PromotedCandidateArtifact(
            candidate_id=cid,
            strategy_name=strat,
            strategy_version="1.0",
            evidence=ev,
            symbol=symbol,
            timeframe=tf,
            operational_stability_score=0.7458282289664787,
            parameters=params,
            governance_decision=gov,
            governance_decision_fingerprint=gov.decision_fingerprint,
        )

    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", synth_resolve)
    monkeypatch.setattr("src.evaluation.live_runtime.resolve_promoted_candidate", synth_resolve)


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
        tz="UTC",
    )

    close = pd.Series(
        [2000.0 + index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _falling_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
        tz="UTC",
    )

    close = pd.Series(
        [2100.0 - index for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def _flat_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
        tz="UTC",
    )

    close = pd.Series(
        [2000.0 + 0.01 * (index % 2) for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
        }
    )


from src.evaluation.live_runtime import build_live_runtime


def test_buy_overlay_contains_all_trade_lines():
    data = _rising_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    ref_now = data["timestamp"].iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        persist=False,
    )

    overlay = build_live_trade_overlay(data, runtime_res.display)

    assert overlay["decision"] == "BUY"

    levels = overlay["levels"]

    assert levels["entry"] is not None
    assert levels["stop_loss"] is not None
    assert levels["tp1"] is not None
    assert levels["tp2"] is not None
    assert levels["tp3"] is not None

    assert levels["stop_loss"] < levels["entry"]
    assert levels["entry"] < levels["tp1"]
    assert levels["tp1"] < levels["tp2"]
    assert levels["tp2"] < levels["tp3"]


def test_buy_overlay_has_five_visible_lines():
    data = _rising_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    ref_now = data["timestamp"].iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        persist=False,
    )

    overlay = build_live_trade_overlay(data, runtime_res.display)

    assert len(overlay["lines"]) == 5
    assert all(line["visible"] for line in overlay["lines"])

    assert [line["name"] for line in overlay["lines"]] == [
        "Entry",
        "SL",
        "TP1",
        "TP2",
        "TP3",
    ]


def test_no_trade_overlay_hides_all_trade_lines():
    data = _rising_data()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        min_stability_score=0.90,
        persist=False,
    )

    overlay = build_live_trade_overlay(data, runtime_res.display)

    assert overlay["decision"] == "NO TRADE"

    assert overlay["levels"] == {
        "entry": None,
        "stop_loss": None,
        "tp1": None,
        "tp2": None,
        "tp3": None,
    }

    assert len(overlay["lines"]) == 5
    assert all(
        not line["visible"]
        for line in overlay["lines"]
    )


def test_overlay_preserves_strategy_and_stability_metadata():
    data = _rising_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    ref_now = data["timestamp"].iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        persist=False,
    )

    overlay = build_live_trade_overlay(data, runtime_res.display)

    assert overlay["stable_strategy"] == "momentum"
    assert overlay["stability_score"] == pytest.approx(0.7458282289664787)
    assert overlay["trend"] == "UP"
    assert overlay["signal_label"] == "BUY"


def test_overlay_uses_latest_data_timestamp():
    data = _rising_data()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        persist=False,
    )

    overlay = build_live_trade_overlay(data, runtime_res.display)

    assert overlay["timestamp"] == data["timestamp"].iloc[-1]


def test_invalid_decision_is_rejected():
    data = _rising_data()

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        persist=False,
    )
    display = dict(runtime_res.display)

    display["decision"] = "HOLD"

    with pytest.raises(
        ValueError,
        match="decision must be",
    ):
        build_live_trade_overlay(data, display)


def _sell_snapshot(ref_now=False):
    data = _falling_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=(
            data["timestamp"].iloc[-1].to_pydatetime()
            + pd.Timedelta(minutes=5)
            if ref_now
            else None
        ),
        persist=False,
    )
    return data, runtime_res


def test_sell_decision_overlay_is_supported_and_ordered():
    data, runtime_res = _sell_snapshot(ref_now=True)

    assert runtime_res.display["decision"] == "SELL"

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    levels = overlay["levels"]

    assert levels["entry"] is not None
    assert (
        levels["tp3"]
        < levels["tp2"]
        < levels["tp1"]
        < levels["entry"]
        < levels["stop_loss"]
    )


def test_sell_level_order_violation_is_rejected():
    data, runtime_res = _sell_snapshot(ref_now=True)
    display = dict(runtime_res.display)

    # Break SELL geometry: tp2 above tp1.
    display["tp2"] = display["tp1"] + 1.0

    with pytest.raises(
        ValueError,
        match="SELL levels must satisfy",
    ):
        build_live_trade_overlay(data, display)


def test_no_trade_decision_overlay_has_no_levels():
    data = _flat_data()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    runtime_res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=data["timestamp"].iloc[-1].to_pydatetime()
        + pd.Timedelta(minutes=5),
        persist=False,
    )

    assert runtime_res.display["decision"] == "NO TRADE"

    overlay = build_live_trade_overlay(
        data,
        runtime_res.display,
    )

    assert (
        overlay["levels"]["entry"]
        is None
    )
    assert (
        overlay["levels"]["stop_loss"]
        is None
    )
    assert (
        overlay["lines"][0]["visible"]
        is False
    )


def test_missing_timestamp_is_rejected():
    data = _rising_data().drop(columns=["timestamp"])

    runtime_res = build_live_runtime(
        _rising_data(),
        stable_strategy="momentum",
        stability_score=None,
        persist=False,
    )
    display = runtime_res.display

    with pytest.raises(
        ValueError,
        match="data must contain timestamp",
    ):
        build_live_trade_overlay(data, display)
