from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.live_release_gate import (
    validate_live_release,
)
from src.evaluation.live_runtime import build_live_runtime


@pytest.fixture(autouse=True)
def mock_candidate_resolution(monkeypatch):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact

    def synth_resolve(candidate_id=None, strategy_id=None, symbol="XAUUSD", timeframe="1d", base_dir=None, **kwargs):
        strat = strategy_id or "momentum"
        tf = timeframe or "1d"
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
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=rows),
        periods=rows,
        freq="1D",
    )

    close = pd.Series(
        [2000.0 + index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def _falling_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=rows),
        periods=rows,
        freq="1D",
    )

    close = pd.Series(
        [2160.0 - index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close + 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def _stale_data(rows: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="1D",
        tz="UTC",
    )

    close = pd.Series(
        [2000.0 + index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "isOpen": False,
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def _runtime(store_path=None):
    data = _rising_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()
    return build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
        store_path=store_path,
    )


def _falling_runtime(store_path=None):
    data = _falling_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()
    return build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
        store_path=store_path,
    )


def _stale_runtime(store_path=None):
    data = _stale_data()
    return build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        store_path=store_path,
    )


def test_release_gate_passes_complete_buy_runtime(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    result = validate_live_release(
        decision=runtime.decision,
        display=runtime.display,
    )

    assert result["release_ready"] is True
    assert all(result["checks"].values())
    assert result["decision"] == "BUY"
    assert result["stable_strategy"] == "momentum"


def test_release_gate_rejects_low_stability(tmp_path):
    data = _rising_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()
    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        min_stability_score=0.90,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    result = validate_live_release(
        decision=runtime.decision,
        display=runtime.display,
        min_stability_score=0.90,
    )

    assert result["release_ready"] is False
    assert result["checks"]["stability_threshold"] is False


def test_release_gate_accepts_no_trade_with_valid_levels_state(tmp_path):
    data = _rising_data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime()
    runtime = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="1d",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    decision = dict(runtime.decision)
    display = dict(runtime.display)

    decision["decision"] = "NO TRADE"
    display["decision"] = "NO TRADE"

    for field in (
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "tp3",
    ):
        decision.pop(field, None)
        display[field] = None

    result = validate_live_release(
        decision=decision,
        display=display,
    )

    assert result["release_ready"] is True


def test_release_gate_rejects_decision_display_mismatch(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    display = dict(runtime.display)
    display["decision"] = "NO TRADE"

    result = validate_live_release(
        decision=runtime.decision,
        display=display,
    )

    assert result["release_ready"] is False
    assert result["checks"]["decision_display_match"] is False


def test_release_gate_rejects_wrong_strategy(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    decision = dict(runtime.decision)
    display = dict(runtime.display)

    decision["stable_strategy"] = "moving_average"
    display["stable_strategy"] = "moving_average"

    result = validate_live_release(
        decision=decision,
        display=display,
        required_strategy="momentum",
    )

    assert result["release_ready"] is False
    assert result["checks"]["required_strategy"] is False


def test_release_gate_rejects_invalid_trade_level_order(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    display = dict(runtime.display)
    display["tp1"] = display["entry_price"] - 1.0

    result = validate_live_release(
        decision=runtime.decision,
        display=display,
    )

    assert result["release_ready"] is False
    assert result["checks"]["trade_levels_valid"] is False


def test_release_gate_requires_decision_fields(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    decision = dict(runtime.decision)
    decision.pop("trend")

    with pytest.raises(
        ValueError,
        match="Missing decision fields",
    ):
        validate_live_release(
            decision=decision,
            display=runtime.display,
        )


def test_release_gate_requires_display_fields(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    display = dict(runtime.display)
    display.pop("tp3")

    with pytest.raises(
        ValueError,
        match="Missing display fields",
    ):
        validate_live_release(
            decision=runtime.decision,
            display=display,
        )


def test_release_gate_passes_complete_sell_runtime(tmp_path):
    runtime = _falling_runtime(store_path=tmp_path / "store_sell.json")

    result = validate_live_release(
        decision=runtime.decision,
        display=runtime.display,
    )

    assert result["decision"] == "SELL"
    assert result["release_ready"] is True
    assert all(result["checks"].values())

    levels = runtime.display
    assert (
        levels["tp3"]
        < levels["tp2"]
        < levels["tp1"]
        < levels["entry_price"]
        < levels["stop_loss"]
    )


def test_release_gate_rejects_broken_sell_level_order(tmp_path):
    runtime = _falling_runtime(store_path=tmp_path / "store_sell.json")

    display = dict(runtime.display)
    display["tp2"] = display["entry_price"] + 1.0

    result = validate_live_release(
        decision=runtime.decision,
        display=display,
    )

    assert result["release_ready"] is False
    assert result["checks"]["trade_levels_valid"] is False


def test_release_gate_blocks_stale_no_trade():
    """Stale market data must fail closed even for a NO TRADE outcome:
    NO TRADE must never be a vehicle that launders stale or invalid
    market data into a releasable state."""
    runtime = _stale_runtime()

    decision = dict(runtime.decision)
    assert decision["decision"] == "NO TRADE"
    assert decision["quote_stale"] is True

    result = validate_live_release(
        decision=decision,
        display=runtime.display,
    )

    assert result["release_ready"] is False
    assert result["checks"]["market_data_fresh"] is False


def test_release_gate_blocks_stale_buy():
    """If the data is stale, no outcome — BUY included — may be releasable."""
    runtime = _runtime(store_path=tmp_path_store())

    decision = dict(runtime.decision)
    display = dict(runtime.display)

    assert decision["decision"] == "BUY"

    # Force the fail-closed interlock: same BUY payload on stale inputs.
    decision["quote_stale"] = True

    result = validate_live_release(
        decision=decision,
        display=display,
    )

    assert result["release_ready"] is False


def tmp_path_store():
    import tempfile
    import pathlib

    return pathlib.Path(tempfile.mkdtemp()) / "store_stale.json"


def test_release_gate_rejects_no_trade_with_fabricated_levels(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    decision = dict(runtime.decision)
    display = dict(runtime.display)

    decision["decision"] = "NO TRADE"
    display["decision"] = "NO TRADE"

    # NO TRADE must not carry active trade levels.
    display["stop_loss"] = 1900.0

    result = validate_live_release(
        decision=decision,
        display=display,
    )

    assert result["release_ready"] is False
    assert result["checks"]["trade_levels_valid"] is False


def test_release_gate_rejects_unsupported_decision(tmp_path):
    runtime = _runtime(store_path=tmp_path / "store.json")

    decision = dict(runtime.decision)
    display = dict(runtime.display)

    decision["decision"] = "HOLD"
    display["decision"] = "HOLD"

    result = validate_live_release(
        decision=decision,
        display=display,
    )

    assert result["release_ready"] is False
    assert result["checks"]["decision_supported"] is False
