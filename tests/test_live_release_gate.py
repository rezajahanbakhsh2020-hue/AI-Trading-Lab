from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.live_release_gate import (
    validate_live_release,
)
from src.evaluation.live_runtime import build_live_runtime


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
