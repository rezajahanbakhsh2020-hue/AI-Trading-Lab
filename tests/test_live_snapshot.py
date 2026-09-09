import json

import pytest

import live_snapshot


def make_system_output():
    return {
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "momentum": 0.02,
        "entry_price": 3500.0,
        "stop_loss": 3465.0,
        "take_profit": 3570.0,
        "risk_reward_ratio": 2.0,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "momentum_window": 10,
        "fast_window": 20,
        "slow_window": 50,
        "timestamp": "2026-01-01T00:25:00+00:00",
    }


def test_build_live_snapshot_normalizes_system_output():
    result = live_snapshot.build_live_snapshot(
        make_system_output(),
        symbol="XAUUSD",
        interval="5m",
    )

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["signal"] == 1
    assert result["signal_label"] == "BUY"
    assert result["trend"] == "UP"
    assert result["entry_price"] == 3500.0
    assert result["stop_loss"] == 3465.0
    assert result["take_profit"] == 3570.0
    assert result["risk_reward_ratio"] == 2.0
    assert result["timestamp"] == (
        "2026-01-01T00:25:00+00:00"
    )


def test_build_live_snapshot_preserves_no_trade_risk_levels():
    output = make_system_output()

    output["signal"] = 0
    output["signal_label"] = "NO TRADE"
    output["trend"] = "DOWN"
    output["stop_loss"] = None
    output["take_profit"] = None
    output["risk_reward_ratio"] = None

    result = live_snapshot.build_live_snapshot(output)

    assert result["signal"] == 0
    assert result["signal_label"] == "NO TRADE"
    assert result["trend"] == "DOWN"
    assert result["stop_loss"] is None
    assert result["take_profit"] is None
    assert result["risk_reward_ratio"] is None


def test_save_and_load_live_snapshot(tmp_path):
    snapshot = live_snapshot.build_live_snapshot(
        make_system_output()
    )

    path = tmp_path / "live_snapshot.json"

    saved_path = live_snapshot.save_live_snapshot(
        snapshot,
        path,
    )

    assert saved_path == path
    assert path.exists()

    loaded = live_snapshot.load_live_snapshot(path)

    assert loaded == snapshot


def test_saved_snapshot_is_valid_json(tmp_path):
    snapshot = live_snapshot.build_live_snapshot(
        make_system_output()
    )

    path = tmp_path / "live_snapshot.json"

    live_snapshot.save_live_snapshot(
        snapshot,
        path,
    )

    parsed = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert parsed["symbol"] == "XAUUSD"
    assert parsed["signal_label"] == "BUY"


def test_save_live_snapshot_creates_parent_directory(tmp_path):
    snapshot = live_snapshot.build_live_snapshot(
        make_system_output()
    )

    path = (
        tmp_path
        / "nested"
        / "live"
        / "snapshot.json"
    )

    live_snapshot.save_live_snapshot(
        snapshot,
        path,
    )

    assert path.exists()


def test_build_live_snapshot_rejects_missing_fields():
    output = make_system_output()
    del output["take_profit"]

    with pytest.raises(
        ValueError,
        match="Missing required system output fields",
    ):
        live_snapshot.build_live_snapshot(output)


def test_build_live_snapshot_rejects_invalid_input():
    with pytest.raises(
        ValueError,
        match="system_output must be a mapping",
    ):
        live_snapshot.build_live_snapshot([])


def test_load_live_snapshot_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid snapshot JSON",
    ):
        live_snapshot.load_live_snapshot(path)


def test_load_live_snapshot_rejects_non_object_json(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(
        "[1, 2, 3]",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain an object",
    ):
        live_snapshot.load_live_snapshot(path)
