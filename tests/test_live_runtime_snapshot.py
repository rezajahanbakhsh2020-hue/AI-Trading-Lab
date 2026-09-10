import pandas as pd
import pytest

from src.evaluation import live_runtime_snapshot as snapshot_module
from src.evaluation.live_runtime_snapshot import (
    LiveRuntimeSnapshot,
    build_live_runtime_snapshot,
    live_runtime_snapshot_dict,
    live_runtime_snapshot_message,
)


def _market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=10,
                freq="D",
            ),
            "open": [
                2600.0,
                2610.0,
                2620.0,
                2630.0,
                2640.0,
                2650.0,
                2660.0,
                2670.0,
                2680.0,
                2690.0,
            ],
            "high": [
                2610.0,
                2620.0,
                2630.0,
                2640.0,
                2650.0,
                2660.0,
                2670.0,
                2680.0,
                2690.0,
                2700.0,
            ],
            "low": [
                2590.0,
                2600.0,
                2610.0,
                2620.0,
                2630.0,
                2640.0,
                2650.0,
                2660.0,
                2670.0,
                2680.0,
            ],
            "close": [
                2605.0,
                2615.0,
                2625.0,
                2635.0,
                2645.0,
                2655.0,
                2665.0,
                2675.0,
                2685.0,
                2695.0,
            ],
        }
    )


def _allowed_controller():
    return {
        "runtime_allowed": True,
        "status": "READY",
        "strategy": "momentum",
        "readiness": {
            "strategy_stability_score": 0.90,
        },
        "runtime_gate": {
            "runtime_ready": True,
            "status": "READY",
            "failed_gates": [],
        },
    }


def test_snapshot_builds(
    monkeypatch,
):
    monkeypatch.setattr(
        snapshot_module,
        "build_live_runtime_controller",
        lambda *args, **kwargs: (
            type(
                "Controller",
                (),
                {
                    "production_decision": (
                        _allowed_controller()
                    ),
                    "controller_ready": True,
                    "session": None,
                },
            )()
        ),
    )

    snapshot = build_live_runtime_snapshot(
        _market_data(),
        0.90,
    )

    assert isinstance(
        snapshot,
        LiveRuntimeSnapshot,
    )
    assert snapshot.symbol == "XAUUSD"
    assert snapshot.interval == "5m"
    assert snapshot.strategy == "momentum"
    assert snapshot.controller_ready is True
    assert snapshot.session_ready is False
    assert snapshot.timestamp


def test_snapshot_blocks_bad_market_data(
    monkeypatch,
):
    monkeypatch.setattr(
        snapshot_module,
        "build_live_runtime_controller",
        lambda *args, **kwargs: (
            type(
                "Controller",
                (),
                {
                    "production_decision": (
                        _allowed_controller()
                    ),
                    "controller_ready": False,
                    "session": type(
                        "Session",
                        (),
                        {
                            "session_ready": False,
                            "preflight": {
                                "failed_checks": [
                                    "market_data"
                                ]
                            },
                            "runtime": type(
                                "Runtime",
                                (),
                                {
                                    "decision": {},
                                    "display": {},
                                },
                            )(),
                        },
                    )(),
                },
            )()
        ),
    )

    snapshot = build_live_runtime_snapshot(
        _market_data(),
        0.90,
        market_data_ready=False,
    )

    assert snapshot.controller_ready is False
    assert snapshot.session_ready is False
    assert "market_data" in (
        snapshot.failed_checks
    )


def test_snapshot_dict_is_serializable(
    monkeypatch,
):
    monkeypatch.setattr(
        snapshot_module,
        "build_live_runtime_controller",
        lambda *args, **kwargs: (
            type(
                "Controller",
                (),
                {
                    "production_decision": (
                        _allowed_controller()
                    ),
                    "controller_ready": True,
                    "session": None,
                },
            )()
        ),
    )

    snapshot = build_live_runtime_snapshot(
        _market_data(),
        0.90,
    )

    result = live_runtime_snapshot_dict(
        snapshot
    )

    assert isinstance(result, dict)
    assert result["symbol"] == "XAUUSD"
    assert result["strategy"] == "momentum"
    assert result["controller_ready"] is True


def test_snapshot_message_ready(
    monkeypatch,
):
    monkeypatch.setattr(
        snapshot_module,
        "build_live_runtime_controller",
        lambda *args, **kwargs: (
            type(
                "Controller",
                (),
                {
                    "production_decision": (
                        _allowed_controller()
                    ),
                    "controller_ready": True,
                    "session": None,
                },
            )()
        ),
    )

    snapshot = build_live_runtime_snapshot(
        _market_data(),
        0.90,
    )

    message = live_runtime_snapshot_message(
        snapshot
    )

    assert message.startswith(
        "LIVE RUNTIME SNAPSHOT READY:"
    )
    assert "momentum" in message


def test_snapshot_message_blocked():
    snapshot = LiveRuntimeSnapshot(
        timestamp="2025-01-01T00:00:00+00:00",
        symbol="XAUUSD",
        interval="5m",
        strategy="momentum",
        stability_score=0.40,
        controller_ready=False,
        session_ready=False,
        decision=None,
        trend=None,
        entry=None,
        stop_loss=None,
        take_profit_1=None,
        take_profit_2=None,
        take_profit_3=None,
        failed_gates=("strategy_stability",),
        failed_checks=("quote",),
    )

    message = live_runtime_snapshot_message(
        snapshot
    )

    assert message.startswith(
        "LIVE RUNTIME SNAPSHOT BLOCKED:"
    )
    assert "strategy_stability" in message
    assert "quote" in message


def test_snapshot_rejects_non_dataframe():
    with pytest.raises(TypeError):
        build_live_runtime_snapshot(
            [],
            0.90,
        )


def test_snapshot_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        build_live_runtime_snapshot(
            pd.DataFrame(),
            0.90,
        )


def test_snapshot_dict_rejects_wrong_type():
    with pytest.raises(TypeError):
        live_runtime_snapshot_dict({})


def test_snapshot_message_rejects_wrong_type():
    with pytest.raises(TypeError):
        live_runtime_snapshot_message({})
