import pandas as pd
import pytest

from src.evaluation import live_runtime_controller as controller_module
from src.evaluation.live_runtime_controller import (
    build_live_runtime_controller,
    is_live_runtime_controller_ready,
    live_runtime_controller_message,
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


def _allowed_production_decision():
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


def _blocked_production_decision():
    return {
        "runtime_allowed": False,
        "status": "BLOCKED",
        "strategy": None,
        "readiness": {
            "strategy_stability_score": None,
        },
        "runtime_gate": {
            "runtime_ready": False,
            "status": "BLOCKED",
            "failed_gates": [
                "strategy_stability"
            ],
        },
    }


def test_controller_builds_when_runtime_is_allowed(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _allowed_production_decision()
        ),
    )

    controller = build_live_runtime_controller(
        _market_data(),
        0.90,
    )

    assert controller.production_decision[
        "runtime_allowed"
    ] is True
    assert controller.session is not None
    assert isinstance(
        controller.controller_ready,
        bool,
    )


def test_controller_blocks_when_production_runtime_is_blocked(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _blocked_production_decision()
        ),
    )

    controller = build_live_runtime_controller(
        _market_data(),
        0.40,
    )

    assert controller.session is None
    assert controller.controller_ready is False


def test_controller_blocks_bad_market_data(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _allowed_production_decision()
        ),
    )

    controller = build_live_runtime_controller(
        _market_data(),
        0.90,
        market_data_ready=False,
    )

    assert controller.controller_ready is False
    assert controller.session is not None
    assert "market_data" in controller.session.preflight[
        "failed_checks"
    ]


def test_controller_blocks_bad_quote(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _allowed_production_decision()
        ),
    )

    controller = build_live_runtime_controller(
        _market_data(),
        0.90,
        quote_ready=False,
    )

    assert controller.controller_ready is False
    assert "quote" in controller.session.preflight[
        "failed_checks"
    ]


def test_controller_blocks_bad_timestamp(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _allowed_production_decision()
        ),
    )

    controller = build_live_runtime_controller(
        _market_data(),
        0.90,
        timestamp_ready=False,
    )

    assert controller.controller_ready is False
    assert "timestamp" in controller.session.preflight[
        "failed_checks"
    ]


def test_controller_ready_returns_boolean(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _allowed_production_decision()
        ),
    )

    result = is_live_runtime_controller_ready(
        _market_data(),
        0.90,
    )

    assert isinstance(result, bool)


def test_controller_message_ready(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _allowed_production_decision()
        ),
    )

    message = live_runtime_controller_message(
        _market_data(),
        0.90,
    )

    assert message == (
        "LIVE RUNTIME CONTROLLER READY: "
        "momentum"
    )


def test_controller_message_reports_gate_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        controller_module,
        "build_production_runtime_decision",
        lambda *args, **kwargs: (
            _blocked_production_decision()
        ),
    )

    message = live_runtime_controller_message(
        _market_data(),
        0.40,
    )

    assert message.startswith(
        "LIVE RUNTIME CONTROLLER BLOCKED:"
    )
    assert "strategy_stability" in message


def test_controller_rejects_non_dataframe():
    with pytest.raises(TypeError):
        build_live_runtime_controller(
            [],
            0.90,
        )


def test_controller_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        build_live_runtime_controller(
            pd.DataFrame(),
            0.90,
        )
