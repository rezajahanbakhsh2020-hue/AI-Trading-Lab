import pandas as pd
import pytest

from src.evaluation.live_runtime_session import (
    build_live_runtime_session,
    is_live_runtime_session_ready,
    live_runtime_session_message,
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


def test_live_runtime_session_builds(tmp_path):
    session = build_live_runtime_session(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        store_path=tmp_path / "store.json",
    )

    assert session.runtime is not None
    assert session.preflight is not None
    assert isinstance(
        session.session_ready,
        bool,
    )


def test_live_runtime_session_blocks_bad_market_data(tmp_path):
    session = build_live_runtime_session(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        market_data_ready=False,
        store_path=tmp_path / "store.json",
    )

    assert session.session_ready is False
    assert "market_data" in session.preflight[
        "failed_checks"
    ]


def test_live_runtime_session_blocks_bad_quote(tmp_path):
    session = build_live_runtime_session(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        quote_ready=False,
        store_path=tmp_path / "store.json",
    )

    assert session.session_ready is False
    assert "quote" in session.preflight[
        "failed_checks"
    ]


def test_live_runtime_session_blocks_bad_timestamp(tmp_path):
    session = build_live_runtime_session(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        timestamp_ready=False,
        store_path=tmp_path / "store.json",
    )

    assert session.session_ready is False
    assert "timestamp" in session.preflight[
        "failed_checks"
    ]


def test_session_ready_returns_boolean(tmp_path):
    result = is_live_runtime_session_ready(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        store_path=tmp_path / "store.json",
    )

    assert isinstance(result, bool)


def test_session_message_contains_strategy(tmp_path):
    message = live_runtime_session_message(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        store_path=tmp_path / "store.json",
    )

    assert "momentum" in message
    assert message.startswith(
        "LIVE RUNTIME SESSION "
    )


def test_session_message_reports_failed_preflight(tmp_path):
    message = live_runtime_session_message(
        _market_data(),
        stable_strategy="momentum",
        stability_score=None,
        quote_ready=False,
        store_path=tmp_path / "store.json",
    )

    assert message.startswith(
        "LIVE RUNTIME SESSION BLOCKED:"
    )
    assert "quote" in message


def test_session_rejects_non_dataframe():
    with pytest.raises(TypeError):
        build_live_runtime_session(
            [],
            stable_strategy="momentum",
            stability_score=None,
        )


def test_session_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        build_live_runtime_session(
            pd.DataFrame(),
            stable_strategy="momentum",
            stability_score=None,
        )
