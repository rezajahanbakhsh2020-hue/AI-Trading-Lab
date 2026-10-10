import pandas as pd
import pytest

from src.evaluation.live_runtime import (
    build_live_runtime,
)


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=80,
                freq="5min",
            ),
            "isOpen": False,
            "open": [4400.0 + i for i in range(80)],
            "high": [4401.0 + i for i in range(80)],
            "low": [4399.0 + i for i in range(80)],
            "close": [4400.5 + i for i in range(80)],
        }
    )


def test_live_runtime_builds_decision_and_display(tmp_path):
    result = build_live_runtime(
        data=_data(),
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
    )

    assert result.decision["symbol"] == "XAUUSD"
    assert result.decision["interval"] == "5m"

    assert result.decision["decision"] in {
        "BUY",
        "NO TRADE",
    }

    assert result.display["decision"] == result.decision["decision"]


def test_live_runtime_display_contains_tp_levels(tmp_path):
    df = _data()
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime()
    result = build_live_runtime(
        data=df,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    assert "tp1" in result.display
    assert "tp2" in result.display
    assert "tp3" in result.display

    assert result.display["tp1"] is not None
    assert result.display["tp2"] is not None
    assert result.display["tp3"] is not None


def test_live_runtime_keeps_decision_and_display_consistent(tmp_path):
    result = build_live_runtime(
        data=_data(),
        stable_strategy="momentum",
        stability_score=None,
        store_path=tmp_path / "store.json",
    )

    assert (
        result.display["decision"]
        == result.decision["decision"]
    )


def test_live_runtime_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        build_live_runtime(
            data=[],
            stable_strategy="momentum",
            stability_score=None,
        )


from unittest.mock import MagicMock, patch

def test_evaluate_authorized_live_runtime_publication_exception_handled(tmp_path):
    df = _data()
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime()
    mock_publisher = MagicMock()
    mock_publisher.publish.side_effect = RuntimeError("Publisher network fault")

    result = build_live_runtime(
        data=df,
        stable_strategy="momentum",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
        publisher=mock_publisher,
        publish=True,
    )

    assert result.canonical_decision is not None
    assert result.canonical_decision.current_state.value == "PERSISTED"
    assert result.decision["decision"] == "BUY"
    assert "publish_result" in result.decision
    assert result.decision["publish_result"]["status"] == "FAILED"
    assert "Publisher network fault" in str(result.decision["publish_result"]["error"])


def test_evaluate_authorized_live_runtime_display_exception_handled(tmp_path):
    df = _data()
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime()

    with patch("src.evaluation.live_runtime.build_live_trade_display") as mock_display:
        mock_display.side_effect = RuntimeError("Display formatting error")

        result = build_live_runtime(
            data=df,
            stable_strategy="momentum",
            stability_score=None,
            symbol="XAUUSD",
            interval="5m",
            reference_now=ref_now,
            store_path=tmp_path / "store.json",
        )

        assert result.canonical_decision is not None
        assert result.canonical_decision.current_state.value == "PERSISTED"
        assert result.decision["decision"] == "BUY"
        assert result.display["decision"] == "BUY"


def test_live_runtime_rejects_empty_dataframe():
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_live_runtime(
            data=pd.DataFrame(),
            stable_strategy="momentum",
            stability_score=None,
        )
