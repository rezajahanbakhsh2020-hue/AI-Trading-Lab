"""Tests for Headless Live Execution Runtime."""
import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.evaluation.live_execution_runtime import LiveExecutionRuntime, load_live_market_data
from src.integration.project2_publisher import Project2Publisher


def make_dummy_df() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 10:00", periods=100, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "openTime": timestamps,
        "open": [2000.0 + i for i in range(100)],
        "high": [2005.0 + i for i in range(100)],
        "low": [1995.0 + i for i in range(100)],
        "close": [2002.0 + i for i in range(100)],
        "timestamp": timestamps,
    })
    return df


@patch("src.evaluation.live_execution_runtime.fetch_xauusd_ohlc")
def test_load_live_market_data(mock_fetch) -> None:
    mock_fetch.return_value = make_dummy_df()
    df = load_live_market_data("XAUUSD", "5m", 100)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "close" in df.columns


def test_load_live_market_data_invalid_symbol() -> None:
    with pytest.raises(ValueError, match="Unsupported instrument symbol"):
        load_live_market_data("EURUSD", "5m", 100)


def test_provider_capability_registry_custom_adapter(monkeypatch) -> None:
    """Verify that registering a valid adapter for another symbol (e.g. BTCUSD) flows through cleanly without hardcoded restrictions."""
    from src.evaluation.live_execution_runtime import LIVE_DATA_PROVIDERS

    def mock_btc_adapter(interval="5m", limit=100):
        df = make_dummy_df()
        return df

    monkeypatch.setitem(LIVE_DATA_PROVIDERS, "BTCUSD", mock_btc_adapter)

    df = load_live_market_data("BTCUSD", "5m", 100)
    assert not df.empty
    assert "timestamp" in df.columns


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_run_once(mock_load_data, mock_load_selection) -> None:
    mock_load_data.return_value = make_dummy_df()
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
    }

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
    )

    result = runtime.run_once(publish=True)

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["strategy"] == "momentum"
    assert result["stability_score"] == 0.85
    assert "contract_payload" in result
    assert result["contract_payload"]["contract_version"] == "1.0"
    assert mock_publisher.publish.called


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_idempotency_key(mock_load_data, mock_load_selection) -> None:
    mock_load_data.return_value = make_dummy_df()
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    runtime = LiveExecutionRuntime(symbol="XAUUSD", interval="5m")

    res1 = runtime.run_once(publish=False, persist=False)
    res2 = runtime.run_once(publish=False, persist=False)

    event_id1 = res1["contract_payload"]["event_id"]
    event_id2 = res2["contract_payload"]["event_id"]

    assert event_id1 == event_id2
    assert len(event_id1) == 32


def make_buy_market_data(start_time="2025-01-01 10:00") -> pd.DataFrame:
    """Create market data that triggers a momentum BUY signal (upward trend and positive momentum)."""
    timestamps = pd.date_range(start_time, periods=100, freq="5min", tz="UTC")
    # Base prices increasing strongly to ensure positive momentum and UP trend
    prices = [2000.0 + (i * 2.0) for i in range(100)]
    df = pd.DataFrame({
        "openTime": timestamps,
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "timestamp": timestamps,
    })
    return df


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_buy_signal_field_propagation(
    mock_load_data,
    mock_load_selection,
    tmp_path,
) -> None:
    """Verify end-to-end propagation of BUY trade levels (Entry, SL, TP1-TP3) from decision engine to Contract v1 payload."""
    mock_load_data.return_value = make_buy_market_data()
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
    }

    store_path = tmp_path / "decision_history.json"
    snapshot_path = tmp_path / "latest_execution.json"

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=store_path,
        snapshot_path=snapshot_path,
    )

    df = mock_load_data.return_value
    ref_now = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()

    result = runtime.run_once(publish=True, persist=True, reference_now=ref_now)

    # 1. Decision & Signal
    assert result["decision"] == "BUY"
    assert result["strategy"] == "momentum"
    assert result["stability_score"] == 0.85

    # 2. Record fields
    rec = result["record"]
    assert rec["symbol"] == "XAUUSD"
    assert rec["interval"] == "5m"
    assert rec["signal"] == 1
    assert rec["signal_label"] == "BUY"
    assert rec["trend"] == "UP"
    assert rec["entry_price"] > 0
    assert rec["stop_loss"] < rec["entry_price"]
    assert rec["take_profit"] > rec["entry_price"]
    assert rec["risk_reward_ratio"] > 0

    # 3. Contract v1.0 payload trade setup
    ts = result["contract_payload"]["trade_setup"]
    assert ts["entry_price"] == rec["entry_price"]
    assert ts["stop_loss"] == rec["stop_loss"]
    assert ts["tp1"] > ts["entry_price"]
    assert ts["tp2"] > ts["tp1"]
    assert ts["tp3"] > ts["tp2"]
    assert ts["take_profit"] == rec["take_profit"]
    assert ts["risk_reward_ratio"] == rec["risk_reward_ratio"]

    # 4. Provenance
    prov = result["contract_payload"]["provenance"]
    assert prov["source"] == "AI-Trading-Lab"
    assert "produced_at" in prov


@patch("src.evaluation.live_execution_runtime.build_live_runtime")
@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_stale_data_blocked(
    mock_load_data,
    mock_load_selection,
    mock_build_runtime,
    tmp_path,
) -> None:
    """Verify that stale market data fails closed: build_live_runtime is NOT called, decision is set to NO TRADE with reason stale_market_data and quote_stale=True."""
    df = make_buy_market_data()
    mock_load_data.return_value = df
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "SKIPPED_NO_TRADE",
        "published": False,
        "reason": "Decision is NO TRADE and skip_if_no_trade=True",
    }

    store_path = tmp_path / "decision_history.json"
    snapshot_path = tmp_path / "latest_execution.json"

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=store_path,
        snapshot_path=snapshot_path,
        max_age_seconds=300.0,
    )

    df_ts = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    stale_ref_now = df_ts + pd.Timedelta(seconds=1000)

    result = runtime.run_once(publish=True, skip_if_no_trade=True, persist=True, reference_now=stale_ref_now)

    # CRITICAL INVARIANT: build_live_runtime MUST NOT BE CALLED FOR STALE DATA
    assert not mock_build_runtime.called, "build_live_runtime was called for stale market data!"

    # Signal must fail closed to NO TRADE with rejection reason
    assert result["decision"] == "NO TRADE"
    assert result["record"]["signal_label"] == "NO TRADE"
    assert result["record"]["quote_stale"] is True
    assert result["record"]["quote_age_seconds"] == 1000.0

    # Contract payload must reflect NO TRADE
    assert result["contract_payload"]["signal"]["decision"] == "NO TRADE"
    assert result["contract_payload"]["signal"]["signal_label"] == "NO TRADE"

    # Publisher was called with NO TRADE payload, skipping publication
    assert mock_publisher.publish.called


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_missing_invalid_timestamp_blocked(
    mock_load_data,
    mock_load_selection,
    tmp_path,
) -> None:
    """Verify missing/invalid candle timestamps fail closed cleanly."""
    df = make_buy_market_data()
    # corrupt latest timestamp
    df["timestamp"] = pd.NaT
    mock_load_data.return_value = df
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    mock_publisher = MagicMock()
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
    )

    result = runtime.run_once(publish=False, persist=True)

    assert result["decision"] == "NO TRADE"
    assert result["record"]["quote_stale"] is True


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_future_timestamp_blocked(
    mock_load_data,
    mock_load_selection,
    tmp_path,
) -> None:
    """Verify future candle timestamps fail closed with reason future_candle_timestamp."""
    df = make_buy_market_data()
    mock_load_data.return_value = df
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    df_ts = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    # reference_now is BEFORE candle timestamp (candle in future)
    past_ref_now = df_ts - pd.Timedelta(seconds=100)

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=MagicMock(),
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
    )

    result = runtime.run_once(publish=False, persist=True, reference_now=past_ref_now)

    assert result["decision"] == "NO TRADE"
    assert result["record"]["quote_stale"] is True


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_freshness_boundary_conditions(
    mock_load_data,
    mock_load_selection,
    tmp_path,
) -> None:
    """Verify exact boundary conditions around max_age_seconds (max_age-1 is fresh, max_age+1 is stale)."""
    df = make_buy_market_data()
    mock_load_data.return_value = df
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    df_ts = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    max_age = 300.0

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=MagicMock(),
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        max_age_seconds=max_age,
    )

    # 1. age = 299s <= 300s -> FRESH -> BUY
    res_fresh = runtime.run_once(publish=False, persist=False, reference_now=df_ts + pd.Timedelta(seconds=299))
    assert res_fresh["decision"] == "BUY"
    assert res_fresh["record"]["quote_stale"] is False

    # 2. age = 301s > 300s -> STALE -> NO TRADE
    res_stale = runtime.run_once(publish=False, persist=False, reference_now=df_ts + pd.Timedelta(seconds=301))
    assert res_stale["decision"] == "NO TRADE"
    assert res_stale["record"]["quote_stale"] is True


@patch("src.evaluation.live_execution_runtime.load_production_selection")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
@patch("urllib.request.urlopen")
def test_live_execution_runtime_persistence_freshness_isolation(
    mock_urlopen,
    mock_load_data,
    mock_load_selection,
    tmp_path,
) -> None:
    """Verify that latest_execution.json snapshot is cleanly overwritten on every execution cycle with current publish results."""
    df = make_dummy_df()
    mock_load_data.return_value = df
    ref_now = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    mock_load_selection.return_value = {
        "stable_strategy": "momentum",
        "stability_score": 0.85,
    }

    # Setup mock HTTP response for successful publish
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b'{"status": "ACKNOWLEDGED"}'
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    snapshot_path = tmp_path / "latest_execution.json"
    store_path = tmp_path / "decision_history.json"

    # Execution 1: Publishing disabled
    publisher1 = Project2Publisher(enabled=False)
    runtime1 = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=publisher1,
        store_path=store_path,
        snapshot_path=snapshot_path,
    )
    res1 = runtime1.run_once(publish=True, persist=True, reference_now=ref_now)
    assert res1["publish_result"]["status"] == "SKIPPED_DISABLED"

    data_snap1 = json.loads(snapshot_path.read_text())
    assert data_snap1["publish_result"]["status"] == "SKIPPED_DISABLED"

    # Execution 2: Publishing enabled with mock successful response
    publisher2 = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="valid-key",
        enabled=True,
        max_age_seconds=1000000000,
    )
    runtime2 = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=publisher2,
        store_path=store_path,
        snapshot_path=snapshot_path,
    )
    res2 = runtime2.run_once(publish=True, persist=True, reference_now=ref_now)
    assert res2["publish_result"]["status"] == "PUBLISHED"

    data_snap2 = json.loads(snapshot_path.read_text())
    # Verify snapshot was overwritten with current execution result
    assert data_snap2["publish_result"]["status"] == "PUBLISHED"
    assert data_snap2["publish_result"]["published"] is True

    # Execution 3: Execution without publishing requested (publish=False)
    res3 = runtime2.run_once(publish=False, persist=True, reference_now=ref_now)
    assert res3["publish_result"] is None

    data_snap3 = json.loads(snapshot_path.read_text())
    assert data_snap3["publish_result"] is None


@patch("src.evaluation.live_execution_runtime.LiveExecutionRuntime")
def test_main_cli_misconfigured_exit_code(mock_runtime_cls) -> None:
    mock_instance = MagicMock()
    mock_runtime_cls.return_value = mock_instance
    mock_instance.run_once.return_value = {
        "symbol": "XAUUSD",
        "interval": "5m",
        "decision": "BUY",
        "strategy": "momentum",
        "stability_score": 0.85,
        "publish_result": {
            "status": "FAILED",
            "reason": "Missing PROJECT2_PUBLISH_URL configuration",
        },
    }

    with patch("sys.argv", ["run_live_execution.py", "--publish"]):
        with pytest.raises(SystemExit) as exc:
            from src.evaluation.live_execution_runtime import main
            main()
        assert exc.value.code == 2
