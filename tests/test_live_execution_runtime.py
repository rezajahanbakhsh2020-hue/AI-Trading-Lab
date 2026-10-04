"""Tests for Headless Live Execution Runtime."""
import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.data.provider import (
    BiQuoteProvider,
    FunctionMarketDataProvider,
    UnsupportedInstrumentError,
    resolve_provider_for_symbol,
    resolve_requested_symbol,
)
import src.evaluation.live_execution_runtime as runtime_module
from src.evaluation.live_execution_runtime import (
    LIVE_DATA_PROVIDERS,
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
    load_live_market_data,
)
from src.evaluation.live_runtime import (
    evaluate_authorized_live_runtime,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.evaluation.research_store import save_research_candidate
from src.integration.project2_publisher import Project2Publisher


def persist_momentum_candidate(
    base_dir, candidate_id: str = "cand_momentum_live", symbol: str = "XAUUSD"
):
    ds = DatasetScope(
        dataset_id=f"ds_{symbol.lower()}_5m",
        symbol=symbol,
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    spec = ResearchExperimentSpec(
        hypothesis="Persisted live-runtime momentum candidate",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001, slippage=0.001, latency_ms=10.0
        ),
        code_provenance=CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c"),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-02",
        total_return=0.20,
        max_drawdown=0.05,
        sharpe_ratio=2.0,
        observations=50,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-02T00:00:00+00:00",
    )
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-03",
        end_date="2025-01-05",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=30,
        start_timestamp_utc="2025-01-03T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-01-05",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict={
            "passed": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )
    save_research_candidate(candidate_id=candidate_id, evidence=evidence, base_dir=base_dir, operational_stability_score=0.85)
    return candidate_id


def production_config_for(
    tmp_path, candidate_id: str = "cand_momentum_live", symbol: str = "XAUUSD"
) -> ProductionRuntimeConfig:
    persist_momentum_candidate(tmp_path, candidate_id=candidate_id, symbol=symbol)
    return ProductionRuntimeConfig(
        symbol=symbol,
        timeframe="5m",
        candidate_id=candidate_id,
        strategy_id="momentum",
        strategy_version="1.0",
        research_dir=tmp_path,
    )


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


def test_headless_import_and_constants() -> None:
    """Verify src.evaluation.live_execution_runtime can be imported cleanly without plotly or streamlit in sys.modules."""
    import sys
    import importlib

    assert runtime_module.DEFAULT_INTERVAL == "5m"
    assert runtime_module.DEFAULT_LIMIT == 200

    assert "plotly" not in sys.modules
    assert "streamlit" not in sys.modules


def test_requested_symbol_is_not_replaced_by_xauusd() -> None:
    req = {"symbol": "eurusd"}
    resolved = resolve_requested_symbol(req)
    assert resolved == "EURUSD"
    assert resolved != "XAUUSD"


def test_supported_non_xauusd_symbol_reaches_provider_unchanged(monkeypatch) -> None:
    recorded_symbol = None

    class MockBTCProvider:
        def supports_symbol(self, symbol: str) -> bool:
            return symbol == "BTCUSD"

        def get_quote(self, symbol: str) -> dict:
            nonlocal recorded_symbol
            recorded_symbol = symbol
            return {"symbol": symbol, "mid": 95000.0}

        def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
            nonlocal recorded_symbol
            recorded_symbol = symbol
            return make_dummy_df()

    monkeypatch.setitem(LIVE_DATA_PROVIDERS, "BTCUSD", MockBTCProvider())

    df = load_live_market_data("BTCUSD", "5m", 100)
    assert recorded_symbol == "BTCUSD"
    assert not df.empty


def test_unsupported_symbol_fails_closed(tmp_path) -> None:
    with pytest.raises(UnsupportedInstrumentError, match="No market-data provider supports EURUSD"):
        load_live_market_data("EURUSD", "5m", 100)

    runtime = LiveExecutionRuntime(
        symbol="EURUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path, symbol="EURUSD"),
    )

    result = runtime.run_once(publish=False, persist=True)
    assert result["blocked"] is True
    assert result["reason"] == "UNSUPPORTED_INSTRUMENT"
    assert result["symbol"] == "EURUSD"
    assert result["decision"] == "NO TRADE"
    assert result["blocked_state"]["code"] == "UNSUPPORTED_INSTRUMENT"
    assert result["blocked_state"]["error"]["code"] == "UNSUPPORTED_INSTRUMENT"


def test_provider_is_selected_by_symbol_capability() -> None:
    provider1 = BiQuoteProvider(supported_symbols=("XAUUSD",))
    provider2 = BiQuoteProvider(supported_symbols=("BTCUSD",))
    providers = [provider1, provider2]

    selected = resolve_provider_for_symbol("BTCUSD", providers)
    assert selected is provider2
    assert selected.supports_symbol("BTCUSD")
    assert not selected.supports_symbol("XAUUSD")

    none_selected = resolve_provider_for_symbol("ETHUSD", providers)
    assert none_selected is None


def test_provider_cannot_silently_substitute_another_symbol() -> None:
    def bad_quote_fetcher():
        return {"symbol": "XAUUSD", "mid": 2650.0}

    provider = BiQuoteProvider(
        supported_symbols=("EURUSD",),
        quote_fetcher=bad_quote_fetcher,
    )
    with pytest.raises(UnsupportedInstrumentError, match="does not match requested symbol"):
        provider.get_quote("EURUSD")


def test_snapshot_preserves_canonical_instrument_identity(tmp_path) -> None:
    config = production_config_for(tmp_path, symbol="XAUUSD")
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )
    with patch("src.evaluation.live_execution_runtime.load_live_market_data") as mock_load:
        mock_load.return_value = make_dummy_df()
        res = runtime.run_once(publish=False, persist=True)
        assert res["symbol"] == "XAUUSD"
        snapshot_data = json.loads((tmp_path / "snap.json").read_text())
        assert snapshot_data["symbol"] == "XAUUSD"


def test_xauusd_existing_runtime_path_remains_valid(tmp_path) -> None:
    config = production_config_for(tmp_path, symbol="XAUUSD")
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )
    with patch("src.evaluation.live_execution_runtime.load_live_market_data") as mock_load:
        mock_load.return_value = make_dummy_df()
        res = runtime.run_once(publish=False, persist=False)
        assert res["blocked"] is False
        assert res["symbol"] == "XAUUSD"


@patch("app_live.fetch_xauusd_ohlc")
def test_market_data_integrity_symbol_mismatch_prevented(mock_fetch) -> None:
    """Verify market data for symbol A can never be returned while runtime claims it belongs to symbol B."""
    mock_fetch.return_value = make_dummy_df()

    result = load_live_market_data("XAUUSD", "5m", 100)
    assert not result.empty

    with pytest.raises(UnsupportedInstrumentError) as exc_info:
        load_live_market_data("EURUSD", "5m", 100)

    assert "No market-data provider supports EURUSD" in str(exc_info.value)


@patch("app_live.fetch_xauusd_ohlc")
def test_load_live_market_data(mock_fetch) -> None:
    mock_fetch.return_value = make_dummy_df()
    df = load_live_market_data("XAUUSD", "5m", 100)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "close" in df.columns


def test_load_live_market_data_invalid_symbol() -> None:
    with pytest.raises(UnsupportedInstrumentError, match="No market-data provider supports EURUSD"):
        load_live_market_data("EURUSD", "5m", 100)


def test_provider_capability_registry_custom_adapter(monkeypatch) -> None:
    """Verify that registering a valid adapter for another symbol (e.g. BTCUSD) flows through cleanly without hardcoded restrictions."""
    def mock_btc_adapter(interval="5m", limit=100):
        df = make_dummy_df()
        return df

    monkeypatch.setitem(LIVE_DATA_PROVIDERS, "BTCUSD", mock_btc_adapter)

    df = load_live_market_data("BTCUSD", "5m", 100)
    assert not df.empty
    assert "timestamp" in df.columns


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_run_once(mock_load_data, tmp_path) -> None:
    mock_load_data.return_value = make_dummy_df()

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
    }

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    result = runtime.run_once(publish=True)

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["strategy"] == "momentum"
    assert result["stability_score"] == 0.85
    assert "contract_payload" in result
    assert result["contract_payload"]["contract_version"] == "1.0"
    assert mock_publisher.publish.called


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_idempotency_key(mock_load_data, tmp_path) -> None:
    mock_load_data.return_value = make_dummy_df()
    ref_now = pd.to_datetime(mock_load_data.return_value["openTime"], utc=True).iloc[-1].to_pydatetime()

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    res1 = runtime.run_once(publish=False, persist=False, reference_now=ref_now)
    res2 = runtime.run_once(publish=False, persist=False, reference_now=ref_now)

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


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_buy_signal_field_propagation(
    mock_load_data,
    tmp_path,
) -> None:
    """Verify end-to-end propagation of BUY trade levels (Entry, SL, TP1-TP3) from decision engine to Contract v1 payload."""
    mock_load_data.return_value = make_buy_market_data()

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
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
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


@patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime")
@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_stale_data_blocked(
    mock_load_data,
    mock_evaluate_runtime,
    tmp_path,
) -> None:
    """Verify that stale market data delegates to evaluate_authorized_live_runtime with a stale LiveMarketEvaluation."""
    df = make_buy_market_data()
    mock_load_data.return_value = df

    mock_publisher = MagicMock()
    store_path = tmp_path / "decision_history.json"
    snapshot_path = tmp_path / "latest_execution.json"

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=store_path,
        snapshot_path=snapshot_path,
        max_age_seconds=300.0,
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    df_ts = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    stale_ref_now = df_ts + pd.Timedelta(seconds=1000)

    # Let evaluate_authorized_live_runtime run for real or mock it
    mock_evaluate_runtime.side_effect = evaluate_authorized_live_runtime

    result = runtime.run_once(publish=True, skip_if_no_trade=True, persist=True, reference_now=stale_ref_now)

    assert mock_evaluate_runtime.called
    eval_arg = mock_evaluate_runtime.call_args.kwargs["evaluation"]
    assert eval_arg.fresh is False
    assert eval_arg.freshness_reason == "stale_market_data"

    # Signal must fail closed to NO TRADE with rejection reason
    assert result["decision"] == "NO TRADE"
    assert result["record"]["signal_label"] == "NO TRADE"
    assert result["record"]["quote_stale"] is True
    assert result["record"]["quote_age_seconds"] == 1000.0

    # Contract payload must reflect NO TRADE
    assert result["contract_payload"]["signal"]["decision"] == "NO TRADE"
    assert result["contract_payload"]["signal"]["signal_label"] == "NO TRADE"


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_missing_invalid_timestamp_blocked(
    mock_load_data,
    tmp_path,
) -> None:
    """Verify missing/invalid candle timestamps fail closed cleanly in market data validation stage."""
    mock_load_data.side_effect = ValueError("No valid live market data available for XAUUSD.")

    mock_publisher = MagicMock()
    store_file = tmp_path / "store.json"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    res = runtime.run_once(publish=False, persist=True)
    assert res["blocked"] is True
    assert res["reason"] == "MARKET_DATA_VALIDATION_FAILED"
    assert res["detail"] == "No valid live market data available for XAUUSD."
    assert res["decision"] == "NO TRADE"
    assert res["market_data"] is None
    assert not store_file.exists()
    assert mock_publisher.publish.call_count == 0


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_future_timestamp_blocked(
    mock_load_data,
    tmp_path,
) -> None:
    """Verify future candle timestamps fail closed with reason future_candle_timestamp."""
    df = make_buy_market_data()
    mock_load_data.return_value = df

    df_ts = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    # reference_now is BEFORE candle timestamp (candle in future)
    past_ref_now = df_ts - pd.Timedelta(seconds=100)

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=MagicMock(),
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    result = runtime.run_once(publish=False, persist=True, reference_now=past_ref_now)

    assert result["decision"] == "NO TRADE"
    assert result["record"]["quote_stale"] is True


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
def test_live_execution_runtime_freshness_boundary_conditions(
    mock_load_data,
    tmp_path,
) -> None:
    """Verify exact boundary conditions around max_age_seconds (max_age-1 is fresh, max_age+1 is stale)."""
    df = make_buy_market_data()
    mock_load_data.return_value = df

    df_ts = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()
    max_age = 300.0

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=MagicMock(),
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        max_age_seconds=max_age,
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    # 1. age = 299s <= 300s -> FRESH -> BUY
    res_fresh = runtime.run_once(publish=False, persist=False, reference_now=df_ts + pd.Timedelta(seconds=299))
    assert res_fresh["decision"] == "BUY"
    assert res_fresh["record"]["quote_stale"] is False

    # 2. age = 301s > 300s -> STALE -> NO TRADE
    res_stale = runtime.run_once(publish=False, persist=False, reference_now=df_ts + pd.Timedelta(seconds=301))
    assert res_stale["decision"] == "NO TRADE"
    assert res_stale["record"]["quote_stale"] is True


@patch("src.evaluation.live_execution_runtime.load_live_market_data")
@patch("urllib.request.urlopen")
def test_live_execution_runtime_persistence_freshness_isolation(
    mock_urlopen,
    mock_load_data,
    tmp_path,
) -> None:
    """Verify that latest_execution.json snapshot is cleanly overwritten on every execution cycle with current publish results."""
    df = make_dummy_df()
    mock_load_data.return_value = df
    ref_now = pd.to_datetime(df["openTime"], utc=True, errors="coerce").iloc[-1].to_pydatetime()

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
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
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
        research_dir=tmp_path,
        production_config=production_config_for(tmp_path),
    )

    def mock_urlopen_impl(req, *args, **kwargs):
        body = json.loads(req.data.decode("utf-8")) if hasattr(req, "data") and req.data else {}
        evt_id = body.get("event_id") or "test_event_id"
        m_resp = MagicMock()
        m_resp.getcode.return_value = 200
        m_resp.read.return_value = json.dumps({"status": "INGESTED", "event_id": evt_id, "publication_id": evt_id}).encode("utf-8")
        m_resp.__enter__.return_value = m_resp
        return m_resp

    mock_urlopen.side_effect = mock_urlopen_impl

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


def test_runtime_injected_snapshot_without_market_provider(tmp_path) -> None:
    """Requirement 3: run_once(market_data=...) evaluates injected snapshot without calling market-data provider."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    injected_df = make_buy_market_data()
    ref_now = pd.to_datetime(injected_df["openTime"], utc=True).iloc[-1].to_pydatetime()

    with patch("src.evaluation.live_execution_runtime.load_live_market_data") as mock_load:
        mock_load.side_effect = AssertionError("load_live_market_data must NOT be called when market_data is supplied")

        res = runtime.run_once(
            publish=False,
            persist=True,
            reference_now=ref_now,
            market_data=injected_df,
        )

        assert res["blocked"] is False
        assert res["decision"] == "BUY"
        assert res["strategy"] == "momentum"
        assert res["candidate_id"] == "cand_momentum_live"
        assert res["market_data"] is not None
        assert mock_load.call_count == 0


def test_normal_runtime_path_fetches_once(tmp_path) -> None:
    """Requirement 4: When market_data=None and market_data_loader=None, runtime calls load_live_market_data() exactly once."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df) as mock_load:
        res = runtime.run_once(
            publish=False,
            persist=True,
            reference_now=ref_now,
            market_data=None,
        )

        assert res["blocked"] is False
        assert res["market_data"] is not None
        assert mock_load.call_count == 1


def test_blocked_before_fetch(tmp_path) -> None:
    """Requirement 5 & C: When promoted candidate is missing/corrupt, market data loader is not called."""
    config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="cand_nonexistent_999",
        strategy_id="momentum",
        research_dir=tmp_path,
    )

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    mock_loader = MagicMock()
    with patch("src.evaluation.live_execution_runtime.load_live_market_data") as mock_load:
        res = runtime.run_once(publish=False, persist=True, market_data_loader=mock_loader)

        assert res["blocked"] is True
        assert res["reason"] == "PromotionUnavailable"
        assert res["market_data"] is None
        assert mock_load.call_count == 0
        assert mock_loader.call_count == 0


def test_market_data_and_loader_mutually_exclusive(tmp_path) -> None:
    """Requirement 1: Passing both market_data and market_data_loader fails closed with ValueError."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    with pytest.raises(ValueError, match="Cannot supply both market_data and market_data_loader"):
        runtime.run_once(
            publish=False,
            market_data=df,
            market_data_loader=lambda: df,
        )


def test_mutation_isolation_on_injected_snapshot(tmp_path) -> None:
    """Requirement G: Mutation on injected DataFrame after invocation does not affect canonical result snapshot."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()
    original_close = df["close"].iloc[-1]

    res = runtime.run_once(publish=False, persist=True, reference_now=ref_now, market_data=df)

    # Mutate original DataFrame on caller side
    df["close"] = 9999.0

    assert res["market_data"]["close"].iloc[-1] == original_close
    assert res["market_data"]["close"].iloc[-1] != 9999.0


def test_invalid_injected_snapshot_fails_closed(tmp_path) -> None:
    """Requirement H: Invalid/empty injected snapshot fails closed without executing evaluation."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    # Empty DataFrame
    empty_df = pd.DataFrame()
    res_empty = runtime.run_once(publish=False, market_data=empty_df)
    assert res_empty["blocked"] is True
    assert res_empty["reason"] == "MARKET_DATA_VALIDATION_FAILED"
    assert res_empty["market_data"] is None

    # Missing OHLC columns
    invalid_cols_df = pd.DataFrame({"openTime": ["2025-01-01T10:00:00+00:00"]})
    res_cols = runtime.run_once(publish=False, market_data=invalid_cols_df)
    assert res_cols["blocked"] is True
    assert res_cols["reason"] == "MARKET_DATA_VALIDATION_FAILED"
    assert res_cols["market_data"] is None


def test_acquisition_boundary_loader_success_and_ordering(tmp_path) -> None:
    """Scenario 1 & Ordering Proof: market_data_loader success follows exact required sequence."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    event_log: list[str] = []

    def loader_with_event():
        event_log.append("3. acquire_market_snapshot")
        return df

    orig_resolve = runtime_module.resolve_authoritative_promoted_candidate
    orig_auth = runtime_module.authorize_production_runtime
    orig_eval = runtime_module.create_live_market_evaluation
    orig_eval_rt = runtime_module.evaluate_authorized_live_runtime

    def side_resolve(*args, **kwargs):
        event_log.append("1. resolve_candidate")
        return orig_resolve(*args, **kwargs)

    def side_auth(*args, **kwargs):
        event_log.append("2. authorize_runtime")
        return orig_auth(*args, **kwargs)

    def side_eval(*args, **kwargs):
        event_log.append("4. create_live_market_evaluation")
        return orig_eval(*args, **kwargs)

    def side_eval_rt(*args, **kwargs):
        event_log.append("5. evaluate_authorized_live_runtime")
        return orig_eval_rt(*args, **kwargs)

    with patch("src.evaluation.live_execution_runtime.resolve_authoritative_promoted_candidate", side_effect=side_resolve), \
         patch("src.evaluation.live_execution_runtime.authorize_production_runtime", side_effect=side_auth), \
         patch("src.evaluation.live_execution_runtime.create_live_market_evaluation", side_effect=side_eval), \
         patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime", side_effect=side_eval_rt):

        res = runtime.run_once(
            publish=False,
            persist=True,
            reference_now=ref_now,
            market_data_loader=loader_with_event,
        )

        assert res["blocked"] is False
        assert res["decision"] == "BUY"
        assert res["market_data"] is not None

        expected_sequence = [
            "1. resolve_candidate",
            "2. authorize_runtime",
            "3. acquire_market_snapshot",
            "4. create_live_market_evaluation",
            "5. evaluate_authorized_live_runtime",
        ]
        assert event_log == expected_sequence


def test_acquisition_boundary_loader_exception_blocked(tmp_path) -> None:
    """Scenario 2 & 7 & 8: market_data_loader exception converts to blocked=True, no decision, no persistence."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    event_log: list[str] = []

    def failing_loader():
        event_log.append("acquire_market_snapshot_fail")
        raise RuntimeError("Provider connection failed")

    mock_publisher = MagicMock()

    runtime.publisher = mock_publisher
    with patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime") as mock_eval_rt:
        res = runtime.run_once(
            publish=True,
            persist=True,
            market_data_loader=failing_loader,
        )

        assert res["blocked"] is True
        assert res["reason"] == "MARKET_DATA_ACQUISITION_FAILED"
        assert res["detail"] == "Provider connection failed"
        assert res["decision"] == "NO TRADE"
        assert res["market_data"] is None
        assert mock_eval_rt.call_count == 0
        assert mock_publisher.publish.call_count == 0

        # Verify no canonical decision record was appended to decision history store
        assert not store_file.exists()


def test_acquisition_boundary_loader_invalid_dataframe_blocked(tmp_path) -> None:
    """Scenario 3: market_data_loader returning invalid DataFrame converts to blocked=True."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    # Loader returns empty DataFrame
    res = runtime.run_once(publish=False, market_data_loader=lambda: pd.DataFrame())
    assert res["blocked"] is True
    assert res["reason"] == "MARKET_DATA_VALIDATION_FAILED"
    assert res["decision"] == "NO TRADE"
    assert res["market_data"] is None

    # Loader returns DataFrame missing OHLC columns
    res_bad_cols = runtime.run_once(publish=False, market_data_loader=lambda: pd.DataFrame({"foo": [1]}))
    assert res_bad_cols["blocked"] is True
    assert res_bad_cols["reason"] == "MARKET_DATA_VALIDATION_FAILED"
    assert res_bad_cols["market_data"] is None


def test_candidate_resolution_failure_prevents_loader_call(tmp_path) -> None:
    """Scenario 5: Candidate resolution failure prevents loader execution."""
    config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id="nonexistent_candidate",
        strategy_id="momentum",
        research_dir=tmp_path,
    )

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    loader_called = False

    def spy_loader():
        nonlocal loader_called
        loader_called = True
        return make_buy_market_data()

    res = runtime.run_once(publish=False, market_data_loader=spy_loader)

    assert res["blocked"] is True
    assert res["reason"] == "PromotionUnavailable"
    assert loader_called is False
    assert res["market_data"] is None


def test_production_authorization_failure_prevents_loader_call(tmp_path) -> None:
    """Scenario 6: Production authorization failure prevents loader execution."""
    config = production_config_for(tmp_path)
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=tmp_path / "store.json",
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    loader_called = False

    def spy_loader():
        nonlocal loader_called
        loader_called = True
        return make_buy_market_data()

    from src.evaluation.live_production_decision import ProductionRuntimeAuthorizationError

    with patch("src.evaluation.live_execution_runtime.authorize_production_runtime") as mock_auth:
        mock_auth.side_effect = ProductionRuntimeAuthorizationError("Authorization rejected")

        res = runtime.run_once(publish=False, market_data_loader=spy_loader)

        assert res["blocked"] is True
        assert res["reason"] == "ProductionRuntimeAuthorizationError"
        assert res["detail"] == "Authorization rejected"
        assert loader_called is False
        assert res["market_data"] is None


def test_market_evaluation_exception_blocked(tmp_path) -> None:
    """Requirement 1 & 2 & 7: Exception in create_live_market_evaluation yields blocked=True, decision='NO TRADE', market_data=None, and evaluate_authorized_live_runtime is not reached."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    mock_publisher = MagicMock()
    runtime.publisher = mock_publisher

    with patch("src.evaluation.live_execution_runtime.create_live_market_evaluation") as mock_create_eval, \
         patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime") as mock_eval_rt:
        mock_create_eval.side_effect = RuntimeError("Market evaluation error")

        res = runtime.run_once(publish=True, persist=True, market_data=df)

        assert res["blocked"] is True
        assert res["reason"] == "LIVE_MARKET_EVALUATION_FAILED"
        assert res["detail"] == "Market evaluation error"
        assert res["decision"] == "NO TRADE"
        assert res["market_data"] is None
        assert mock_eval_rt.call_count == 0
        assert mock_publisher.publish.call_count == 0
        assert not store_file.exists()


def test_runtime_evaluation_exception_blocked(tmp_path) -> None:
    """Requirement 3 & 4 & 5 & 6 & 7: Exception in evaluate_authorized_live_runtime yields blocked=True, decision='NO TRADE', market_data=None, and does not construct publication, publish, or persist decision history."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    mock_publisher = MagicMock()
    runtime.publisher = mock_publisher

    with patch("src.evaluation.live_execution_runtime.evaluate_authorized_live_runtime") as mock_eval_rt, \
         patch("src.evaluation.live_production_decision.ProductionIntelligencePublication.from_artifacts") as mock_from_artifacts:
        mock_eval_rt.side_effect = RuntimeError("Runtime evaluation error")

        res = runtime.run_once(publish=True, persist=True, market_data=df)

        assert res["blocked"] is True
        assert res["reason"] == "LIVE_RUNTIME_EVALUATION_FAILED"
        assert res["detail"] == "Runtime evaluation error"
        assert res["decision"] == "NO TRADE"
        assert res["market_data"] is None
        assert mock_from_artifacts.call_count == 0
        assert mock_publisher.publish.call_count == 0
        assert not store_file.exists()


def test_publication_construction_exception_preserves_persisted_decision(tmp_path) -> None:
    """Requirement: Exception in from_artifacts after successful persistence preserves blocked=False and persisted decision."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    with patch("src.evaluation.live_production_decision.ProductionIntelligencePublication.from_artifacts") as mock_from_artifacts:
        mock_from_artifacts.side_effect = ValueError("Publication construction error")

        res = runtime.run_once(publish=False, persist=True, market_data=df, reference_now=ref_now)

        assert res["blocked"] is False
        assert res["decision"] == "BUY"
        assert res["current_lifecycle_state"] == "PERSISTED"
        assert res["publish_result"]["status"] == "FAILED"
        assert "Publication construction error" in res["publish_result"]["error"]
        assert store_file.exists()
        history = json.loads(store_file.read_text())
        assert len(history) == 1
        assert history[0]["signal_label"] == "BUY"


def test_failure_before_persistence_yields_blocked_no_trade_no_history(tmp_path) -> None:
    """Mandatory Test Scenario 1: Failure before persistence yields blocked=True, NO TRADE, no decision history."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    # Trigger exception in strategy evaluation BEFORE persistence
    with patch("src.evaluation.live_runtime.evaluate_production_decision") as mock_eval_dec:
        mock_eval_dec.side_effect = ValueError("Strategy evaluation failure before persistence")

        res = runtime.run_once(publish=True, persist=True, market_data=df, reference_now=ref_now)

        assert res["blocked"] is True
        assert res["decision"] == "NO TRADE"
        assert not store_file.exists()


def test_failure_during_publication_after_persistence_preserves_persisted_decision(tmp_path) -> None:
    """Mandatory Test Scenario 2: Failure during publication after successful persistence preserves decision and blocked=False."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    mock_publisher = MagicMock()
    mock_publisher.publish.side_effect = RuntimeError("Publisher network connection error")

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    res = runtime.run_once(publish=True, persist=True, market_data=df, reference_now=ref_now)

    # 1. Canonical decision really persisted in decision_history.json
    assert store_file.exists()
    history = json.loads(store_file.read_text())
    assert len(history) == 1
    assert history[0]["signal_label"] == "BUY"

    # 2. Result is NOT blocked=True / NO TRADE
    assert res["blocked"] is False
    assert res["decision"] == "BUY"
    assert res["current_lifecycle_state"] == "PERSISTED"
    assert res["publish_result"]["status"] == "FAILED"
    assert "Publisher network connection error" in str(res["publish_result"]["error"])


def test_no_blocked_true_with_persisted_decision_invariant(tmp_path) -> None:
    """Mandatory Test Scenario 5: Explicit proof that no state of blocked=True + persisted normal canonical decision ever occurs."""
    config = production_config_for(tmp_path)
    store_file = tmp_path / "store.json"
    mock_publisher = MagicMock()
    mock_publisher.publish.side_effect = RuntimeError("Outbound transport failure")

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=store_file,
        snapshot_path=tmp_path / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    df = make_buy_market_data()
    ref_now = pd.to_datetime(df["openTime"], utc=True).iloc[-1].to_pydatetime()

    res = runtime.run_once(publish=True, persist=True, market_data=df, reference_now=ref_now)

    if store_file.exists() and len(json.loads(store_file.read_text())) > 0:
        assert res["blocked"] is False, "Forbidden state: decision is persisted in store but returned result claims blocked=True!"
