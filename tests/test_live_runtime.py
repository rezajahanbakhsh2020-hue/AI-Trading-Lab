import pandas as pd
import pytest

from src.evaluation.live_runtime import (
    build_live_runtime,
)


def _make_synth_candidate(strategy_name="momentum", timeframe="5m", stability_score=0.85, fast_window=10, slow_window=20):
    from tests.test_production_decision_integrity import make_promoted_evidence
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from src.evaluation.live_production_decision import PromotedCandidateArtifact
    ev = make_promoted_evidence(timeframe=timeframe, strategy_name=strategy_name)
    rob = assess_research_robustness(ev)
    gov = qualify_research_evidence(ev, robustness_assessment=rob)
    params = {"fast_window": fast_window, "slow_window": slow_window, "stop_loss_pct": 0.01, "take_profit_pct": 0.02} if strategy_name in ("moving_average", "baseline") else {"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02}
    return PromotedCandidateArtifact(
        candidate_id=f"cand_{strategy_name}_{timeframe}",
        strategy_name=strategy_name,
        strategy_version="1.0",
        evidence=ev,
        symbol="XAUUSD",
        timeframe=timeframe,
        operational_stability_score=stability_score,
        parameters=params,
        governance_decision=gov,
        governance_decision_fingerprint=gov.decision_fingerprint,
    )


@pytest.fixture(autouse=True)
def mock_default_resolve_promoted_candidate(monkeypatch):
    def mock_resolve(*args, candidate_id=None, strategy_id=None, symbol=None, timeframe=None, base_dir=None, **kwargs):
        strat = strategy_id or "momentum"
        tf = timeframe or "5m"
        return _make_synth_candidate(strategy_name=strat, timeframe=tf)

    monkeypatch.setattr("src.evaluation.live_runtime.resolve_promoted_candidate", mock_resolve)
    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", mock_resolve)


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=80,
                freq="5min",
                tz="UTC",
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
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)
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

def test_custom_moving_average_windows_retained_through_runtime_and_display(tmp_path, monkeypatch):
    from src.evaluation.research_store import resolve_promoted_candidate
    from src.evaluation.live_production_decision import PromotedCandidateArtifact
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from dataclasses import replace

    ma_cand = resolve_promoted_candidate(strategy_id="moving_average", symbol="XAUUSD", timeframe="5m")
    custom_params = {"fast_window": 10, "slow_window": 30, "stop_loss_pct": 0.01, "take_profit_pct": 0.02}
    custom_spec = replace(ma_cand.evidence.spec, parameters=custom_params)
    custom_evidence = replace(ma_cand.evidence, spec=custom_spec, experiment_fingerprint=custom_spec.fingerprint)
    rob = assess_research_robustness(custom_evidence)
    gov = qualify_research_evidence(custom_evidence, robustness_assessment=rob)
    custom_cand = PromotedCandidateArtifact(
        candidate_id=ma_cand.candidate_id,
        strategy_name=ma_cand.strategy_name,
        strategy_version=ma_cand.strategy_version,
        evidence=custom_evidence,
        symbol=ma_cand.symbol,
        timeframe=ma_cand.timeframe,
        operational_stability_score=ma_cand.operational_stability_score,
        governance_decision=gov,
        governance_decision_fingerprint=gov.decision_fingerprint,
        parameters=custom_params,
    )

    monkeypatch.setattr(
        "src.evaluation.live_runtime.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )
    monkeypatch.setattr(
        "src.evaluation.research_store.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )

    df = _data()
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)
    result = build_live_runtime(
        data=df,
        stable_strategy="moving_average",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    assert result.decision["fast_window"] == 10
    assert result.decision["slow_window"] == 30
    assert result.display["fast_window"] == 10
    assert result.display["slow_window"] == 30


def test_display_trend_follows_candidate_windows_not_default_20_50(tmp_path, monkeypatch):
    from src.evaluation.research_store import resolve_promoted_candidate
    from src.evaluation.live_production_decision import PromotedCandidateArtifact
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    from dataclasses import replace
    from live_trend import build_live_trend_snapshot

    ma_cand = resolve_promoted_candidate(strategy_id="moving_average", symbol="XAUUSD", timeframe="5m")
    custom_params = {"fast_window": 10, "slow_window": 30, "stop_loss_pct": 0.01, "take_profit_pct": 0.02}
    custom_spec = replace(ma_cand.evidence.spec, parameters=custom_params)
    custom_evidence = replace(ma_cand.evidence, spec=custom_spec, experiment_fingerprint=custom_spec.fingerprint)
    rob = assess_research_robustness(custom_evidence)
    gov = qualify_research_evidence(custom_evidence, robustness_assessment=rob)
    custom_cand = PromotedCandidateArtifact(
        candidate_id=ma_cand.candidate_id,
        strategy_name=ma_cand.strategy_name,
        strategy_version=ma_cand.strategy_version,
        evidence=custom_evidence,
        symbol=ma_cand.symbol,
        timeframe=ma_cand.timeframe,
        operational_stability_score=ma_cand.operational_stability_score,
        governance_decision=gov,
        governance_decision_fingerprint=gov.decision_fingerprint,
        parameters=custom_params,
    )

    monkeypatch.setattr(
        "src.evaluation.live_runtime.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )
    monkeypatch.setattr(
        "src.evaluation.research_store.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )

    # Discriminating fixture: 80 closes
    # closes = [100.0]*60 + [10.0]*10 + [120.0]*10
    closes = [100.0] * 60 + [10.0] * 10 + [120.0] * 10
    dates = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")

    df = pd.DataFrame({
        "timestamp": dates,
        "isOpen": False,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    # Assert discriminating math independent oracle:
    oracle_10_30 = build_live_trend_snapshot(df, fast_window=10, slow_window=30)
    oracle_20_50 = build_live_trend_snapshot(df, fast_window=20, slow_window=50)
    assert oracle_10_30["trend"] == "UP"
    assert oracle_20_50["trend"] == "DOWN"

    ref_now = dates[-1].to_pydatetime() + pd.Timedelta(minutes=5)
    result = build_live_runtime(
        data=df,
        stable_strategy="moving_average",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    assert result.decision["fast_window"] == 10
    assert result.decision["slow_window"] == 30
    assert result.display["fast_window"] == 10
    assert result.display["slow_window"] == 30
    assert result.display["trend"] == "UP"


def test_live_runtime_preserves_moving_average_buy_when_momentum_inactive(tmp_path):
    rows = 80
    dates = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    closes = [100.0 + i * 0.5 for i in range(rows)]
    closes[-1] = closes[-11] - 0.1  # momentum signal is 0

    df = pd.DataFrame({
        "timestamp": dates,
        "isOpen": False,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    ref_now = dates[-1].to_pydatetime() + pd.Timedelta(minutes=5)
    result = build_live_runtime(
        data=df,
        stable_strategy="moving_average",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    assert result.decision["decision"] == "BUY"
    assert result.decision["signal"] == 1
    assert result.display["decision"] == "BUY"


def test_live_runtime_preserves_moving_average_no_trade_when_momentum_active(tmp_path):
    rows = 80
    dates = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    closes = [200.0 - i * 0.5 for i in range(rows)]
    closes[-1] = closes[-11] + 5.0  # momentum signal is 1, but fast MA < slow MA

    df = pd.DataFrame({
        "timestamp": dates,
        "isOpen": False,
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    ref_now = dates[-1].to_pydatetime() + pd.Timedelta(minutes=5)
    result = build_live_runtime(
        data=df,
        stable_strategy="moving_average",
        stability_score=None,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    assert result.decision["decision"] == "NO TRADE"
    assert result.decision["signal"] == 0
    assert result.display["decision"] == "NO TRADE"


def test_live_runtime_operational_veto_blocks_execution(tmp_path, monkeypatch):
    df = _data()
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)

    from src.evaluation.research_store import resolve_promoted_candidate
    from dataclasses import replace

    ma_cand = resolve_promoted_candidate(strategy_id="moving_average", symbol="XAUUSD", timeframe="5m")
    custom_cand = replace(ma_cand, operational_stability_score=0.40)

    monkeypatch.setattr(
        "src.evaluation.live_runtime.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )
    monkeypatch.setattr(
        "src.evaluation.research_store.resolve_promoted_candidate",
        lambda *args, **kwargs: custom_cand,
    )

    result = build_live_runtime(
        data=df,
        stable_strategy="moving_average",
        stability_score=0.40,
        min_stability_score=0.50,
        symbol="XAUUSD",
        interval="5m",
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
    )

    assert result.decision["decision"] == "NO TRADE"
    assert result.decision["reason"] == "stability_score_below_threshold"
    assert result.display["decision"] == "NO TRADE"


def test_evaluate_authorized_live_runtime_publication_exception_handled(tmp_path):
    df = _data()
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)
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
    ref_now = pd.to_datetime(df["timestamp"], utc=True).iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)

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
