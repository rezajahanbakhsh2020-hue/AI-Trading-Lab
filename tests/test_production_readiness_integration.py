"""Integration tests against the real committed research store and market data.

Unlike tests/test_production_end_to_end.py, these tests do NOT mock candidate
resolution: they exercise the canonical promotion lineage path end-to-end
(resolve_promoted_candidate against DEFAULT_RESEARCH_DIR) with the committed
XAUUSD daily dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.production_end_to_end import (
    DEFAULT_DATA_PATH,
    load_production_market_data,
    run_production_end_to_end,
)
from src.evaluation.final_release import validate_final_release
from src.evaluation.research_store import (
    DEFAULT_RESEARCH_DIR,
    resolve_promoted_candidate,
)


COMMITTED_CANDIDATE_ID = "cand_moving_average_1d_verified_20261020"


def _committed_data() -> pd.DataFrame:
    return load_production_market_data(DEFAULT_DATA_PATH)


def test_load_production_market_data_returns_tz_aware_utc():
    data = _committed_data()

    assert not data.empty
    assert isinstance(data["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(data["timestamp"].dt.tz) == "UTC"


def test_committed_candidate_resolves_strategy_identity_consistently():
    """Resolution is consistent with the persisted strategy, no hardcoded identity."""
    artifact = resolve_promoted_candidate(
        candidate_id=COMMITTED_CANDIDATE_ID,
        symbol="XAUUSD",
        timeframe="1d",
        base_dir=DEFAULT_RESEARCH_DIR,
    )

    assert artifact is not None
    assert artifact.candidate_id == COMMITTED_CANDIDATE_ID
    assert artifact.strategy_name == "moving_average"
    assert artifact.timeframe == "1D"
    # Stability values must be finite, non-boolean evidence, not defaults.
    assert isinstance(artifact.operational_stability_score, float)
    assert artifact.operational_stability_score != 1.0
    assert artifact.operational_stability_score == pytest.approx(
        0.5418264967501166
    )


def test_hardcoded_strategy_identity_triggers_integrity_error():
    """A wrong strategy_id against an explicit candidate fails closed."""
    from src.evaluation.research_store import PromotionIntegrityError

    with pytest.raises(PromotionIntegrityError, match="momentum"):
        resolve_promoted_candidate(
            candidate_id=COMMITTED_CANDIDATE_ID,
            strategy_id="momentum",
            symbol="XAUUSD",
            timeframe="1d",
            base_dir=DEFAULT_RESEARCH_DIR,
        )


def test_run_production_end_to_end_stale_data_fails_closed(tmp_path):
    """The committed dataset predates now: freshness must fail closed and the
    runtime must refuse to trade historical data as live.  A NO TRADE produced
    by stale inputs must NOT be releasable (owner contract: NO TRADE must not
    bypass stale/invalid market data)."""
    data = _committed_data()

    result = run_production_end_to_end(
        data,
        candidate_id=COMMITTED_CANDIDATE_ID,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        store_path=tmp_path / "integration_store.json",
    )

    selection = result["production_selection"]
    assert selection["candidate_id"] == COMMITTED_CANDIDATE_ID
    assert selection["stable_strategy"] == "moving_average"
    assert selection["source"] == "promoted_candidate_binding"
    # Real binding stability score, finite and non-boolean.
    assert isinstance(selection["stability_score"], float)
    assert selection["stability_score"] == pytest.approx(0.5418264967501166)

    decision = result["decision"]
    assert decision["decision"] == "NO TRADE"
    assert decision["quote_stale"] is True
    assert decision["reason"] == "stale_market_data"
    # Stale data must not fabricate trade levels.
    assert result["display"]["entry_price"] is None
    assert result["display"]["stop_loss"] is None
    assert result["display"]["tp1"] is None
    assert result["display"]["tp2"] is None
    assert result["display"]["tp3"] is None

    # Fail-closed: stale-data NO TRADE is not releasable.
    assert result["release_gate"]["release_ready"] is False
    assert result["release_gate"]["checks"]["market_data_fresh"] is False
    assert result["end_to_end_ready"] is False

    final_release = validate_final_release(result)
    assert final_release["final_release_ready"] is False
    assert final_release["checks"]["market_data_fresh"] is False
    assert final_release["status"] == "RELEASE BLOCKED"


def _fresh_reference_now(data: pd.DataFrame):
    """Reference clock slightly after the CLOSE of the final 1D candle (opens at 00:00 UTC): deterministic freshness."""
    last_ts = pd.to_datetime(data["timestamp"], utc=True).iloc[-1]
    # Provider timestamps mark candle opens; a 1D candle closes one day later.
    return last_ts.to_pydatetime() + pd.Timedelta(days=1) + pd.Timedelta(seconds=60)


def test_run_production_end_to_end_fresh_data_releasable(tmp_path):
    """With a deterministic fresh reference clock aligned to the committed
    dataset, all mandatory gates must pass and the release must be READY for
    whichever canonical decision the strategy legitimately produces —
    BUY/SELL with ordered levels, or a legitimate NO TRADE with no levels."""
    data = _committed_data()

    result = run_production_end_to_end(
        data,
        candidate_id=COMMITTED_CANDIDATE_ID,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        reference_now=_fresh_reference_now(data),
        store_path=tmp_path / "integration_store_fresh.json",
    )

    decision = result["decision"]
    assert decision["quote_stale"] is False
    assert result["release_gate"]["release_ready"] is True
    assert result["release_gate"]["checks"]["market_data_fresh"] is True
    assert result["end_to_end_ready"] is True

    release_decision = decision["decision"]
    levels = result["overlay"]["levels"]

    if release_decision == "NO TRADE":
        assert levels["entry"] is None
        assert levels["stop_loss"] is None
        assert levels["tp1"] is None
        assert levels["tp2"] is None
        assert levels["tp3"] is None
    else:
        assert levels["entry"] is not None
        assert all(
            levels[k] is not None
            for k in ("stop_loss", "tp1", "tp2", "tp3")
        )
        if release_decision == "BUY":
            assert (
                levels["stop_loss"]
                < levels["entry"]
                < levels["tp1"]
                < levels["tp2"]
                < levels["tp3"]
            )
        else:
            assert (
                levels["tp3"]
                < levels["tp2"]
                < levels["tp1"]
                < levels["entry"]
                < levels["stop_loss"]
            )

    final_release = validate_final_release(result)
    assert final_release["final_release_ready"] is True
    assert final_release["status"] == "RELEASE READY"
    assert final_release["decision"] == release_decision


def test_run_production_end_to_end_replays_idempotently(tmp_path):
    """Two consecutive executions (as CI steps on fresh stores) must both succeed:
    re-executing the same canonical decision after re-authorization is an
    idempotent no-op, not a conflicting replay (wall clock must not be identity)."""
    data = _committed_data()
    store_path = tmp_path / "decision_history.json"

    first = run_production_end_to_end(
        data,
        candidate_id=COMMITTED_CANDIDATE_ID,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        reference_now=_fresh_reference_now(data),
        store_path=store_path,
    )
    assert first["end_to_end_ready"] is True

    second = run_production_end_to_end(
        data,
        candidate_id=COMMITTED_CANDIDATE_ID,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        reference_now=_fresh_reference_now(data),
        store_path=store_path,
    )
    assert second["end_to_end_ready"] is True
    assert second["production_selection"] == first["production_selection"]


def test_run_production_end_to_end_rejects_unresolved_candidate(tmp_path):
    data = _committed_data()

    with pytest.raises(ValueError, match="No promoted candidate found"):
        run_production_end_to_end(
            data,
            candidate_id="cand_does_not_exist",
            results_dir=tmp_path,
            symbol="XAUUSD",
            interval="1d",
        )
