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


def test_run_production_end_to_end_real_store_and_data(tmp_path):
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

    # The committed dataset predates now: freshness must fail closed and the
    # runtime must refuse to trade historical data as live.
    assert result["decision"]["decision"] == "NO TRADE"
    assert result["display"]["entry_price"] is None
    assert result["display"]["stop_loss"] is None
    assert result["display"]["tp1"] is None
    assert result["display"]["tp2"] is None
    assert result["display"]["tp3"] is None

    assert result["end_to_end_ready"] is True
    assert result["release_gate"]["release_ready"] is True


def test_run_production_end_to_end_releases_and_replays_idempotently(tmp_path):
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
        reference_now=pd.Timestamp("2026-10-26T05:00:00Z"),
        store_path=store_path,
    )
    assert first["end_to_end_ready"] is True

    second = run_production_end_to_end(
        data,
        candidate_id=COMMITTED_CANDIDATE_ID,
        results_dir=tmp_path,
        symbol="XAUUSD",
        interval="1d",
        reference_now=pd.Timestamp("2026-10-26T05:07:00Z"),
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
