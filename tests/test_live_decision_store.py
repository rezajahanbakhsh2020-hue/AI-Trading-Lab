from __future__ import annotations

import json

import pytest

from src.evaluation.live_decision_store import (
    append_live_decision_to_store,
    load_live_decision_history,
    save_live_decision_history,
)


def _record(
    timestamp: str,
    signal_label: str = "BUY",
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_save_live_decision_history_writes_json(tmp_path):
    path = tmp_path / "decisions.json"
    history = [
        _record("2026-09-10T05:45:00+00:00"),
    ]

    result = save_live_decision_history(history, path)

    assert result == path
    assert path.exists()

    stored = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert len(stored) == 1
    assert stored[0]["symbol"] == "XAUUSD"
    assert stored[0]["signal_label"] == "BUY"


def test_load_live_decision_history_restores_records(tmp_path):
    path = tmp_path / "decisions.json"
    history = [
        _record("2026-09-10T05:45:00+00:00"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
        ),
    ]

    save_live_decision_history(history, path)

    loaded = load_live_decision_history(path)

    assert loaded == history


def test_save_live_decision_history_creates_parent_directory(
    tmp_path,
):
    path = tmp_path / "nested" / "decisions.json"

    save_live_decision_history(
        [_record("2026-09-10T05:45:00+00:00")],
        path,
    )

    assert path.exists()


def test_append_live_decision_to_store_creates_new_store(
    tmp_path,
):
    path = tmp_path / "decisions.json"
    record = _record("2026-09-10T05:45:00+00:00")

    history = append_live_decision_to_store(
        record,
        path,
    )

    assert history == [record]

    loaded = load_live_decision_history(path)

    assert loaded == [record]


def test_append_live_decision_to_store_preserves_existing_history(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    first = _record("2026-09-10T05:45:00+00:00")
    second = _record(
        "2026-09-10T05:50:00+00:00",
        "NO TRADE",
    )

    save_live_decision_history([first], path)

    history = append_live_decision_to_store(
        second,
        path,
    )

    assert history == [first, second]
    assert load_live_decision_history(path) == [
        first,
        second,
    ]


def test_store_does_not_recalculate_trading_values(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    record = _record("2026-09-10T05:45:00+00:00")
    record["entry_price"] = 9999.0
    record["stop_loss"] = 9000.0
    record["take_profit"] = 12000.0

    save_live_decision_history([record], path)

    loaded = load_live_decision_history(path)

    assert loaded[0]["entry_price"] == 9999.0
    assert loaded[0]["stop_loss"] == 9000.0
    assert loaded[0]["take_profit"] == 12000.0


def test_load_live_decision_history_rejects_invalid_json_shape(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    path.write_text(
        json.dumps({"record": "invalid"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_live_decision_history(path)


def test_load_live_decision_history_rejects_missing_file(
    tmp_path,
):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_live_decision_history(path)


def test_save_live_decision_history_rejects_invalid_history(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    with pytest.raises(TypeError):
        save_live_decision_history(
            "invalid",
            path,
        )


def test_save_live_decision_history_rejects_invalid_record(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    with pytest.raises(TypeError):
        save_live_decision_history(
            [[]],
            path,
        )


def test_append_live_decision_to_store_rejects_invalid_record(
    tmp_path,
):
    path = tmp_path / "decisions.json"

    with pytest.raises(TypeError):
        append_live_decision_to_store(
            [],
            path,
        )


def _worker_append(args):
    rec, path = args
    return append_live_decision_to_store(rec, path)


def test_concurrent_append_preservation(tmp_path):
    import concurrent.futures

    path = tmp_path / "decisions.json"

    rec1 = _record("2026-09-10T05:45:00+00:00", "BUY")
    rec1["decision_id"] = "dec-1"
    rec1["signal_id"] = "sig-1"

    rec2 = _record("2026-09-10T05:50:00+00:00", "NO TRADE")
    rec2["decision_id"] = "dec-2"
    rec2["signal_id"] = "sig-2"

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_worker_append, (rec1, path))
        f2 = executor.submit(_worker_append, (rec2, path))
        f1.result()
        f2.result()

    loaded = load_live_decision_history(path)
    assert len(loaded) == 2
    dec_ids = {r.get("decision_id") for r in loaded}
    assert dec_ids == {"dec-1", "dec-2"}


def test_concurrent_identical_replay(tmp_path):
    import concurrent.futures

    path = tmp_path / "decisions.json"

    rec = _record("2026-09-10T05:45:00+00:00", "BUY")
    rec["decision_id"] = "dec-same"
    rec["signal_id"] = "sig-same"

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_worker_append, (rec, path))
        f2 = executor.submit(_worker_append, (rec, path))
        f1.result()
        f2.result()

    loaded = load_live_decision_history(path)
    assert len(loaded) == 1


def test_atomic_write_failure(tmp_path, monkeypatch):
    import os

    path = tmp_path / "decisions.json"
    initial_rec = _record("2026-09-10T05:00:00+00:00", "BUY")
    initial_rec["decision_id"] = "dec-initial"
    save_live_decision_history([initial_rec], path)

    # Mock os.replace to simulate failure (e.g. disk/filesystem error during atomic swap)
    def mock_replace(src, dst):
        raise OSError("Simulated replacement failure")

    monkeypatch.setattr(os, "replace", mock_replace)

    new_rec = _record("2026-09-10T05:45:00+00:00", "NO TRADE")
    new_rec["decision_id"] = "dec-fail"

    with pytest.raises(OSError, match="Simulated replacement failure"):
        append_live_decision_to_store(new_rec, path)

    # Original file must remain intact and valid JSON
    loaded = load_live_decision_history(path)
    assert len(loaded) == 1
    assert loaded[0]["decision_id"] == "dec-initial"

    # Temporary file should be cleaned up
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_existing_history_preservation(tmp_path):
    import concurrent.futures

    path = tmp_path / "decisions.json"

    # Create pre-existing history records
    pre1 = _record("2026-09-10T05:00:00+00:00", "BUY")
    pre1["decision_id"] = "dec-pre1"
    pre1["signal_id"] = "sig-pre1"

    pre2 = _record("2026-09-10T05:15:00+00:00", "NO TRADE")
    pre2["decision_id"] = "dec-pre2"
    pre2["signal_id"] = "sig-pre2"

    save_live_decision_history([pre1, pre2], path)

    rec1 = _record("2026-09-10T05:45:00+00:00", "BUY")
    rec1["decision_id"] = "dec-new1"
    rec1["signal_id"] = "sig-new1"

    rec2 = _record("2026-09-10T05:50:00+00:00", "NO TRADE")
    rec2["decision_id"] = "dec-new2"
    rec2["signal_id"] = "sig-new2"

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_worker_append, (rec1, path))
        f2 = executor.submit(_worker_append, (rec2, path))
        f1.result()
        f2.result()

    loaded = load_live_decision_history(path)
    assert len(loaded) == 4
    # Check that original order of pre-existing records is intact
    assert loaded[0]["decision_id"] == "dec-pre1"
    assert loaded[1]["decision_id"] == "dec-pre2"
    all_ids = [r.get("decision_id") for r in loaded]
    assert set(all_ids) == {"dec-pre1", "dec-pre2", "dec-new1", "dec-new2"}


def test_concurrent_conflicting_replay(tmp_path):
    import concurrent.futures

    path = tmp_path / "decisions.json"

    rec1 = _record("2026-09-10T05:45:00+00:00", "BUY")
    rec1["decision_id"] = "dec-conflict"
    rec1["signal_id"] = "sig-conflict"

    rec2 = _record("2026-09-10T05:45:00+00:00", "NO TRADE")
    rec2["decision_id"] = "dec-conflict"
    rec2["signal_id"] = "sig-conflict"

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_worker_append, (rec1, path))
        f2 = executor.submit(_worker_append, (rec2, path))
        res1 = f1.exception()
        res2 = f2.exception()

    # Exactly one worker must succeed and the other must fail closed with ValueError
    exceptions = [e for e in (res1, res2) if e is not None]
    assert len(exceptions) == 1
    assert isinstance(exceptions[0], ValueError)
    assert "Conflicting replay detected" in str(exceptions[0])

    loaded = load_live_decision_history(path)
    assert len(loaded) == 1
