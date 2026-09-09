# ============================================================
# AI-Trading-Lab/
#
# فایل جدید:
# live_snapshot_history.py
#
# فایل تست:
# tests/test_live_snapshot_history.py
# ============================================================


# =========================
# live_snapshot_history.py
# =========================

"""Persist and load an append-only history of live trading snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_HISTORY_PATH = Path("results/live/live_snapshot_history.jsonl")


def append_live_snapshot(
    snapshot: Mapping[str, Any],
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> Path:
    """Append one JSON-serializable live snapshot to a JSONL history file."""

    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serialized = json.dumps(
            dict(snapshot),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"snapshot is not JSON serializable: {exc}"
        ) from exc

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")

    return output_path


def load_live_snapshot_history(
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> list[dict[str, Any]]:
    """Load all valid snapshot records from a JSONL history file."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Snapshot history file not found: {input_path}"
        )

    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid snapshot history JSON at line {line_number}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Each snapshot history record must contain an object."
            )

        records.append(payload)

    return records


def latest_live_snapshot(
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> dict[str, Any]:
    """Return the most recently appended live snapshot."""

    records = load_live_snapshot_history(path)

    if not records:
        raise ValueError("Snapshot history contains no records.")

    return records[-1]


# ==============================
# tests/test_live_snapshot_history.py
# ==============================

from __future__ import annotations

import json

import pytest

from live_snapshot_history import (
    append_live_snapshot,
    latest_live_snapshot,
    load_live_snapshot_history,
)


def sample_snapshot(
    *,
    signal: int = 0,
    signal_label: str = "NO TRADE",
    timestamp: str = "2026-09-09T15:30:00+00:00",
) -> dict:
    return {
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": signal,
        "signal_label": signal_label,
        "trend": "UP",
        "strategy": "momentum",
        "momentum": -0.0053,
        "entry_price": 4395.596,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "momentum_window": 10,
        "fast_window": 20,
        "slow_window": 50,
        "timestamp": timestamp,
    }


def test_append_creates_parent_directory_and_file(tmp_path):
    path = tmp_path / "nested" / "live" / "history.jsonl"

    result = append_live_snapshot(
        sample_snapshot(),
        path,
    )

    assert result == path
    assert path.exists()
    assert path.parent.exists()


def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "history.jsonl"
    first = sample_snapshot()
    second = sample_snapshot(
        signal=1,
        signal_label="BUY",
        timestamp="2026-09-09T15:35:00+00:00",
    )

    append_live_snapshot(first, path)
    append_live_snapshot(second, path)

    records = load_live_snapshot_history(path)

    assert records == [first, second]


def test_history_is_append_only(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(sample_snapshot(), path)
    append_live_snapshot(
        sample_snapshot(
            timestamp="2026-09-09T15:35:00+00:00",
        ),
        path,
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert all(line.strip() for line in lines)


def test_latest_snapshot_returns_last_record(tmp_path):
    path = tmp_path / "history.jsonl"

    first = sample_snapshot(
        timestamp="2026-09-09T15:30:00+00:00",
    )
    second = sample_snapshot(
        signal=1,
        signal_label="BUY",
        timestamp="2026-09-09T15:35:00+00:00",
    )

    append_live_snapshot(first, path)
    append_live_snapshot(second, path)

    assert latest_live_snapshot(path) == second


def test_empty_history_has_no_latest_snapshot(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no records"):
        latest_live_snapshot(path)


def test_missing_history_raises_file_not_found(tmp_path):
    path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        load_live_snapshot_history(path)


def test_invalid_snapshot_input_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"

    with pytest.raises(ValueError, match="mapping"):
        append_live_snapshot(["invalid"], path)


def test_non_serializable_snapshot_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"

    snapshot = sample_snapshot()
    snapshot["bad_value"] = object()

    with pytest.raises(ValueError, match="not JSON serializable"):
        append_live_snapshot(snapshot, path)


def test_invalid_json_line_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        '{"symbol": "XAUUSD"}\n'
        '{"broken": \n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_live_snapshot_history(path)


def test_non_object_record_is_rejected(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        '{"symbol": "XAUUSD"}\n'
        '[1, 2, 3]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain an object"):
        load_live_snapshot_history(path)


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "history.jsonl"

    path.write_text(
        "\n"
        + json.dumps(sample_snapshot())
        + "\n\n",
        encoding="utf-8",
    )

    records = load_live_snapshot_history(path)

    assert records == [sample_snapshot()]
