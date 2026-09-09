"""Tests for the human-readable live proof report."""

from __future__ import annotations

import json

import pytest

from live_proof_report import (
    build_live_proof_html,
    build_report_from_snapshot,
    save_live_proof_report,
)


def make_snapshot() -> dict:
    """Return a representative persisted live snapshot."""
    return {
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "momentum": 0.0013,
        "entry_price": 4421.063,
        "stop_loss": 4376.85237,
        "take_profit": 4509.48426,
        "risk_reward_ratio": 2.0,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "momentum_window": 10,
        "fast_window": 20,
        "slow_window": 50,
        "timestamp": "2026-09-09T14:45:00+00:00",
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "latest_bid": 4421.106,
        "latest_ask": 4421.288,
        "latest_mid": 4421.197,
        "candle_count": 201,
    }


def test_build_live_proof_html_contains_core_live_output():
    html = build_live_proof_html(make_snapshot())

    assert "<!DOCTYPE html>" in html
    assert "XAUUSD" in html
    assert "BUY" in html
    assert "UP" in html
    assert "4421.063" in html
    assert "4376.85237" in html
    assert "4509.48426" in html
    assert "open" in html
    assert "LIVE" in html
    assert "201" in html


def test_build_live_proof_html_escapes_html_content():
    snapshot = make_snapshot()
    snapshot["market_state"] = "<script>alert('x')</script>"

    html = build_live_proof_html(snapshot)

    assert "<script>alert('x')</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_live_proof_html_supports_no_trade_state():
    snapshot = make_snapshot()
    snapshot["signal"] = 0
    snapshot["signal_label"] = "NO TRADE"
    snapshot["stop_loss"] = None
    snapshot["take_profit"] = None
    snapshot["risk_reward_ratio"] = None

    html = build_live_proof_html(snapshot)

    assert "NO TRADE" in html
    assert "N/A" in html


def test_build_live_proof_html_marks_stale_quote():
    snapshot = make_snapshot()
    snapshot["quote_stale"] = True

    html = build_live_proof_html(snapshot)

    assert "STALE" in html
    assert "LIVE" not in html


def test_save_live_proof_report_creates_html_file(tmp_path):
    output_path = tmp_path / "proof" / "live_proof.html"

    result = save_live_proof_report(
        make_snapshot(),
        output_path,
    )

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    assert "XAUUSD" in content
    assert "BUY" in content


def test_build_report_from_snapshot_reads_persisted_json(tmp_path):
    snapshot_path = tmp_path / "live_snapshot.json"
    report_path = tmp_path / "live_proof.html"

    snapshot_path.write_text(
        json.dumps(make_snapshot()),
        encoding="utf-8",
    )

    result = build_report_from_snapshot(
        snapshot_path,
        report_path,
    )

    assert result == report_path
    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")

    assert "XAUUSD" in content
    assert "BUY" in content
    assert "UP" in content


def test_build_live_proof_html_rejects_missing_required_fields():
    snapshot = make_snapshot()
    del snapshot["trend"]

    with pytest.raises(
        ValueError,
        match="Missing required snapshot fields",
    ):
        build_live_proof_html(snapshot)


def test_build_live_proof_html_rejects_non_mapping():
    with pytest.raises(
        ValueError,
        match="snapshot must be a mapping",
    ):
        build_live_proof_html(None)
