from __future__ import annotations

import pytest

from src.evaluation.report_alert_sequence import (
    build_alert_sequence_summary,
    calculate_alert_sequences,
    calculate_longest_alert_sequence,
)


def test_calculate_alert_sequences():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert calculate_alert_sequences(history) == 3


def test_calculate_alert_sequences_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_sequences(history) == 1


def test_calculate_alert_sequences_none_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_sequences(history) == 0


def test_calculate_alert_sequences_single_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
    ]

    assert calculate_alert_sequences(history) == 1


def test_calculate_alert_sequences_empty():
    with pytest.raises(ValueError):
        calculate_alert_sequences([])


def test_calculate_longest_alert_sequence():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    assert calculate_longest_alert_sequence(history) == 3


def test_calculate_longest_alert_sequence_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_longest_alert_sequence(history) == 4


def test_calculate_longest_alert_sequence_none_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_longest_alert_sequence(history) == 0


def test_calculate_longest_alert_sequence_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    assert calculate_longest_alert_sequence(history) == 1


def test_calculate_longest_alert_sequence_empty():
    with pytest.raises(ValueError):
        calculate_longest_alert_sequence([])


def test_build_alert_sequence_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    result = build_alert_sequence_summary(history)

    assert result == {
        "snapshot_count": 7,
        "alert_sequences": 3,
        "longest_alert_sequence": 2,
    }


def test_build_alert_sequence_summary_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    result = build_alert_sequence_summary(history)

    assert result == {
        "snapshot_count": 3,
        "alert_sequences": 1,
        "longest_alert_sequence": 3,
    }


def test_build_alert_sequence_summary_empty():
    with pytest.raises(ValueError):
        build_alert_sequence_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_sequences({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_sequences(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_sequences(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_sequences(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_sequences(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_sequences(
            [{"alert_count": -1}]
        )
