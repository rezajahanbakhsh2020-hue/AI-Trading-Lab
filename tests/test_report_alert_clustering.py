from __future__ import annotations

import pytest

from src.evaluation.report_alert_clustering import (
    build_alert_clustering_summary,
    calculate_alert_clusters,
    calculate_clustered_snapshots,
)


def test_calculate_alert_clusters():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    assert calculate_alert_clusters(history) == 3


def test_calculate_alert_clusters_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_clusters(history) == 1


def test_calculate_alert_clusters_no_active():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_clusters(history) == 0


def test_calculate_alert_clusters_custom_threshold():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_alert_clusters(
        history,
        threshold=3,
    ) == 2


def test_calculate_alert_clusters_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    assert calculate_alert_clusters(history) == 1


def test_calculate_alert_clusters_empty():
    with pytest.raises(ValueError):
        calculate_alert_clusters([])


def test_calculate_clustered_snapshots():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    assert calculate_clustered_snapshots(history) == 5


def test_calculate_clustered_snapshots_custom_threshold():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_clustered_snapshots(
        history,
        threshold=3,
    ) == 3


def test_calculate_clustered_snapshots_empty():
    with pytest.raises(ValueError):
        calculate_clustered_snapshots([])


def test_build_alert_clustering_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 2},
    ]

    result = build_alert_clustering_summary(history)

    assert result == {
        "snapshot_count": 7,
        "alert_clusters": 3,
        "clustered_snapshots": 5,
        "threshold": 1,
    }


def test_build_alert_clustering_summary_custom_threshold():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    result = build_alert_clustering_summary(
        history,
        threshold=3,
    )

    assert result == {
        "snapshot_count": 5,
        "alert_clusters": 2,
        "clustered_snapshots": 3,
        "threshold": 3,
    }


def test_build_alert_clustering_summary_empty():
    with pytest.raises(ValueError):
        build_alert_clustering_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_alert_clusters({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_alert_clusters(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_clusters(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_clusters(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_clusters(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_alert_clusters(
            [{"alert_count": -1}]
        )


def test_rejects_invalid_threshold_type():
    history = [{"alert_count": 1}]

    with pytest.raises(ValueError):
        calculate_alert_clusters(
            history,
            threshold=1.5,
        )


def test_rejects_boolean_threshold():
    history = [{"alert_count": 1}]

    with pytest.raises(ValueError):
        calculate_alert_clusters(
            history,
            threshold=True,
        )


def test_rejects_negative_threshold():
    history = [{"alert_count": 1}]

    with pytest.raises(ValueError):
        calculate_alert_clusters(
            history,
            threshold=-1,
        )
