import pandas as pd
import pytest

from src.strategies.ensemble import (
    build_ensemble_from_evaluations,
    build_weighted_ensemble,
    equal_weights,
    ranking_weights,
)


def strategy_frame(values):
    return pd.DataFrame(
        {
            "signal": values,
        },
        index=range(len(values)),
    )


def test_equal_weights():
    result = equal_weights(
        ["momentum", "baseline", "breakout"]
    )

    assert result == {
        "momentum": 1.0,
        "baseline": 1.0,
        "breakout": 1.0,
    }


def test_equal_weights_rejects_duplicates():
    with pytest.raises(ValueError):
        equal_weights(
            ["momentum", "momentum"]
        )


def test_weighted_ensemble_requires_consensus():
    outputs = {
        "a": strategy_frame([1, 1, 0, 0]),
        "b": strategy_frame([1, 0, 0, 1]),
    }

    result = build_weighted_ensemble(
        outputs,
        {"a": 1.0, "b": 1.0},
    )

    assert result["signal"].tolist() == [
        1,
        1,
        0,
        1,
    ]


def test_weighted_ensemble_preserves_score():
    outputs = {
        "a": strategy_frame([1, 0]),
        "b": strategy_frame([0, 1]),
    }

    result = build_weighted_ensemble(
        outputs,
        {"a": 3.0, "b": 1.0},
    )

    assert result["ensemble_score"].tolist() == [
        0.75,
        0.25,
    ]

    assert result["signal"].tolist() == [
        1,
        1,
    ]


def test_zero_consensus_produces_no_trade_signal():
    outputs = {
        "a": strategy_frame([1, 0]),
        "b": strategy_frame([0, 0]),
    }

    result = build_weighted_ensemble(
        outputs,
        {"a": 1.0, "b": 1.0},
    )

    assert result["signal"].tolist() == [
        1,
        0,
    ]


def test_ranking_weights_use_top_n():
    evaluations = pd.DataFrame(
        {
            "name": [
                "momentum",
                "baseline",
                "breakout",
                "other",
            ],
            "ranking_score": [
                0.80,
                0.60,
                0.40,
                0.20,
            ],
        }
    )

    result = ranking_weights(
        evaluations,
        top_n=3,
    )

    assert set(result) == {
        "momentum",
        "baseline",
        "breakout",
    }

    assert result["momentum"] == 0.80
    assert result["baseline"] == 0.60
    assert result["breakout"] == 0.40


def test_negative_scores_fall_back_to_equal_weights():
    evaluations = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "ranking_score": [-0.5, -0.2, -0.1],
        }
    )

    result = ranking_weights(
        evaluations,
        top_n=3,
    )

    assert result == {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
    }


def test_build_ensemble_from_evaluations():
    outputs = {
        "momentum": strategy_frame([1, 1, 0]),
        "baseline": strategy_frame([1, 0, 0]),
        "breakout": strategy_frame([0, 1, 0]),
    }

    evaluations = pd.DataFrame(
        {
            "name": [
                "momentum",
                "baseline",
                "breakout",
            ],
            "ranking_score": [
                0.8,
                0.6,
                0.4,
            ],
        }
    )

    result = build_ensemble_from_evaluations(
        outputs,
        evaluations,
        top_n=3,
    )

    assert len(result) == 3
    assert "ensemble_score" in result.columns
    assert "signal" in result.columns


def test_missing_strategy_output_is_rejected():
    outputs = {
        "momentum": strategy_frame([1, 0]),
    }

    with pytest.raises(ValueError):
        build_weighted_ensemble(
            outputs,
            {
                "momentum": 1.0,
                "baseline": 1.0,
            },
        )


def test_different_lengths_are_rejected():
    outputs = {
        "a": strategy_frame([1, 0]),
        "b": strategy_frame([1, 0, 1]),
    }

    with pytest.raises(ValueError):
        build_weighted_ensemble(
            outputs,
            {"a": 1.0, "b": 1.0},
        )


def test_invalid_weights_are_rejected():
    outputs = {
        "a": strategy_frame([1, 0]),
    }

    with pytest.raises(ValueError):
        build_weighted_ensemble(
            outputs,
            {"a": -1.0},
        )

    with pytest.raises(ValueError):
        build_weighted_ensemble(
            outputs,
            {"a": 0.0},
        )


def test_empty_strategy_outputs_are_rejected():
    with pytest.raises(ValueError):
        build_weighted_ensemble(
            {},
            {"a": 1.0},
        )
