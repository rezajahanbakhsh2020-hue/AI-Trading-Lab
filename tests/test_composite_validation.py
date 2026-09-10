import pandas as pd
import pytest

from src.evaluation.composite_validation import (
    CompositeValidation,
    require_accepted_composite,
    validate_composite,
    validate_composite_from_evaluations,
)


def test_composite_is_accepted_when_it_beats_best_component():
    result = validate_composite(
        composite_score=0.90,
        component_scores=[0.70, 0.75, 0.80],
    )

    assert isinstance(result, CompositeValidation)
    assert result.accepted is True
    assert result.best_component_score == 0.80
    assert result.mean_component_score == pytest.approx(
        0.75
    )
    assert result.improvement_vs_best == pytest.approx(
        0.10
    )


def test_composite_is_rejected_when_best_component_is_better():
    result = validate_composite(
        composite_score=0.70,
        component_scores=[0.80, 0.60, 0.50],
    )

    assert result.accepted is False
    assert result.best_component_score == 0.80
    assert result.improvement_vs_best == pytest.approx(
        -0.10
    )


def test_minimum_improvement_is_enforced():
    result = validate_composite(
        composite_score=0.81,
        component_scores=[0.80, 0.70, 0.60],
        minimum_improvement=0.02,
    )

    assert result.accepted is False

    result = validate_composite(
        composite_score=0.83,
        component_scores=[0.80, 0.70, 0.60],
        minimum_improvement=0.02,
    )

    assert result.accepted is True


def test_equal_score_is_accepted_when_zero_margin_is_required():
    result = validate_composite(
        composite_score=0.80,
        component_scores=[0.80, 0.70, 0.60],
    )

    assert result.accepted is True
    assert result.improvement_vs_best == 0.0


def test_validation_from_evaluations_uses_top_n():
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
                0.70,
                0.60,
                0.95,
            ],
        }
    )

    result = validate_composite_from_evaluations(
        composite_score=0.85,
        evaluations=evaluations,
        top_n=3,
    )

    assert result.best_component_score == 0.80
    assert result.accepted is True


def test_require_accepted_composite_returns_valid_result():
    validation = validate_composite(
        composite_score=0.90,
        component_scores=[0.80, 0.70],
    )

    result = require_accepted_composite(validation)

    assert result is validation


def test_require_accepted_composite_rejects_failed_validation():
    validation = validate_composite(
        composite_score=0.70,
        component_scores=[0.80, 0.75],
    )

    with pytest.raises(ValueError, match="rejected"):
        require_accepted_composite(validation)


def test_empty_components_are_rejected():
    with pytest.raises(ValueError):
        validate_composite(
            composite_score=0.80,
            component_scores=[],
        )


def test_nan_scores_are_rejected():
    with pytest.raises(ValueError):
        validate_composite(
            composite_score=float("nan"),
            component_scores=[0.80, 0.70],
        )

    with pytest.raises(ValueError):
        validate_composite(
            composite_score=0.80,
            component_scores=[0.80, float("nan")],
        )


def test_invalid_top_n_is_rejected():
    evaluations = pd.DataFrame(
        {
            "name": ["a"],
            "ranking_score": [0.8],
        }
    )

    with pytest.raises(ValueError):
        validate_composite_from_evaluations(
            composite_score=0.9,
            evaluations=evaluations,
            top_n=0,
        )


def test_missing_score_column_is_rejected():
    evaluations = pd.DataFrame(
        {
            "name": ["a"],
            "score": [0.8],
        }
    )

    with pytest.raises(ValueError):
        validate_composite_from_evaluations(
            composite_score=0.9,
            evaluations=evaluations,
        )


def test_invalid_composite_name_is_rejected():
    with pytest.raises(ValueError):
        validate_composite(
            composite_score=0.9,
            component_scores=[0.8],
            composite_name=" ",
        )
