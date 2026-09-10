import numpy as np
import pandas as pd
import pytest

from src.evaluation.composite_walk_forward import (
    CompositeWalkForwardResult,
    run_composite_walk_forward,
)
from src.strategies.registry import DEFAULT_REGISTRY


def make_data(size: int = 180) -> pd.DataFrame:
    close = pd.Series(
        np.linspace(100.0, 190.0, size)
        + np.sin(np.arange(size) / 5.0),
        name="close",
    )

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=size,
                freq="D",
            ),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def test_composite_walk_forward_creates_oos_folds():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=2,
    )

    assert isinstance(
        result,
        CompositeWalkForwardResult,
    )
    assert result.fold_count == 6
    assert len(result.oos_evaluations) == 6


def test_each_fold_has_selected_strategies():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=3,
    )

    for fold in result.folds:
        assert len(fold.selected_strategies) == 3
        assert len(set(fold.selected_strategies)) == 3


def test_test_window_is_after_training_window():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=2,
    )

    for fold in result.folds:
        assert fold.train_start < fold.train_end
        assert fold.train_end == fold.test_start
        assert fold.test_start < fold.test_end


def test_walk_forward_returns_are_finite():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=2,
    )

    assert np.isfinite(result.oos_total_return)
    assert np.isfinite(result.mean_fold_return)
    assert np.isfinite(result.return_std)
    assert np.isfinite(result.positive_fold_rate)
    assert np.isfinite(result.mean_sharpe)


def test_positive_fold_rate_is_between_zero_and_one():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=2,
    )

    assert 0.0 <= result.positive_fold_rate <= 1.0


def test_acceptance_threshold_is_applied():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=2,
        minimum_positive_fold_rate=0.0,
    )

    assert result.accepted is True


def test_custom_registry_is_supported():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=1,
        strategy_registry=DEFAULT_REGISTRY,
    )

    assert result.fold_count > 0


def test_invalid_train_size_is_rejected():
    with pytest.raises(ValueError):
        run_composite_walk_forward(
            make_data(),
            train_size=0,
            test_size=20,
        )


def test_invalid_test_size_is_rejected():
    with pytest.raises(ValueError):
        run_composite_walk_forward(
            make_data(),
            train_size=80,
            test_size=0,
        )


def test_invalid_top_n_is_rejected():
    with pytest.raises(ValueError):
        run_composite_walk_forward(
            make_data(),
            train_size=80,
            test_size=20,
            top_n=0,
        )


def test_no_fold_configuration_is_rejected():
    with pytest.raises(ValueError):
        run_composite_walk_forward(
            make_data(80),
            train_size=60,
            test_size=30,
        )


def test_invalid_positive_fold_rate_is_rejected():
    with pytest.raises(ValueError):
        run_composite_walk_forward(
            make_data(),
            train_size=80,
            test_size=20,
            minimum_positive_fold_rate=1.5,
        )


def test_as_dict_contains_oos_metrics():
    result = run_composite_walk_forward(
        make_data(),
        train_size=80,
        test_size=20,
        top_n=2,
    )

    payload = result.as_dict()

    assert payload["name"] == "ensemble"
    assert payload["fold_count"] == result.fold_count
    assert payload["oos_total_return"] == pytest.approx(
        result.oos_total_return
    )
    assert "accepted" in payload
