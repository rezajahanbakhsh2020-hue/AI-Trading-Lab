import pandas as pd
import pytest

from src.evaluation.walk_forward import (
    WalkForwardWindow,
    generate_walk_forward_windows,
)


def create_sample_data(size: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=size,
                freq="D",
            ),
            "close": range(size),
        }
    )


def test_walk_forward_creates_chronological_windows():
    df = create_sample_data(20)

    windows = generate_walk_forward_windows(
        df,
        train_size=10,
        test_size=5,
    )

    assert windows == [
        WalkForwardWindow(
            train_start=0,
            train_end=10,
            test_start=10,
            test_end=15,
        ),
        WalkForwardWindow(
            train_start=5,
            train_end=15,
            test_start=15,
            test_end=20,
        ),
    ]


def test_walk_forward_never_uses_future_data_in_training():
    df = create_sample_data(30)

    windows = generate_walk_forward_windows(
        df,
        train_size=10,
        test_size=5,
    )

    for window in windows:
        assert window.train_end <= window.test_start
        assert window.train_start < window.train_end
        assert window.test_start < window.test_end


def test_walk_forward_default_step_equals_test_size():
    df = create_sample_data(30)

    windows = generate_walk_forward_windows(
        df,
        train_size=10,
        test_size=5,
    )

    assert windows[0].test_start == 10
    assert windows[1].test_start == 15
    assert windows[2].test_start == 20


def test_walk_forward_supports_custom_step():
    df = create_sample_data(30)

    windows = generate_walk_forward_windows(
        df,
        train_size=10,
        test_size=5,
        step=2,
    )

    assert windows[0].train_start == 0
    assert windows[1].train_start == 2
    assert windows[2].train_start == 4


def test_walk_forward_returns_empty_when_data_is_too_short():
    df = create_sample_data(14)

    windows = generate_walk_forward_windows(
        df,
        train_size=10,
        test_size=5,
    )

    assert windows == []


def test_walk_forward_rejects_invalid_sizes():
    df = create_sample_data(20)

    with pytest.raises(ValueError):
        generate_walk_forward_windows(
            df,
            train_size=0,
            test_size=5,
        )

    with pytest.raises(ValueError):
        generate_walk_forward_windows(
            df,
            train_size=10,
            test_size=0,
        )

    with pytest.raises(ValueError):
        generate_walk_forward_windows(
            df,
            train_size=10,
            test_size=5,
            step=0,
        )


def test_walk_forward_requires_dataframe():
    with pytest.raises(TypeError):
        generate_walk_forward_windows(
            [1, 2, 3],
            train_size=2,
            test_size=1,
        )
