import pandas as pd

from src.evaluation.stability import (
    calculate_stability_score,
    get_stable_strategy,
)


def create_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "momentum",
                "total_return": 0.10,
                "max_drawdown": -0.08,
                "sharpe_ratio": 1.20,
                "positive_window_rate": 0.75,
            },
            {
                "strategy": "momentum",
                "total_return": 0.08,
                "max_drawdown": -0.07,
                "sharpe_ratio": 1.10,
                "positive_window_rate": 0.70,
            },
            {
                "strategy": "moving_average",
                "total_return": 0.05,
                "max_drawdown": -0.12,
                "sharpe_ratio": 0.70,
                "positive_window_rate": 0.60,
            },
            {
                "strategy": "moving_average",
                "total_return": 0.03,
                "max_drawdown": -0.15,
                "sharpe_ratio": 0.50,
                "positive_window_rate": 0.55,
            },
        ]
    )


def test_calculate_stability_score():
    result = calculate_stability_score(
        create_comparison()
    )

    assert not result.empty

    assert {
        "strategy",
        "experiments",
        "stability_score",
        "avg_total_return",
        "avg_sharpe_ratio",
        "avg_positive_window_rate",
    }.issubset(result.columns)

    assert result.iloc[0]["strategy"] == (
        "momentum"
    )


def test_get_stable_strategy():
    result = get_stable_strategy(
        create_comparison()
    )

    assert result == "momentum"


def test_empty_comparison():
    comparison = pd.DataFrame(
        columns=[
            "strategy",
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "positive_window_rate",
        ]
    )

    result = calculate_stability_score(
        comparison
    )

    assert result.empty
    assert get_stable_strategy(
        comparison
    ) is None


def test_invalid_comparison_type():
    try:
        calculate_stability_score([])
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Expected TypeError."
        )


def test_missing_columns():
    comparison = pd.DataFrame(
        {
            "strategy": ["momentum"],
        }
    )

    try:
        calculate_stability_score(comparison)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError."
        )
