import pandas as pd
import pytest

from src.evaluation.portfolio_selection import (
    build_strategy_portfolio,
    calculate_portfolio_score,
    get_portfolio_strategy_names,
    rank_portfolio_candidates,
    select_portfolio_candidates,
)


def make_report():
    return pd.DataFrame(
        {
            "strategy": [
                "gold_trend",
                "gold_momentum",
                "gold_mean_reversion",
                "fragile_strategy",
                "weak_strategy",
            ],
            "total_return": [
                0.32,
                0.28,
                0.22,
                0.70,
                0.05,
            ],
            "max_drawdown": [
                -0.09,
                -0.11,
                -0.13,
                -0.40,
                -0.04,
            ],
            "sharpe_ratio": [
                1.45,
                1.30,
                1.10,
                1.20,
                0.20,
            ],
            "stability_score": [
                0.88,
                0.82,
                0.76,
                0.35,
                0.40,
            ],
            "worst_window_return": [
                -0.04,
                -0.06,
                -0.08,
                -0.35,
                -0.02,
            ],
            "max_consecutive_losses": [
                1,
                2,
                3,
                7,
                2,
            ],
        }
    )


def test_portfolio_score_is_created():
    scored = calculate_portfolio_score(
        make_report()
    )

    assert "portfolio_score" in scored.columns
    assert scored["portfolio_score"].notna().all()


def test_input_is_not_modified():
    report = make_report()
    original = report.copy(deep=True)

    calculate_portfolio_score(report)

    pd.testing.assert_frame_equal(
        report,
        original,
    )


def test_ranking_orders_candidates():
    ranked = rank_portfolio_candidates(
        make_report()
    )

    assert ranked.iloc[0]["strategy"] == "gold_trend"
    assert "portfolio_score" in ranked.columns


def test_risk_filters_remove_fragile_strategy():
    selected = select_portfolio_candidates(
        make_report()
    )

    names = selected["strategy"].tolist()

    assert "fragile_strategy" not in names
    assert "weak_strategy" not in names


def test_portfolio_is_limited():
    portfolio = build_strategy_portfolio(
        make_report(),
        max_strategies=2,
    )

    assert len(portfolio) == 2
    assert list(portfolio["strategy"]) == [
        "gold_trend",
        "gold_momentum",
    ]


def test_portfolio_names():
    names = get_portfolio_strategy_names(
        make_report(),
        max_strategies=3,
    )

    assert names == [
        "gold_trend",
        "gold_momentum",
        "gold_mean_reversion",
    ]


def test_custom_max_strategies():
    names = get_portfolio_strategy_names(
        make_report(),
        max_strategies=1,
    )

    assert names == ["gold_trend"]


def test_empty_report():
    report = make_report().iloc[0:0]

    portfolio = build_strategy_portfolio(
        report
    )

    assert portfolio.empty


def test_no_strategy_passes():
    portfolio = build_strategy_portfolio(
        make_report(),
        min_stability_score=0.99,
    )

    assert portfolio.empty


def test_invalid_max_strategies():
    with pytest.raises(ValueError):
        build_strategy_portfolio(
            make_report(),
            max_strategies=0,
        )


def test_invalid_max_drawdown():
    with pytest.raises(ValueError):
        select_portfolio_candidates(
            make_report(),
            max_drawdown=-0.1,
        )


def test_invalid_sharpe_threshold():
    with pytest.raises(ValueError):
        select_portfolio_candidates(
            make_report(),
            min_sharpe_ratio=-1.0,
        )


def test_invalid_stability_threshold():
    with pytest.raises(ValueError):
        select_portfolio_candidates(
            make_report(),
            min_stability_score=1.2,
        )


def test_missing_required_column():
    report = make_report().drop(
        columns=["stability_score"]
    )

    with pytest.raises(ValueError):
        calculate_portfolio_score(report)
