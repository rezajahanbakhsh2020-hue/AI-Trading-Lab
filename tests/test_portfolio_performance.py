import pandas as pd
import pytest

from src.evaluation.portfolio_performance import (
    build_portfolio_performance,
    calculate_max_drawdown,
    calculate_portfolio_drawdown,
    calculate_portfolio_equity,
    calculate_portfolio_returns,
)


def make_strategy_results():
    timestamps = pd.date_range(
        "2026-01-01",
        periods=4,
        freq="D",
    )

    trend = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_return": [
                0.10,
                0.00,
                -0.05,
                0.10,
            ],
        }
    )

    momentum = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_return": [
                0.00,
                0.10,
                0.00,
                -0.05,
            ],
        }
    )

    return {
        "trend": trend,
        "momentum": momentum,
    }


def test_portfolio_returns_are_weighted():
    result = calculate_portfolio_returns(
        make_strategy_results(),
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
    )

    assert result["portfolio_return"].tolist() == [
        pytest.approx(0.06),
        pytest.approx(0.04),
        pytest.approx(-0.03),
        pytest.approx(0.04),
    ]


def test_portfolio_equity_is_calculated():
    returns = calculate_portfolio_returns(
        make_strategy_results(),
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
    )

    result = calculate_portfolio_equity(
        returns,
        initial_capital=10000.0,
    )

    assert result["equity"].iloc[0] == pytest.approx(
        10600.0
    )


def test_equity_uses_compounding():
    returns = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=2,
                freq="D",
            ),
            "portfolio_return": [
                0.10,
                0.10,
            ],
        }
    )

    result = calculate_portfolio_equity(
        returns,
        initial_capital=10000.0,
    )

    assert result["equity"].iloc[-1] == pytest.approx(
        12100.0
    )


def test_drawdown_is_calculated():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
            "equity": [
                10000.0,
                12000.0,
                9000.0,
            ],
        }
    )

    result = calculate_portfolio_drawdown(
        equity
    )

    assert result["equity_peak"].tolist() == [
        10000.0,
        12000.0,
        12000.0,
    ]

    assert result["drawdown"].iloc[-1] == pytest.approx(
        -0.25
    )


def test_max_drawdown():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),
            "equity": [
                10000.0,
                12000.0,
                9000.0,
                11000.0,
            ],
        }
    )

    assert calculate_max_drawdown(
        equity
    ) == pytest.approx(-0.25)


def test_partial_weights_leave_cash_unallocated():
    result = calculate_portfolio_returns(
        make_strategy_results(),
        {
            "trend": 0.50,
            "momentum": 0.30,
        },
    )

    assert result["portfolio_return"].iloc[0] == pytest.approx(
        0.05
    )


def test_missing_strategy_weight_rejected():
    with pytest.raises(ValueError):
        calculate_portfolio_returns(
            make_strategy_results(),
            {
                "trend": 1.0,
            },
        )


def test_weights_above_one_rejected():
    with pytest.raises(ValueError):
        calculate_portfolio_returns(
            make_strategy_results(),
            {
                "trend": 0.80,
                "momentum": 0.30,
            },
        )


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        calculate_portfolio_returns(
            make_strategy_results(),
            {
                "trend": -0.10,
                "momentum": 0.50,
            },
        )


def test_invalid_strategy_return_rejected():
    results = make_strategy_results()
    results["trend"].loc[0, "strategy_return"] = "invalid"

    with pytest.raises(ValueError):
        calculate_portfolio_returns(
            results,
            {
                "trend": 0.60,
                "momentum": 0.40,
            },
        )


def test_missing_timestamp_rejected():
    results = make_strategy_results()
    results["trend"] = results["trend"].drop(
        columns=["timestamp"]
    )

    with pytest.raises(ValueError):
        calculate_portfolio_returns(
            results,
            {
                "trend": 0.60,
                "momentum": 0.40,
            },
        )


def test_misaligned_timestamps_are_supported():
    results = make_strategy_results()

    results["momentum"] = results[
        "momentum"
    ].iloc[1:].reset_index(drop=True)

    result = calculate_portfolio_returns(
        results,
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
    )

    assert len(result) == 4
    assert result["portfolio_return"].iloc[0] == pytest.approx(
        0.06
    )


def test_empty_strategy_results():
    result = calculate_portfolio_returns(
        {},
        {},
    )

    assert result.empty
    assert list(result.columns) == [
        "timestamp",
        "portfolio_return",
    ]


def test_zero_initial_capital():
    returns = calculate_portfolio_returns(
        make_strategy_results(),
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
    )

    result = calculate_portfolio_equity(
        returns,
        initial_capital=0.0,
    )

    assert (result["equity"] == 0.0).all()


def test_negative_initial_capital_rejected():
    returns = calculate_portfolio_returns(
        make_strategy_results(),
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
    )

    with pytest.raises(ValueError):
        calculate_portfolio_equity(
            returns,
            initial_capital=-100.0,
        )


def test_build_portfolio_performance():
    result = build_portfolio_performance(
        make_strategy_results(),
        {
            "trend": 0.60,
            "momentum": 0.40,
        },
        initial_capital=10000.0,
    )

    assert "portfolio_return" in result.columns
    assert "equity" in result.columns
    assert "equity_peak" in result.columns
    assert "drawdown" in result.columns
    assert len(result) == 4
