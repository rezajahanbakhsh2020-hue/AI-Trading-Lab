import pandas as pd

from src.pipeline import (
    load_and_prepare_market_data,
    run_strategy_backtest,
    run_walk_forward_backtest,
)


def create_market_data(size: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=size,
        freq="D",
    )

    close = [
        100.0,
        101.0,
        102.0,
        101.0,
        103.0,
        104.0,
        105.0,
        104.0,
        106.0,
        107.0,
        108.0,
        107.0,
        109.0,
        110.0,
        111.0,
        110.0,
        112.0,
        113.0,
        114.0,
        113.0,
        115.0,
        116.0,
        117.0,
        116.0,
        118.0,
        119.0,
        120.0,
        119.0,
        121.0,
        122.0,
    ][:size]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [value - 0.5 for value in close],
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000 + i * 10 for i in range(size)],
        }
    )


def test_load_and_prepare_market_data(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(size=2)
    data.to_csv(csv_file, index=False)

    result = load_and_prepare_market_data(
        str(csv_file)
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_run_strategy_backtest(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(size=10)
    data.to_csv(csv_file, index=False)

    result, report = run_strategy_backtest(
        str(csv_file)
    )

    assert isinstance(result, pd.DataFrame)
    assert isinstance(report, dict)

    assert "return" in result.columns
    assert "signal" in result.columns
    assert "strategy_return" in result.columns
    assert "equity" in result.columns

    assert "total_return" in report
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report
    assert "calmar_ratio" in report
    assert "sortino_ratio" in report
    assert "exposure" in report
    assert "win_rate" in report
    assert "profit_factor" in report

    assert len(result) == len(data)


def test_run_walk_forward_backtest(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(size=30)
    data.to_csv(csv_file, index=False)

    oos_results, report = run_walk_forward_backtest(
        str(csv_file),
        train_size=10,
        test_size=5,
    )

    assert isinstance(oos_results, list)
    assert len(oos_results) > 0

    assert isinstance(report, dict)

    assert report["windows"] == len(oos_results)
    assert report["observations"] > 0

    assert "total_return" in report
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report
    assert "calmar_ratio" in report
    assert "sortino_ratio" in report
    assert "exposure" in report
    assert "win_rate" in report
    assert "profit_factor" in report

    for result in oos_results:
        assert isinstance(result, pd.DataFrame)

        assert "timestamp" in result.columns
        assert "signal" in result.columns
        assert "strategy_return" in result.columns
        assert "equity" in result.columns

        assert "walk_forward_train_start" in result.columns
        assert "walk_forward_train_end" in result.columns
        assert "walk_forward_test_start" in result.columns
        assert "walk_forward_test_end" in result.columns

        assert "oos_equity" in result.columns
