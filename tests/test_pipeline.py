import pandas as pd

from src.pipeline import (
    load_and_prepare_market_data,
    run_default_walk_forward_pipeline,
    run_strategy_backtest,
)


def create_market_data(rows: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="D",
    )

    close = [
        100.0 + (index % 11) * 0.8 + (index * 0.05)
        for index in range(rows)
    ]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [value - 0.5 for value in close],
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000 + index * 10 for index in range(rows)],
        }
    )


def test_load_and_prepare_market_data(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(10)
    data.to_csv(csv_file, index=False)

    result = load_and_prepare_market_data(str(csv_file))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)


def test_run_strategy_backtest(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(20)
    data.to_csv(csv_file, index=False)

    result, report = run_strategy_backtest(str(csv_file))

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


def test_run_default_walk_forward_pipeline(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(120)
    data.to_csv(csv_file, index=False)

    comparison, final_report = run_default_walk_forward_pipeline(
        path=str(csv_file),
        train_size=60,
        test_size=20,
    )

    assert isinstance(comparison, dict)
    assert set(comparison) == {
        "moving_average",
        "momentum",
    }

    assert isinstance(final_report, pd.DataFrame)
    assert not final_report.empty

    assert "rank" in final_report.columns
    assert "strategy" in final_report.columns
    assert "total_return" in final_report.columns
    assert "max_drawdown" in final_report.columns
    assert "positive_window_rate" in final_report.columns

    assert final_report["rank"].tolist() == [1, 2]
    assert set(final_report["strategy"]) == {
        "moving_average",
        "momentum",
    }


def test_run_default_walk_forward_pipeline_supports_custom_metric(
    tmp_path,
):
    csv_file = tmp_path / "market_data.csv"

    data = create_market_data(120)
    data.to_csv(csv_file, index=False)

    comparison, final_report = run_default_walk_forward_pipeline(
        path=str(csv_file),
        train_size=60,
        test_size=20,
        metric="sharpe_ratio",
    )

    assert isinstance(comparison, dict)
    assert isinstance(final_report, pd.DataFrame)
    assert not final_report.empty
    assert "sharpe_ratio" in final_report.columns
