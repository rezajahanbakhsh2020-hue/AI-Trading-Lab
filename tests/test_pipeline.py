import pandas as pd

from src.pipeline import (
    load_and_prepare_market_data,
    run_strategy_backtest,
)


def test_load_and_prepare_market_data(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [1000, 1100],
        }
    )

    data.to_csv(csv_file, index=False)

    result = load_and_prepare_market_data(str(csv_file))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_run_strategy_backtest(tmp_path):
    csv_file = tmp_path / "market_data.csv"

    data = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
                "2026-01-06",
            ],
            "open": [100, 101, 100, 102, 101, 103],
            "high": [102, 103, 102, 104, 103, 105],
            "low": [99, 100, 99, 101, 100, 102],
            "close": [101, 100, 102, 101, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500],
        }
    )

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
