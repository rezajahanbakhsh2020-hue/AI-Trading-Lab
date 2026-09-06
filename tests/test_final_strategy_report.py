import pandas as pd
import pytest

from src.evaluation.final_report import (
build_final_strategy_report,
get_best_strategy,
)

def create_comparison() -> dict[str, dict]:
return {
"strategy_a": {
"windows": 4,
"observations": 40,
"total_return": 0.20,
"max_drawdown": -0.10,
"sharpe_ratio": 1.50,
"calmar_ratio": 2.00,
"sortino_ratio": 1.80,
"exposure": 0.75,
"win_rate": 0.60,
"profit_factor": 1.80,
"profitable_windows": 4,
"losing_windows": 0,
"positive_window_rate": 1.00,
},
"strategy_b": {
"windows": 4,
"observations": 40,
"total_return": 0.10,
"max_drawdown": -0.05,
"sharpe_ratio": 1.20,
"calmar_ratio": 1.80,
"sortino_ratio": 1.40,
"exposure": 0.70,
"win_rate": 0.55,
"profit_factor": 1.50,
"profitable_windows": 3,
"losing_windows": 1,
"positive_window_rate": 0.75,
},
"strategy_c": {
"windows": 4,
"observations": 40,
"total_return": 0.15,
"max_drawdown": -0.08,
"sharpe_ratio": 1.30,
"calmar_ratio": 1.90,
"sortino_ratio": 1.50,
"exposure": 0.72,
"win_rate": 0.58,
"profit_factor": 1.60,
"profitable_windows": 4,
"losing_windows": 0,
"positive_window_rate": 1.00,
},
}

def test_build_final_strategy_report_returns_dataframe() -> None:
report = build_final_strategy_report(
create_comparison()
)

assert isinstance(report, pd.DataFrame)
assert not report.empty

def test_build_final_strategy_report_is_ranked() -> None:
report = build_final_strategy_report(
create_comparison()
)

assert report["strategy"].tolist() == [
    "strategy_a",
    "strategy_c",
    "strategy_b",
]

assert report["rank"].tolist() == [1, 2, 3]

def test_build_final_strategy_report_contains_key_metrics() -> None:
report = build_final_strategy_report(
create_comparison()
)

expected_columns = [
    "rank",
    "strategy",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
    "calmar_ratio",
    "sortino_ratio",
    "win_rate",
    "profit_factor",
    "positive_window_rate",
]

for column in expected_columns:
    assert column in report.columns

def test_build_final_strategy_report_preserves_values() -> None:
report = build_final_strategy_report(
create_comparison()
)

strategy_a = report[
    report["strategy"] == "strategy_a"
].iloc[0]

assert strategy_a["total_return"] == pytest.approx(0.20)
assert strategy_a["max_drawdown"] == pytest.approx(-0.10)
assert strategy_a["sharpe_ratio"] == pytest.approx(1.50)

def test_build_final_strategy_report_supports_custom_metric() -> None:
comparison = create_comparison()

comparison["strategy_b"]["sharpe_ratio"] = 2.00

report = build_final_strategy_report(
    comparison,
    metric="sharpe_ratio",
)

assert report.iloc[0]["strategy"] == "strategy_b"
assert report.iloc[0]["rank"] == 1

def test_build_final_strategy_report_supports_ascending() -> None:
report = build_final_strategy_report(
create_comparison(),
metric="total_return",
ascending=True,
)

assert report["strategy"].tolist() == [
    "strategy_b",
    "strategy_c",
    "strategy_a",
]

def test_build_final_strategy_report_empty_input() -> None:
report = build_final_strategy_report({})

assert isinstance(report, pd.DataFrame)
assert report.empty

def test_get_best_strategy_returns_top_strategy() -> None:
best = get_best_strategy(
create_comparison()
)

assert best == "strategy_a"

def test_get_best_strategy_supports_custom_metric() -> None:
comparison = create_comparison()

comparison["strategy_b"]["sharpe_ratio"] = 2.00

best = get_best_strategy(
    comparison,
    metric="sharpe_ratio",
)

assert best == "strategy_b"

def test_get_best_strategy_empty_input() -> None:
best = get_best_strategy({})

assert best is None
