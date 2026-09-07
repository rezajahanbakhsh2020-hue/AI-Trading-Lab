import pandas as pd
import pytest

from src.evaluation.walk_forward_report import (
combine_oos_results,
evaluate_walk_forward,
)

def create_oos_result(
timestamps,
strategy_returns,
positions=None,
) -> pd.DataFrame:
strategy_returns = pd.Series(
strategy_returns,
dtype=float,
)

equity = (
    1.0 + strategy_returns.fillna(0.0)
).cumprod()

data = {
    "timestamp": pd.to_datetime(timestamps),
    "strategy_return": strategy_returns,
    "equity": equity,
}

if positions is not None:
    data["position"] = positions

return pd.DataFrame(data)

def test_combine_oos_results_empty():
result = combine_oos_results([])

assert isinstance(result, pd.DataFrame)
assert result.empty

def test_combine_oos_results_rejects_invalid_input():
with pytest.raises(TypeError):
combine_oos_results("invalid")

def test_combine_oos_results_rejects_non_dataframe():
with pytest.raises(TypeError):
combine_oos_results([pd.DataFrame(), "invalid"])

def test_combine_oos_results_rejects_missing_required_columns():
invalid = pd.DataFrame(
{
"timestamp": pd.date_range(
"2026-01-01",
periods=2,
freq="D",
),
"strategy_return": [0.01, 0.02],
}
)

with pytest.raises(ValueError):
    combine_oos_results([invalid])

def test_combine_oos_results_sorts_chronologically():
first = create_oos_result(
[
"2026-01-03",
"2026-01-04",
],
[0.01, 0.02],
[1, 1],
)

second = create_oos_result(
    [
        "2026-01-01",
        "2026-01-02",
    ],
    [0.03, 0.04],
    [1, 1],
)

combined = combine_oos_results(
    [first, second]
)

assert list(
    combined["timestamp"]
) == list(
    pd.to_datetime(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ]
    )
)

def test_evaluate_walk_forward_empty():
report = evaluate_walk_forward([])

assert report["windows"] == 0
assert report["observations"] == 0
assert report["total_return"] == 0.0
assert report["max_drawdown"] == 0.0
assert report["sharpe_ratio"] == 0.0
assert report["calmar_ratio"] == 0.0
assert report["sortino_ratio"] == 0.0
assert report["exposure"] == 0.0
assert report["win_rate"] == 0.0
assert report["profit_factor"] == 0.0
assert report["window_returns"] == []
assert report["profitable_windows"] == 0
assert report["losing_windows"] == 0
assert report["positive_window_rate"] == 0.0

def test_evaluate_walk_forward_rejects_invalid_input():
with pytest.raises(TypeError):
evaluate_walk_forward("invalid")

def test_evaluate_walk_forward_calculates_window_statistics():
first = create_oos_result(
pd.date_range(
"2026-01-01",
periods=3,
freq="D",
),
[0.10, 0.00, 0.10],
[1, 1, 1],
)

second = create_oos_result(
    pd.date_range(
        "2026-01-04",
        periods=3,
        freq="D",
    ),
    [-0.05, -0.05, 0.00],
    [1, 1, 0],
)

report = evaluate_walk_forward(
    [first, second]
)

assert report["windows"] == 2
assert report["observations"] == 6
assert len(report["window_returns"]) == 2
assert report["profitable_windows"] == 1
assert report["losing_windows"] == 1
assert report["positive_window_rate"] == pytest.approx(
    0.5
)

def test_evaluate_walk_forward_uses_position_for_exposure():
result = create_oos_result(
pd.date_range(
"2026-01-01",
periods=4,
freq="D",
),
[0.01, 0.01, 0.01, 0.01],
[1, 1, 0, 0],
)

report = evaluate_walk_forward([result])

assert report["exposure"] == pytest.approx(
    0.5
)

def test_evaluate_walk_forward_uses_signal_when_position_missing():
result = create_oos_result(
pd.date_range(
"2026-01-01",
periods=4,
freq="D",
),
[0.01, 0.01, 0.01, 0.01],
)

result["signal"] = [1, 0, 1, 0]

report = evaluate_walk_forward([result])

assert report["exposure"] == pytest.approx(
    0.5
)

def test_evaluate_walk_forward_rejects_missing_exposure():
result = create_oos_result(
pd.date_range(
"2026-01-01",
periods=3,
freq="D",
),
[0.01, 0.01, 0.01],
)

with pytest.raises(ValueError):
    evaluate_walk_forward([result])

def test_evaluate_walk_forward_preserves_window_count_with_empty_window():
non_empty = create_oos_result(
pd.date_range(
"2026-01-01",
periods=3,
freq="D",
),
[0.01, 0.01, 0.01],
[1, 1, 1],
)

empty = pd.DataFrame(
    columns=[
        "timestamp",
        "strategy_return",
        "equity",
        "position",
    ]
)

report = evaluate_walk_forward(
    [non_empty, empty]
)

assert report["windows"] == 2
assert report["observations"] == 3
assert len(report["window_returns"]) == 2
assert report["window_returns"][1] == 0.0
assert report["profitable_windows"] == 1
assert report["positive_window_rate"] == pytest.approx(
    0.5
)

:::writing

این تغییر فقط تست را واقعی می‌کند و به "src/" دست نمی‌زند. بعد از جایگزینی، commit کن تا Actions دوباره اجرا شود.
