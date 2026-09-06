1s
Run if [ -d tests ] && find tests -type f -name "*.py" | grep -q .; then
============================= test session starts ==============================
platform linux -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/runner/work/AI-Trading-Lab/AI-Trading-Lab
collected 227 items

tests/test_backtest.py ..............                                    [  6%]
tests/test_backtest_quality.py ........                                  [  9%]
tests/test_data.py ...                                                   [ 11%]
tests/test_data_config.py .....                                          [ 13%]
tests/test_data_provider.py .....                                        [ 15%]
tests/test_data_providers.py ....                                        [ 17%]
tests/test_evaluation_report.py ........                                 [ 20%]
tests/test_features.py .                                                 [ 21%]
tests/test_loader.py .....                                               [ 23%]
tests/test_market_data.py ....                                           [ 25%]
tests/test_metrics.py ............................                       [ 37%]
tests/test_pipeline.py ..                                                [ 38%]
tests/test_preprocessing.py ....                                         [ 40%]
tests/test_report.py ........                                            [ 43%]
tests/test_strategies_baseline.py ........                               [ 47%]
tests/test_strategies_momentum.py ........                               [ 50%]
tests/test_strategy.py ........                                          [ 54%]
tests/test_strategy_comparison.py .....................                  [ 63%]
tests/test_strategy_comparison_real.py ...                               [ 64%]
tests/test_strategy_config.py ....                                       [ 66%]
tests/test_strategy_suite.py ....                                        [ 68%]
tests/test_transaction_costs.py ..............                           [ 74%]
tests/test_validation.py ..............                                  [ 80%]
tests/test_walk_forward.py .......                                       [ 83%]
tests/test_walk_forward_report.py .....F..                               [ 87%]
tests/test_walk_forward_runner.py .......                                [ 90%]
tests/test_xauusd_csv_workflow.py .                                      [ 90%]
tests/test_xauusd_pipeline.py .                                          [ 91%]
tests/test_xauusd_provider.py ......                                     [ 93%]
tests/test_xauusd_validation.py ..........                               [ 98%]
tests/test_xauusd_workflow.py ....                                       [100%]

=================================== FAILURES ===================================
__________________ test_evaluate_walk_forward_returns_metrics __________________

    def test_evaluate_walk_forward_returns_metrics():
        first = create_oos_result(
            "2026-01-01",
            size=3,
        )
    
        second = create_oos_result(
            "2026-01-04",
            size=3,
        )
    
>       report = evaluate_walk_forward(
            [first, second]
        )

tests/test_walk_forward_report.py:114: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
src/evaluation/walk_forward_report.py:97: in evaluate_walk_forward
    "exposure": exposure(combined),
                ^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

df =    timestamp  strategy_return    equity  position
0 2026-01-01             0.01  1.010000       1.0
1 2026-01-02      ... 1.010000       1.0
4 2026-01-05             0.02  1.030200       1.0
5 2026-01-06            -0.01  1.019898       1.0
signal_column = 'signal'

    def exposure(
        df: pd.DataFrame,
        signal_column: str = "signal",
    ) -> float:
        """
        Calculate the percentage of periods spent in the market.
    
        Returns a value between 0.0 and 1.0.
        """
    
        if signal_column not in df.columns:
>           raise ValueError(f"Column '{signal_column}' not found.")
E           ValueError: Column 'signal' not found.

src/evaluation/metrics.py:144: ValueError
=========================== short test summary info ============================
FAILED tests/test_walk_forward_report.py::test_evaluate_walk_forward_returns_metrics - ValueError: Column 'signal' not found.
======================== 1 failed, 226 passed in 1.23s =========================
Error: Process completed with exit code 1.
0s
