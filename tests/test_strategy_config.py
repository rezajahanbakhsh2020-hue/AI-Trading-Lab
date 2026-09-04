from configs.strategies import (
    MOVING_AVERAGE_CONFIG,
    MOMENTUM_CONFIG,
    BACKTEST_CONFIG,
)


def test_moving_average_config():
    assert MOVING_AVERAGE_CONFIG["fast_window"] == 20
    assert MOVING_AVERAGE_CONFIG["slow_window"] == 50
    assert (
        MOVING_AVERAGE_CONFIG["fast_window"]
        < MOVING_AVERAGE_CONFIG["slow_window"]
    )


def test_momentum_config():
    assert MOMENTUM_CONFIG["window"] == 10
    assert MOMENTUM_CONFIG["window"] > 0


def test_backtest_config():
    assert BACKTEST_CONFIG["transaction_cost"] == 0.0
    assert BACKTEST_CONFIG["slippage"] == 0.0


def test_strategy_configs_are_dictionaries():
    assert isinstance(MOVING_AVERAGE_CONFIG, dict)
    assert isinstance(MOMENTUM_CONFIG, dict)
    assert isinstance(BACKTEST_CONFIG, dict)
