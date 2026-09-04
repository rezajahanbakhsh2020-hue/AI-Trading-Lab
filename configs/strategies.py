"""
Strategy configuration for AI-Trading-Lab.

This module contains default parameters for the
currently supported strategies.
"""


MOVING_AVERAGE_CONFIG = {
    "fast_window": 20,
    "slow_window": 50,
}


MOMENTUM_CONFIG = {
    "window": 10,
}


BACKTEST_CONFIG = {
    "transaction_cost": 0.0,
    "slippage": 0.0,
}
