import pandas as pd
import pytest

from src.strategies.registry import (
    DEFAULT_REGISTRY,
    StrategyRegistry,
    StrategySpec,
    build_default_registry,
    get_strategy,
    list_strategies,
    run_strategy,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": range(60),
            "open": [100.0 + i for i in range(60)],
            "high": [101.0 + i for i in range(60)],
            "low": [99.0 + i for i in range(60)],
            "close": [100.0 + i for i in range(60)],
        }
    )


def test_default_registry_contains_existing_strategies():
    assert list_strategies() == (
        "baseline",
        "breakout",
        "momentum",
    )


def test_default_registry_returns_strategy_metadata():
    strategy = get_strategy("momentum")

    assert isinstance(strategy, StrategySpec)
    assert strategy.name == "momentum"
    assert strategy.category == "momentum"
    assert callable(strategy.function)
    assert strategy.supports_long is True


def test_registry_accepts_custom_strategy():
    def custom_signal(df):
        result = df.copy()
        result["signal"] = 1
        return result

    registry = StrategyRegistry()

    strategy = StrategySpec(
        name="custom",
        category="custom",
        function=custom_signal,
        description="Test strategy.",
    )

    registry.register(strategy)

    assert registry.contains("custom")
    assert registry.get("custom") is strategy
    assert registry.names() == ("custom",)


def test_registry_rejects_duplicate_without_replace():
    registry = build_default_registry()

    with pytest.raises(ValueError):
        registry.register(
            StrategySpec(
                name="momentum",
                category="custom",
                function=lambda df: df.copy(),
            )
        )


def test_registry_can_replace_existing_strategy():
    registry = build_default_registry()

    replacement = StrategySpec(
        name="momentum",
        category="experimental",
        function=lambda df: df.copy(),
    )

    registry.register(replacement, replace=True)

    assert registry.get("momentum") is replacement
    assert registry.get("momentum").category == "experimental"


def test_registry_normalizes_lookup_names():
    registry = build_default_registry()

    assert registry.get(" MOMENTUM ").name == "momentum"
    assert registry.contains(" BREAKOUT ")


def test_registry_runs_existing_strategy_without_changing_it():
    df = sample_frame()

    result = DEFAULT_REGISTRY.run(
        "momentum",
        df,
        window=10,
    )

    assert "momentum" in result.columns
    assert "signal" in result.columns
    assert len(result) == len(df)


def test_registry_global_helper_runs_strategy():
    df = sample_frame()

    result = run_strategy(
        "baseline",
        df,
        fast_window=5,
        slow_window=20,
    )

    assert "fast_ma" in result.columns
    assert "slow_ma" in result.columns
    assert "signal" in result.columns


def test_registry_filters_by_category():
    registry = build_default_registry()

    strategies = registry.by_category("MOMENTUM")

    assert [strategy.name for strategy in strategies] == ["momentum"]


def test_registry_rejects_invalid_strategy_spec():
    with pytest.raises(ValueError):
        StrategySpec(
            name="",
            category="test",
            function=lambda df: df,
        )

    with pytest.raises(ValueError):
        StrategySpec(
            name="Invalid Name",
            category="test",
            function=lambda df: df,
        )

    with pytest.raises(TypeError):
        StrategySpec(
            name="invalid",
            category="test",
            function="not-callable",
        )


def test_registry_rejects_invalid_lookup():
    registry = build_default_registry()

    with pytest.raises(KeyError):
        registry.get("unknown")

    with pytest.raises(TypeError):
        registry.get(123)

    with pytest.raises(ValueError):
        registry.get("")


def test_unregister_returns_removed_strategy():
    registry = build_default_registry()

    removed = registry.unregister("momentum")

    assert removed.name == "momentum"
    assert not registry.contains("momentum")


def test_default_registry_is_shared_and_ready_for_future_extensions():
    assert DEFAULT_REGISTRY.contains("baseline")
    assert DEFAULT_REGISTRY.contains("breakout")
    assert DEFAULT_REGISTRY.contains("momentum")
