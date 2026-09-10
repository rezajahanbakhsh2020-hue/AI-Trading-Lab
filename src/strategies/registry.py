from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import pandas as pd

from .baseline import baseline_signal
from .breakout import breakout_signal
from .momentum import momentum_signal


SignalFunction = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class StrategySpec:
    """
    Lightweight description of a registered strategy.

    The registry deliberately stores metadata and a callable only.
    Backtesting, ranking, walk-forward validation, and stability
    evaluation remain outside this module.
    """

    name: str
    category: str
    function: SignalFunction
    description: str = ""
    supports_long: bool = True
    supports_short: bool = False

    def __post_init__(self) -> None:
        normalized_name = self.name.strip().lower()

        if not normalized_name:
            raise ValueError("strategy name must not be empty.")

        if normalized_name != self.name:
            raise ValueError(
                "strategy name must be lowercase and normalized."
            )

        if not self.category.strip():
            raise ValueError("strategy category must not be empty.")

        if not callable(self.function):
            raise TypeError("strategy function must be callable.")


class StrategyRegistry:
    """
    Compact in-memory strategy bank.

    A registry entry contains:
        - stable name
        - category
        - signal function
        - descriptive metadata

    This keeps strategy discovery separate from the existing
    backtest engine and allows future indicators/strategies to
    be added without changing the backtest implementation.
    """

    def __init__(
        self,
        strategies: Iterable[StrategySpec] | None = None,
    ) -> None:
        self._strategies: Dict[str, StrategySpec] = {}

        if strategies is not None:
            for strategy in strategies:
                self.register(strategy)

    def register(
        self,
        strategy: StrategySpec,
        *,
        replace: bool = False,
    ) -> StrategySpec:
        if not isinstance(strategy, StrategySpec):
            raise TypeError("strategy must be a StrategySpec.")

        if strategy.name in self._strategies and not replace:
            raise ValueError(
                f"strategy '{strategy.name}' is already registered."
            )

        self._strategies[strategy.name] = strategy
        return strategy

    def unregister(self, name: str) -> StrategySpec:
        normalized_name = self._normalize_name(name)

        try:
            return self._strategies.pop(normalized_name)
        except KeyError as exc:
            raise KeyError(
                f"strategy '{normalized_name}' is not registered."
            ) from exc

    def get(self, name: str) -> StrategySpec:
        normalized_name = self._normalize_name(name)

        try:
            return self._strategies[normalized_name]
        except KeyError as exc:
            raise KeyError(
                f"strategy '{normalized_name}' is not registered."
            ) from exc

    def contains(self, name: str) -> bool:
        return self._normalize_name(name) in self._strategies

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def all(self) -> tuple[StrategySpec, ...]:
        return tuple(
            self._strategies[name]
            for name in sorted(self._strategies)
        )

    def by_category(self, category: str) -> tuple[StrategySpec, ...]:
        normalized_category = category.strip().lower()

        if not normalized_category:
            raise ValueError("category must not be empty.")

        return tuple(
            strategy
            for strategy in self.all()
            if strategy.category.lower() == normalized_category
        )

    def run(
        self,
        name: str,
        df: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        strategy = self.get(name)
        return strategy.function(df, **kwargs)

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("strategy name must be a string.")

        normalized_name = name.strip().lower()

        if not normalized_name:
            raise ValueError("strategy name must not be empty.")

        return normalized_name


def build_default_registry() -> StrategyRegistry:
    """
    Build the lightweight built-in strategy bank.

    Existing strategy implementations are reused unchanged.
    """

    return StrategyRegistry(
        (
            StrategySpec(
                name="baseline",
                category="trend",
                function=baseline_signal,
                description="Fast/slow moving-average trend baseline.",
                supports_long=True,
                supports_short=False,
            ),
            StrategySpec(
                name="breakout",
                category="breakout",
                function=breakout_signal,
                description="Previous-range high breakout strategy.",
                supports_long=True,
                supports_short=False,
            ),
            StrategySpec(
                name="momentum",
                category="momentum",
                function=momentum_signal,
                description="Positive percentage-change momentum strategy.",
                supports_long=True,
                supports_short=False,
            ),
        )
    )


DEFAULT_REGISTRY = build_default_registry()


def get_strategy(name: str) -> StrategySpec:
    return DEFAULT_REGISTRY.get(name)


def list_strategies() -> tuple[str, ...]:
    return DEFAULT_REGISTRY.names()


def run_strategy(
    name: str,
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    return DEFAULT_REGISTRY.run(name, df, **kwargs)
