"""Candidate Generation Layer for Research Discovery Engine.

Provides first-class, traceable, deterministic strategy candidate generation.
Input parameters, constraints, search space, and seeds are explicitly tracked
so every candidate identity and spec can be fully audited and reproduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateSpec:
    """Identity and specification of a generated research strategy candidate.

    Calculates a unique candidate_id based on generator name, strategy_name,
    parameters, and seed.
    """

    generator_name: str
    generator_version: str
    strategy_name: str
    parameters: dict[str, Any]
    random_seed: int | None = None
    hypothesis_template: str = ""
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.generator_name or not self.generator_name.strip():
            raise ValueError("generator_name must be a non-empty string.")
        if not self.generator_version or not self.generator_version.strip():
            raise ValueError("generator_version must be a non-empty string.")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")

        payload = {
            "generator_name": self.generator_name.strip(),
            "generator_version": self.generator_version.strip(),
            "strategy_name": self.strategy_name.strip(),
            "parameters": self.parameters,
            "random_seed": self.random_seed,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        param_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        object.__setattr__(self, "candidate_id", f"cand_{self.strategy_name.strip()}_{param_hash}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "random_seed": self.random_seed,
            "hypothesis_template": self.hypothesis_template,
        }


@dataclass(frozen=True)
class CandidateGeneratorSpec:
    """Configuration for candidate generation run."""

    generator_name: str
    generator_version: str
    strategy_name: str
    parameter_grid: dict[str, Sequence[Any]]
    hypothesis_template: str = "Automated hypothesis for candidate {candidate_id}"
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not self.generator_name or not self.generator_name.strip():
            raise ValueError("generator_name must be a non-empty string.")
        if not self.generator_version or not self.generator_version.strip():
            raise ValueError("generator_version must be a non-empty string.")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not isinstance(self.parameter_grid, dict):
            raise TypeError("parameter_grid must be a dictionary mapping parameter names to lists of values.")
        for param, vals in self.parameter_grid.items():
            if not isinstance(vals, (list, tuple)):
                raise TypeError(f"Grid values for parameter '{param}' must be a list or tuple.")


class CandidateGenerator:
    """Deterministic candidate generator that produces CandidateSpecs from search spaces."""

    def __init__(self, spec: CandidateGeneratorSpec) -> None:
        if not isinstance(spec, CandidateGeneratorSpec):
            raise TypeError("spec must be a CandidateGeneratorSpec instance.")
        self.spec = spec

    def generate_candidates(self) -> tuple[CandidateSpec, ...]:
        """Generate deterministic candidates using Cartesian product of parameter grid."""
        grid = self.spec.parameter_grid
        if not grid:
            # Single candidate with empty parameters if grid is empty
            cand = CandidateSpec(
                generator_name=self.spec.generator_name,
                generator_version=self.spec.generator_version,
                strategy_name=self.spec.strategy_name,
                parameters={},
                random_seed=self.spec.random_seed,
                hypothesis_template=self.spec.hypothesis_template,
            )
            return (cand,)

        keys = sorted(grid.keys())
        value_lists = [list(grid[k]) for k in keys]

        import itertools
        combinations = list(itertools.product(*value_lists))

        candidates: list[CandidateSpec] = []
        for combo in combinations:
            params = {k: v for k, v in zip(keys, combo)}
            cand = CandidateSpec(
                generator_name=self.spec.generator_name,
                generator_version=self.spec.generator_version,
                strategy_name=self.spec.strategy_name,
                parameters=params,
                random_seed=self.spec.random_seed,
                hypothesis_template=self.spec.hypothesis_template,
            )
            candidates.append(cand)

        return tuple(candidates)
