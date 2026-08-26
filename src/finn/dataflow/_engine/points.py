# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable design points and operation products that carry them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import EvaluationError
from .primitives import QualifiedPath
from .results import AssignmentOrigin, ItemOutcome
from .validation import DesignSpace

_MAPPING_PROXY: type[object] = type(MappingProxyType({}))


@dataclass(frozen=True, eq=False)
class DesignPoint:
    """An immutable partial assignment; caches remain external and non-semantic."""

    design_space: DesignSpace
    problem: Mapping[QualifiedPath, object]
    assignments: Mapping[QualifiedPath, object]
    origins: Mapping[QualifiedPath, AssignmentOrigin]

    def __post_init__(self) -> None:
        if not isinstance(self.problem, _MAPPING_PROXY):
            object.__setattr__(self, "problem", MappingProxyType(dict(self.problem)))
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))
        object.__setattr__(self, "origins", MappingProxyType(dict(self.origins)))

    def equivalent_to(self, other: DesignPoint) -> bool:
        """Compare semantic assignments within the same space and problem instance."""

        if self.design_space is not other.design_space or self.problem is not other.problem:
            return False
        if self.assignments.keys() != other.assignments.keys():
            return False
        for path, value in self.assignments.items():
            try:
                equal = self.design_space.decisions[path].value_semantics.values_equal(
                    value, other.assignments[path]
                )
            except Exception as exc:
                raise EvaluationError(path, "value equality", "comparison raised") from exc
            if not equal:
                return False
        return True


@dataclass(frozen=True, slots=True)
class CommitResult:
    point: DesignPoint
    outcomes: tuple[ItemOutcome, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcomes", tuple(sorted(self.outcomes, key=lambda item: item.path))
        )


@dataclass(frozen=True, slots=True)
class ProposalAdoptionResult:
    point: DesignPoint
    passes: tuple[tuple[ItemOutcome, ...], ...]

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("ProposalAdoptionResult requires at least one pass")
        object.__setattr__(
            self,
            "passes",
            tuple(tuple(sorted(items, key=lambda item: item.path)) for items in self.passes),
        )


def make_initial_point(
    design_space: DesignSpace, problem: Mapping[QualifiedPath, object]
) -> DesignPoint:
    return DesignPoint(design_space, MappingProxyType(dict(problem)), {}, {})


def make_successor(
    point: DesignPoint,
    assignments: Mapping[QualifiedPath, object],
    origins: Mapping[QualifiedPath, AssignmentOrigin],
) -> DesignPoint:
    return DesignPoint(point.design_space, point.problem, assignments, origins)


__all__ = [
    "CommitResult",
    "DesignPoint",
    "ProposalAdoptionResult",
    "make_initial_point",
    "make_successor",
]
