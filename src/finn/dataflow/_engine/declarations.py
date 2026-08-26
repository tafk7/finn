# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed declarations authored by a domain adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from .primitives import QualifiedPath, ValueSemantics
from .results import Answer

R = TypeVar("R")


class AbsenceMode(str, Enum):
    REQUIRES_APPLICABLE = "requires_applicable"
    ALLOWS_ABSENT = "allows_absent"


class DependencyKind(str, Enum):
    DECISION = "decision"
    PROPERTY = "derived_property"
    PROBLEM = "problem_field"


class DeclarationKind(str, Enum):
    DECISION = "decision"
    DERIVED_PROPERTY = "derived_property"
    CONSTRAINT = "constraint"


class _AbsentValue:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSENT"

    def __bool__(self) -> bool:
        raise TypeError("ABSENT is a semantic dependency value, not a Boolean")


ABSENT = _AbsentValue()
DependencyValue = object


class DependencyView(Mapping[str, DependencyValue]):
    """A restricted evaluator view containing exactly the declared inputs."""

    __slots__ = ("__values",)

    def __init__(self, values: Mapping[str, DependencyValue]) -> None:
        self.__values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> DependencyValue:
        return self.__values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)


@dataclass(frozen=True, slots=True)
class DependencyRef:
    name: str
    path: QualifiedPath
    kind: DependencyKind
    value_semantics: ValueSemantics[object]
    absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE

    @classmethod
    def decision(
        cls,
        name: str,
        path: QualifiedPath | str,
        semantics: ValueSemantics[object],
        *,
        absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE,
    ) -> DependencyRef:
        return cls(name, QualifiedPath.parse(path), DependencyKind.DECISION, semantics, absence)

    @classmethod
    def property(
        cls,
        name: str,
        path: QualifiedPath | str,
        semantics: ValueSemantics[object],
        *,
        absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE,
    ) -> DependencyRef:
        return cls(name, QualifiedPath.parse(path), DependencyKind.PROPERTY, semantics, absence)

    @classmethod
    def problem(
        cls,
        name: str,
        path: QualifiedPath | str,
        semantics: ValueSemantics[object],
        *,
        absence: AbsenceMode = AbsenceMode.REQUIRES_APPLICABLE,
    ) -> DependencyRef:
        return cls(name, QualifiedPath.parse(path), DependencyKind.PROBLEM, semantics, absence)


@dataclass(frozen=True, slots=True)
class EvaluatorSpec(Generic[R]):
    dependencies: tuple[DependencyRef, ...]
    evaluator: Callable[[DependencyView], R]


ApplicabilityResult = Answer[bool]
DomainResult = Answer[bool]
ValueResult = Answer[object]
CandidateResult = Answer[tuple[object, ...]]
ConstraintResult = Answer[bool]


@dataclass(frozen=True, slots=True)
class DecisionDomain:
    """Candidate membership plus optional finite-candidate enumeration."""

    dependencies: tuple[DependencyRef, ...]
    accepts: Callable[[object, DependencyView], DomainResult]
    candidates: EvaluatorSpec[CandidateResult] | None = None


@dataclass(frozen=True, slots=True)
class Decision:
    path: QualifiedPath
    value_semantics: ValueSemantics[object]
    domain: DecisionDomain
    applies_if: EvaluatorSpec[ApplicabilityResult] | None = None
    proposal: EvaluatorSpec[ValueResult] | None = None


@dataclass(frozen=True, slots=True)
class DerivedProperty:
    path: QualifiedPath
    value_semantics: ValueSemantics[object]
    evaluator: EvaluatorSpec[ValueResult]
    applies_if: EvaluatorSpec[ApplicabilityResult] | None = None


@dataclass(frozen=True, slots=True)
class Constraint:
    path: QualifiedPath
    evaluator: EvaluatorSpec[ConstraintResult]
    applies_if: EvaluatorSpec[ApplicabilityResult] | None = None


@dataclass(frozen=True, slots=True)
class ProblemField:
    path: QualifiedPath
    value_semantics: ValueSemantics[object]
    required: bool = True
    constraint: Callable[[object], bool] | None = None
    constraint_description: str = ""


@dataclass(frozen=True, slots=True)
class ProblemSchema:
    fields: tuple[ProblemField, ...] = ()


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    name: str
    constraints: tuple[QualifiedPath, ...]


@dataclass(frozen=True, slots=True)
class ReadinessProfile:
    name: str
    decisions: tuple[QualifiedPath, ...] = ()
    properties: tuple[QualifiedPath, ...] = ()
    constraints: tuple[QualifiedPath, ...] = ()


@dataclass(frozen=True, slots=True)
class DesignSpaceSpec:
    problem_schema: ProblemSchema = ProblemSchema()
    decisions: tuple[Decision, ...] = ()
    properties: tuple[DerivedProperty, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    constraint_sets: tuple[ConstraintSet, ...] = ()
    readiness_profiles: tuple[ReadinessProfile, ...] = ()


def as_object_semantics(semantics: ValueSemantics[R]) -> ValueSemantics[object]:
    return cast(ValueSemantics[object], semantics)


__all__ = [
    "ABSENT",
    "AbsenceMode",
    "ApplicabilityResult",
    "CandidateResult",
    "Constraint",
    "ConstraintResult",
    "ConstraintSet",
    "Decision",
    "DecisionDomain",
    "DeclarationKind",
    "DependencyKind",
    "DependencyRef",
    "DependencyValue",
    "DependencyView",
    "DerivedProperty",
    "DesignSpaceSpec",
    "DomainResult",
    "EvaluatorSpec",
    "ProblemField",
    "ProblemSchema",
    "ReadinessProfile",
    "ValueResult",
    "as_object_semantics",
]
