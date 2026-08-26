# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from finn.dataflow._engine import (
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DerivedProperty,
    DesignPoint,
    DesignSpace,
    DesignSpaceSpec,
    Engine,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow._engine.declarations import DomainResult

INT = as_object_semantics(ValueSemantics.immutable_nominal(int))
BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool))
STRING = as_object_semantics(ValueSemantics.immutable_nominal(str))


def path(value: str) -> QualifiedPath:
    return QualifiedPath(value)


def any_int_domain() -> DecisionDomain:
    def check(value: object, _dependencies: object) -> DomainResult:
        return Decided(type(value) is int)

    return DecisionDomain((), check)


def predicate_domain(predicate: Callable[[int], bool]) -> DecisionDomain:
    def check(value: object, _dependencies: object) -> DomainResult:
        return Decided(type(value) is int and predicate(value))

    return DecisionDomain((), check)


def specification(
    *,
    decisions: tuple[Decision, ...] = (),
    properties: tuple[DerivedProperty, ...] = (),
    constraints: tuple[Constraint, ...] = (),
    constraint_sets: tuple[ConstraintSet, ...] = (),
    readiness_profiles: tuple[ReadinessProfile, ...] = (),
    problem_schema: ProblemSchema | None = None,
) -> DesignSpaceSpec:
    return DesignSpaceSpec(
        problem_schema=problem_schema or ProblemSchema(),
        decisions=decisions,
        properties=properties,
        constraints=constraints,
        constraint_sets=constraint_sets,
        readiness_profiles=readiness_profiles,
    )


def validated(engine: Engine, spec: DesignSpaceSpec) -> DesignSpace:
    return engine.validate(spec)


def started(
    engine: Engine, spec: DesignSpaceSpec, problem: dict[str, object] | None = None
) -> DesignPoint:
    return engine.start(validated(engine, spec), problem or {})
