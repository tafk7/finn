# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic algebra over ordinary flat design-space specifications.

This module adds no engine primitive.  It only rebases, gates, and combines
ordinary ``DesignSpaceSpec`` values so that FINN authoring surfaces can compose
reusable declaration fragments into one flat scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar, cast

from finn.dataflow.design import (
    Absent,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
)

T = TypeVar("T")


@dataclass(frozen=True)
class SpecAuthoringIssue:
    """One deterministic error in authoring metadata or spec assembly."""

    code: str
    path: str
    message: str


class SpecAuthoringError(ValueError):
    """Raised when authoring metadata cannot form an unambiguous flat scope."""

    def __init__(self, issues: tuple[SpecAuthoringIssue, ...]) -> None:
        self.issues = tuple(sorted(issues, key=lambda issue: (issue.path, issue.code)))
        super().__init__(f"design-space authoring failed with {len(self.issues)} issue(s)")


def duplicate_values(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return the sorted set of values occurring more than once."""

    return tuple(sorted(value for value in set(values) if values.count(value) > 1))


def spec_declaration_paths(spec: DesignSpaceSpec) -> tuple[QualifiedPath, ...]:
    """Return every declaration path owned by one specification."""

    return tuple(
        [field.path for field in spec.problem_schema.fields]
        + [decision.path for decision in spec.decisions]
        + [prop.path for prop in spec.properties]
        + [constraint.path for constraint in spec.constraints]
    )


def prefixed(prefix: QualifiedPath, path: QualifiedPath) -> QualifiedPath:
    """Return one declaration path rebased beneath an owning prefix."""

    return QualifiedPath(f"{prefix}.{path}")


def _map_diagnostic_value(
    value: object, path_mapping: Mapping[QualifiedPath, QualifiedPath]
) -> object:
    if isinstance(value, QualifiedPath):
        return path_mapping.get(value, value)
    if isinstance(value, tuple):
        return tuple(_map_diagnostic_value(item, path_mapping) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    str(name),
                    _map_diagnostic_value(item, path_mapping),
                )
                for name, item in value.items()
            )
        )
    return value


def map_answer_findings(
    answer: Answer[T], path_mapping: Mapping[QualifiedPath, QualifiedPath]
) -> Answer[T]:
    """Rebase every diagnostic path carried by one answer."""

    if isinstance(answer, Decided):
        return answer
    findings = tuple(
        Finding(
            finding.kind,
            finding.code,
            path_mapping.get(finding.path, finding.path),
            finding.message,
            tuple(
                (name, _map_diagnostic_value(value, path_mapping)) for name, value in finding.values
            ),
            tuple(path_mapping.get(path, path) for path in finding.trace),
        )
        for finding in answer.findings
    )
    if isinstance(answer, Absent):
        return Absent(findings)
    return Unresolved(findings)


def rebase_dependency(
    dependency: DependencyRef, path_mapping: Mapping[QualifiedPath, QualifiedPath]
) -> DependencyRef:
    """Return one dependency reference rebased through a path mapping."""

    return DependencyRef(
        dependency.name,
        path_mapping.get(dependency.path, dependency.path),
        dependency.kind,
        dependency.value_semantics,
        dependency.absence,
    )


def rebase_evaluator(
    evaluator: EvaluatorSpec[Answer[T]],
    path_mapping: Mapping[QualifiedPath, QualifiedPath],
) -> EvaluatorSpec[Answer[T]]:
    """Return one evaluator whose dependencies and findings are rebased."""

    def evaluate(values: DependencyView) -> Answer[T]:
        return map_answer_findings(evaluator.evaluator(values), path_mapping)

    return EvaluatorSpec(
        tuple(rebase_dependency(dependency, path_mapping) for dependency in evaluator.dependencies),
        evaluate,
    )


def rebase_spec(
    spec: DesignSpaceSpec,
    path_mapping: Mapping[QualifiedPath, QualifiedPath],
    *,
    name_prefix: str,
    dropped_problem_paths: frozenset[QualifiedPath] = frozenset(),
) -> DesignSpaceSpec:
    """Return one specification rebased through an explicit path mapping.

    ``dropped_problem_paths`` names definition-local problem fields that the
    caller has mapped onto an outer-scope field it already declares.
    """

    def optional_evaluator(
        value: EvaluatorSpec[Answer[bool]] | None,
    ) -> EvaluatorSpec[Answer[bool]] | None:
        if value is None:
            return None
        return cast(EvaluatorSpec[Answer[bool]], rebase_evaluator(value, path_mapping))

    fields = tuple(
        ProblemField(
            path_mapping[field.path],
            field.value_semantics,
            field.required,
            field.constraint,
            field.constraint_description,
        )
        for field in spec.problem_schema.fields
        if field.path not in dropped_problem_paths
    )
    decisions = []
    for decision in spec.decisions:
        domain_dependencies = tuple(
            rebase_dependency(dependency, path_mapping)
            for dependency in decision.domain.dependencies
        )

        def accepts(
            candidate: object,
            values: DependencyView,
            domain: DecisionDomain = decision.domain,
        ) -> Answer[bool]:
            return cast(
                Answer[bool],
                map_answer_findings(domain.accepts(candidate, values), path_mapping),
            )

        candidates = (
            None
            if decision.domain.candidates is None
            else rebase_evaluator(decision.domain.candidates, path_mapping)
        )
        decisions.append(
            Decision(
                path_mapping[decision.path],
                decision.value_semantics,
                DecisionDomain(domain_dependencies, accepts, candidates),
                optional_evaluator(decision.applies_if),
                None
                if decision.proposal is None
                else rebase_evaluator(decision.proposal, path_mapping),
            )
        )
    properties = tuple(
        DerivedProperty(
            path_mapping[item.path],
            item.value_semantics,
            rebase_evaluator(item.evaluator, path_mapping),
            optional_evaluator(item.applies_if),
        )
        for item in spec.properties
    )
    constraints = tuple(
        Constraint(
            path_mapping[item.path],
            cast(EvaluatorSpec[Answer[bool]], rebase_evaluator(item.evaluator, path_mapping)),
            optional_evaluator(item.applies_if),
        )
        for item in spec.constraints
    )
    constraint_sets = tuple(
        ConstraintSet(
            f"{name_prefix}.{item.name}",
            tuple(path_mapping[path] for path in item.constraints),
        )
        for item in spec.constraint_sets
    )
    readiness_profiles = tuple(
        ReadinessProfile(
            f"{name_prefix}.{item.name}",
            tuple(path_mapping[path] for path in item.decisions),
            tuple(path_mapping[path] for path in item.properties),
            tuple(path_mapping[path] for path in item.constraints),
        )
        for item in spec.readiness_profiles
    )
    return DesignSpaceSpec(
        ProblemSchema(fields),
        tuple(decisions),
        properties,
        constraints,
        constraint_sets,
        readiness_profiles,
    )


def combine_applicability(
    outer: EvaluatorSpec[Answer[bool]],
    inner: EvaluatorSpec[Answer[bool]] | None,
) -> EvaluatorSpec[Answer[bool]]:
    """Compose an outer gate in front of one declaration's own applicability."""

    dependencies = outer.dependencies + (() if inner is None else inner.dependencies)
    names = tuple(dependency.name for dependency in dependencies)
    if len(names) != len(set(names)):
        raise SpecAuthoringError(
            (
                SpecAuthoringIssue(
                    "applicability-dependency-name-duplicate",
                    "applicability",
                    "combined applicability dependencies must have unique names",
                ),
            )
        )

    def evaluate(values: DependencyView) -> Answer[bool]:
        outer_values = DependencyView(
            {dependency.name: values[dependency.name] for dependency in outer.dependencies}
        )
        outer_answer = outer.evaluator(outer_values)
        if not isinstance(outer_answer, Decided) or not outer_answer.value:
            return outer_answer
        if inner is None:
            return Decided(True)
        inner_values = DependencyView(
            {dependency.name: values[dependency.name] for dependency in inner.dependencies}
        )
        return inner.evaluator(inner_values)

    return EvaluatorSpec(dependencies, evaluate)


def gate_spec(
    spec: DesignSpaceSpec,
    applies_if: EvaluatorSpec[Answer[bool]],
) -> DesignSpaceSpec:
    """Compose an outer scope's applicability into every executable declaration."""

    return DesignSpaceSpec(
        spec.problem_schema,
        tuple(
            Decision(
                item.path,
                item.value_semantics,
                item.domain,
                combine_applicability(applies_if, item.applies_if),
                item.proposal,
            )
            for item in spec.decisions
        ),
        tuple(
            DerivedProperty(
                item.path,
                item.value_semantics,
                item.evaluator,
                combine_applicability(applies_if, item.applies_if),
            )
            for item in spec.properties
        ),
        tuple(
            Constraint(
                item.path,
                item.evaluator,
                combine_applicability(applies_if, item.applies_if),
            )
            for item in spec.constraints
        ),
        tuple(ConstraintSet(item.name, item.constraints) for item in spec.constraint_sets),
        spec.readiness_profiles,
    )


def assemble_specs(specs: tuple[DesignSpaceSpec, ...]) -> DesignSpaceSpec:
    """Combine disjoint specifications into one flat specification.

    Only the FINN authoring checks run here.  The returned ordinary
    ``DesignSpaceSpec`` is still validated by the generic engine.
    """

    issues = []
    path_values = tuple(str(path) for spec in specs for path in spec_declaration_paths(spec))
    for duplicate in duplicate_values(path_values):
        issues.append(
            SpecAuthoringIssue(
                "declaration-path-duplicate",
                duplicate,
                f"declaration path {duplicate!r} is duplicated",
            )
        )
    constraint_set_names = tuple(item.name for spec in specs for item in spec.constraint_sets)
    for duplicate in duplicate_values(constraint_set_names):
        issues.append(
            SpecAuthoringIssue(
                "constraint-set-name-duplicate",
                duplicate,
                f"constraint-set name {duplicate!r} is duplicated",
            )
        )
    readiness_names = tuple(item.name for spec in specs for item in spec.readiness_profiles)
    for duplicate in duplicate_values(readiness_names):
        issues.append(
            SpecAuthoringIssue(
                "readiness-profile-name-duplicate",
                duplicate,
                f"readiness-profile name {duplicate!r} is duplicated",
            )
        )
    if issues:
        raise SpecAuthoringError(tuple(issues))
    return DesignSpaceSpec(
        ProblemSchema(tuple(field for spec in specs for field in spec.problem_schema.fields)),
        tuple(decision for spec in specs for decision in spec.decisions),
        tuple(prop for spec in specs for prop in spec.properties),
        tuple(constraint for spec in specs for constraint in spec.constraints),
        tuple(group for spec in specs for group in spec.constraint_sets),
        tuple(profile for spec in specs for profile in spec.readiness_profiles),
    )


__all__ = [
    "SpecAuthoringError",
    "SpecAuthoringIssue",
    "assemble_specs",
    "combine_applicability",
    "duplicate_values",
    "gate_spec",
    "map_answer_findings",
    "prefixed",
    "rebase_dependency",
    "rebase_evaluator",
    "rebase_spec",
    "spec_declaration_paths",
]
