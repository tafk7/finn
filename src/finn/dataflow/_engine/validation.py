# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic validation and construction of indexed design spaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .declarations import (
    Constraint,
    Decision,
    DeclarationKind,
    DependencyKind,
    DependencyRef,
    DerivedProperty,
    DesignSpaceSpec,
    ProblemField,
    ReadinessProfile,
)
from .errors import ValidationError
from .facts import EvaluationPlan, GraphAnalysis, compile_facts, render_provenance
from .primitives import QualifiedPath
from .results import Finding, FindingKind, ordered_findings

_SPEC_PATH = QualifiedPath("specification")


@dataclass(frozen=True, slots=True)
class DeclarationInspection:
    path: QualifiedPath
    kind: DeclarationKind
    value_type: str | None
    has_candidates: bool = False


@dataclass(frozen=True, slots=True)
class DesignSpaceInspection:
    declarations: tuple[DeclarationInspection, ...]
    problem_fields: tuple[QualifiedPath, ...]
    constraint_sets: tuple[str, ...]
    readiness_profiles: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DesignSpace:
    """A validated declaration index and its compact runtime evaluation plan."""

    decisions: Mapping[QualifiedPath, Decision]
    properties: Mapping[QualifiedPath, DerivedProperty]
    constraints: Mapping[QualifiedPath, Constraint]
    problem_fields: Mapping[QualifiedPath, ProblemField]
    constraint_sets: Mapping[str, tuple[QualifiedPath, ...]]
    readiness_profiles: Mapping[str, ReadinessProfile]
    findings: tuple[Finding, ...]
    _plan: EvaluationPlan

    def __post_init__(self) -> None:
        for name in (
            "decisions",
            "properties",
            "constraints",
            "problem_fields",
            "constraint_sets",
            "readiness_profiles",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "findings", ordered_findings(self.findings))


def _authoring(code: str, path: QualifiedPath, message: str) -> Finding:
    return Finding(FindingKind.AUTHORING, code, path, message)


def _source_collections(
    space_or_spec: DesignSpaceSpec | DesignSpace,
) -> tuple[
    Iterable[Decision],
    Iterable[DerivedProperty],
    Iterable[Constraint],
    Iterable[ProblemField],
    Iterable[str],
    Iterable[str],
]:
    if isinstance(space_or_spec, DesignSpace):
        return (
            space_or_spec.decisions.values(),
            space_or_spec.properties.values(),
            space_or_spec.constraints.values(),
            space_or_spec.problem_fields.values(),
            space_or_spec.constraint_sets,
            space_or_spec.readiness_profiles,
        )
    return (
        space_or_spec.decisions,
        space_or_spec.properties,
        space_or_spec.constraints,
        space_or_spec.problem_schema.fields,
        (group.name for group in space_or_spec.constraint_sets),
        (profile.name for profile in space_or_spec.readiness_profiles),
    )


def inspect_design_space(space_or_spec: DesignSpaceSpec | DesignSpace) -> DesignSpaceInspection:
    decisions, properties, constraints, fields, sets, profiles = _source_collections(space_or_spec)
    declarations = [
        *(
            DeclarationInspection(
                item.path,
                DeclarationKind.DECISION,
                item.value_semantics.name,
                item.domain.candidates is not None,
            )
            for item in decisions
        ),
        *(
            DeclarationInspection(
                item.path, DeclarationKind.DERIVED_PROPERTY, item.value_semantics.name
            )
            for item in properties
        ),
        *(
            DeclarationInspection(item.path, DeclarationKind.CONSTRAINT, None)
            for item in constraints
        ),
    ]
    return DesignSpaceInspection(
        tuple(sorted(declarations, key=lambda item: (item.path, item.kind.value))),
        tuple(sorted(field.path for field in fields)),
        tuple(sorted(sets)),
        tuple(sorted(profiles)),
    )


def _duplicate_paths(entries: Iterable[tuple[QualifiedPath, str]]) -> list[Finding]:
    first_kind: dict[QualifiedPath, str] = {}
    duplicates: dict[QualifiedPath, set[str]] = {}
    for path, kind in entries:
        if path in first_kind:
            duplicates.setdefault(path, {first_kind[path]}).add(kind)
        else:
            first_kind[path] = kind
    return [
        _authoring(
            "duplicate-path",
            path,
            f"path is declared more than once ({', '.join(sorted(kinds))})",
        )
        for path, kinds in sorted(duplicates.items())
    ]


def _evaluator_dependencies(
    specification: DesignSpaceSpec,
) -> Iterable[tuple[QualifiedPath, str, tuple[DependencyRef, ...]]]:
    for decision in specification.decisions:
        if decision.applies_if is not None:
            yield decision.path, "applicability", decision.applies_if.dependencies
        yield decision.path, "domain", decision.domain.dependencies
        if decision.domain.candidates is not None:
            yield decision.path, "candidates", decision.domain.candidates.dependencies
        if decision.proposal is not None:
            yield decision.path, "proposal", decision.proposal.dependencies
    for derived_property in specification.properties:
        if derived_property.applies_if is not None:
            yield (
                derived_property.path,
                "applicability",
                derived_property.applies_if.dependencies,
            )
        yield derived_property.path, "property", derived_property.evaluator.dependencies
    for constraint in specification.constraints:
        if constraint.applies_if is not None:
            yield constraint.path, "applicability", constraint.applies_if.dependencies
        yield constraint.path, "constraint", constraint.evaluator.dependencies


def _check_references(
    specification: DesignSpaceSpec,
    decisions: Mapping[QualifiedPath, Decision],
    properties: Mapping[QualifiedPath, DerivedProperty],
    constraints: Mapping[QualifiedPath, Constraint],
    fields: Mapping[QualifiedPath, ProblemField],
) -> list[Finding]:
    errors: list[Finding] = []
    indexes: dict[
        DependencyKind, Mapping[QualifiedPath, Decision | DerivedProperty | ProblemField]
    ] = {
        DependencyKind.DECISION: decisions,
        DependencyKind.PROPERTY: properties,
        DependencyKind.PROBLEM: fields,
    }
    labels = {
        DependencyKind.DECISION: "decision",
        DependencyKind.PROPERTY: "derived property",
        DependencyKind.PROBLEM: "problem field",
    }
    all_paths = {
        **{path: "decision" for path in decisions},
        **{path: "derived property" for path in properties},
        **{path: "constraint" for path in constraints},
        **{path: "problem field" for path in fields},
    }
    for owner, role, dependencies in _evaluator_dependencies(specification):
        names = [ref.name for ref in dependencies]
        for name in sorted({name for name in names if names.count(name) > 1}):
            errors.append(
                _authoring(
                    "duplicate-dependency-name",
                    owner,
                    f"{role} binds dependency name {name!r} more than once",
                )
            )
        for ref in dependencies:
            target = indexes[ref.kind].get(ref.path)
            if target is None:
                actual = all_paths.get(ref.path)
                if actual is None:
                    message = (
                        f"{role} dependency {ref.name!r} does not resolve to a {labels[ref.kind]}"
                    )
                    code = "unresolved-reference"
                else:
                    message = (
                        f"{role} dependency {ref.name!r} is a {actual}, not a {labels[ref.kind]}"
                    )
                    code = "wrong-reference-kind"
                errors.append(_authoring(code, owner, message))
                continue
            semantics = target.value_semantics
            if not ref.value_semantics.is_compatible_with(semantics):
                errors.append(
                    _authoring(
                        "dependency-type-mismatch",
                        owner,
                        f"{role} dependency {ref.name!r} has incompatible value semantics",
                    )
                )
    return errors


def _validate_groups(
    specification: DesignSpaceSpec,
    decisions: Mapping[QualifiedPath, Decision],
    properties: Mapping[QualifiedPath, DerivedProperty],
    constraints: Mapping[QualifiedPath, Constraint],
) -> tuple[
    dict[str, tuple[QualifiedPath, ...]],
    dict[str, ReadinessProfile],
    list[Finding],
]:
    errors: list[Finding] = []
    constraint_sets: dict[str, tuple[QualifiedPath, ...]] = {}
    for group in specification.constraint_sets:
        if group.name in constraint_sets:
            errors.append(
                _authoring("duplicate-constraint-set", _SPEC_PATH, f"duplicate set {group.name!r}")
            )
            continue
        constraint_sets[group.name] = tuple(group.constraints)
        for path in group.constraints:
            if path not in constraints:
                errors.append(
                    _authoring(
                        "constraint-set-reference",
                        path,
                        f"constraint set {group.name!r} names a non-constraint",
                    )
                )

    readiness_profiles: dict[str, ReadinessProfile] = {}
    for profile in specification.readiness_profiles:
        if profile.name in readiness_profiles:
            errors.append(
                _authoring(
                    "duplicate-readiness-profile", _SPEC_PATH, f"duplicate profile {profile.name!r}"
                )
            )
            continue
        readiness_profiles[profile.name] = profile
        for path, index, label in (
            *((path, decisions, "decision") for path in profile.decisions),
            *((path, properties, "property") for path in profile.properties),
            *((path, constraints, "constraint") for path in profile.constraints),
        ):
            if path not in index:
                errors.append(
                    _authoring(
                        "readiness-profile-reference",
                        path,
                        f"profile {profile.name!r} path is not a {label}",
                    )
                )
    return constraint_sets, readiness_profiles, errors


def _cycle_findings(analysis: GraphAnalysis) -> list[Finding]:
    findings: list[Finding] = []
    for component in analysis.core_cycles:
        component_set = frozenset(component)
        records = tuple(
            record
            for source in component
            for dependency in sorted(analysis.prerequisites[source] & component_set)
            for record in analysis.edge_provenance[(source, dependency)]
        )
        findings.append(
            Finding(
                FindingKind.AUTHORING,
                "core-prerequisite-cycle",
                min(node.path for node in component),
                "semantic facts form a fatal prerequisite cycle",
                (
                    ("facts", tuple(str(node) for node in component)),
                    ("provenance", render_provenance(records)),
                ),
                tuple(node.path for node in component),
            )
        )
    return findings


def _dead_proposal_findings(analysis: GraphAnalysis) -> tuple[Finding, ...]:
    return tuple(
        Finding(
            FindingKind.AUTHORING,
            "dead-proposal",
            path,
            "proposal transitively requires its own decision value",
            (
                ("facts", tuple(str(node) for node in dead.fact_trace)),
                ("provenance", render_provenance(dead.provenance)),
            ),
            tuple(node.path for node in dead.fact_trace),
        )
        for path, dead in analysis.dead_proposals.items()
    )


def validate_design_space(specification: DesignSpaceSpec) -> DesignSpace:
    """Validate semantic cross-references and compile the runtime plan."""

    decisions = {item.path: item for item in specification.decisions}
    properties = {item.path: item for item in specification.properties}
    constraints = {item.path: item for item in specification.constraints}
    fields = {item.path: item for item in specification.problem_schema.fields}

    entries = [
        *((item.path, "decision") for item in specification.decisions),
        *((item.path, "derived property") for item in specification.properties),
        *((item.path, "constraint") for item in specification.constraints),
        *((item.path, "problem field") for item in specification.problem_schema.fields),
    ]
    errors = _duplicate_paths(entries)
    errors.extend(_check_references(specification, decisions, properties, constraints, fields))
    constraint_sets, readiness_profiles, group_errors = _validate_groups(
        specification, decisions, properties, constraints
    )
    errors.extend(group_errors)
    if errors:
        raise ValidationError(errors)

    analysis, plan = compile_facts(specification)
    cycles = _cycle_findings(analysis)
    if cycles:
        raise ValidationError(cycles)
    findings = _dead_proposal_findings(analysis)
    return DesignSpace(
        decisions,
        properties,
        constraints,
        fields,
        constraint_sets,
        readiness_profiles,
        findings,
        plan,
    )


__all__ = [
    "DeclarationInspection",
    "DesignSpace",
    "DesignSpaceInspection",
    "inspect_design_space",
    "validate_design_space",
]
