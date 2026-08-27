# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN Kernel authoring metadata over ordinary flat design-space specs.

This module does not add a Kernel declaration kind to the generic engine.  It
records the domain-level identities and paths shared by concrete Kernel
definitions and provides deterministic flat-spec assembly for larger scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
    AbsenceMode,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemSchema,
    QualifiedPath,
    Unresolved,
    as_object_semantics,
    validate_region,
)
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import RegionValidationReport

_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_REPORT_SEMANTICS = as_object_semantics(REGION_VALIDATION_REPORT_SEMANTICS)


@dataclass(frozen=True, order=True)
class RegionDeclaration:
    """Identity and derived-property path of one complete region declaration."""

    id: str
    property_path: QualifiedPath

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("region declaration id must not be empty")


@dataclass(frozen=True, order=True)
class BindingDefinition:
    """Stable identity of one binding alternative in a Kernel definition."""

    id: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("binding definition id must not be empty")


@dataclass(frozen=True)
class KernelAuthoringIssue:
    """One deterministic error in Kernel authoring metadata or assembly."""

    code: str
    path: str
    message: str


class KernelAuthoringError(ValueError):
    """Raised when Kernel metadata cannot form an unambiguous flat scope."""

    def __init__(self, issues: tuple[KernelAuthoringIssue, ...]) -> None:
        self.issues = tuple(sorted(issues, key=lambda issue: (issue.path, issue.code)))
        super().__init__(f"Kernel authoring failed with {len(self.issues)} issue(s)")


def _duplicate_values(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(value for value in set(values) if values.count(value) > 1))


@dataclass(frozen=True)
class KernelDefinition:
    """FINN/domain association around one ordinary Kernel-scoped specification."""

    id: str
    spec: DesignSpaceSpec
    region_declarations: tuple[RegionDeclaration, ...]
    binding_definitions: tuple[BindingDefinition, ...]
    selected_region_path: QualifiedPath
    binding_decision_path: QualifiedPath
    binding_witness_path: QualifiedPath
    structural_readiness_profile: str
    binding_readiness_profile: str

    def __post_init__(self) -> None:
        issues = []
        if not self.id:
            issues.append(
                KernelAuthoringIssue("kernel-id-empty", "kernel", "Kernel id must not be empty")
            )
        for duplicate in _duplicate_values(
            tuple(declaration.id for declaration in self.region_declarations)
        ):
            issues.append(
                KernelAuthoringIssue(
                    "region-declaration-id-duplicate",
                    f"{self.id}.region.{duplicate}",
                    f"region declaration id {duplicate!r} is duplicated",
                )
            )
        for duplicate in _duplicate_values(
            tuple(binding.id for binding in self.binding_definitions)
        ):
            issues.append(
                KernelAuthoringIssue(
                    "binding-definition-id-duplicate",
                    f"{self.id}.binding.{duplicate}",
                    f"binding definition id {duplicate!r} is duplicated",
                )
            )
        declared_properties = {item.path for item in self.spec.properties}
        declared_decisions = {item.path for item in self.spec.decisions}
        if self.selected_region_path not in declared_properties:
            issues.append(
                KernelAuthoringIssue(
                    "selected-region-path-missing",
                    str(self.selected_region_path),
                    "selected region path is not a derived property",
                )
            )
        if self.binding_witness_path not in declared_properties:
            issues.append(
                KernelAuthoringIssue(
                    "binding-witness-path-missing",
                    str(self.binding_witness_path),
                    "binding witness path is not a derived property",
                )
            )
        if self.binding_decision_path not in declared_decisions:
            issues.append(
                KernelAuthoringIssue(
                    "binding-decision-path-missing",
                    str(self.binding_decision_path),
                    "binding path is not a decision",
                )
            )
        for declaration in self.region_declarations:
            if declaration.property_path not in declared_properties:
                issues.append(
                    KernelAuthoringIssue(
                        "region-declaration-path-missing",
                        str(declaration.property_path),
                        f"region declaration {declaration.id!r} has no derived property",
                    )
                )
        profile_names = {profile.name for profile in self.spec.readiness_profiles}
        for profile_name, code in (
            (self.structural_readiness_profile, "structural-readiness-profile-missing"),
            (self.binding_readiness_profile, "binding-readiness-profile-missing"),
        ):
            if profile_name not in profile_names:
                issues.append(
                    KernelAuthoringIssue(
                        code,
                        f"{self.id}.readiness.{profile_name}",
                        f"readiness profile {profile_name!r} is not declared",
                    )
                )
        if issues:
            raise KernelAuthoringError(tuple(issues))


@dataclass(frozen=True)
class KernelInstance:
    """One evaluated Kernel definition associated with selected semantics."""

    definition_id: str
    point: DesignPoint
    region: DataflowRegion
    binding_id: object
    binding_witness: object


@dataclass(frozen=True)
class KernelSemanticDeclarations:
    """Generic selected-region and structural-validation declarations."""

    selected_region: DerivedProperty
    validation_report: DerivedProperty
    structural_constraint: Constraint


def build_kernel_semantic_declarations(
    region_declarations: tuple[RegionDeclaration, ...],
    *,
    selected_region_path: QualifiedPath,
    validation_report_path: QualifiedPath,
    structural_constraint_path: QualifiedPath,
) -> KernelSemanticDeclarations:
    """Build the shared selection and structural-validation spine for a Kernel."""
    if not region_declarations:
        raise KernelAuthoringError(
            (
                KernelAuthoringIssue(
                    "region-declaration-inventory-empty",
                    str(selected_region_path),
                    "a Kernel must declare at least one complete region",
                ),
            )
        )
    dependencies = tuple(
        DependencyRef.property(
            f"region_{index}",
            declaration.property_path,
            _REGION_SEMANTICS,
            absence=AbsenceMode.ALLOWS_ABSENT,
        )
        for index, declaration in enumerate(region_declarations)
    )

    def select_region(values: DependencyView) -> Answer[object]:
        candidates = tuple(value for value in values.values() if value is not ABSENT)
        if len(candidates) != 1:
            raise AssertionError("one region declaration must be applicable")
        return Decided(cast(DataflowRegion, candidates[0]))

    region_ref = DependencyRef.property("region", selected_region_path, _REGION_SEMANTICS)
    report_ref = DependencyRef.property("report", validation_report_path, _REPORT_SEMANTICS)

    def validate_selected_region(values: DependencyView) -> Answer[object]:
        return Decided(validate_region(cast(DataflowRegion, values["region"])))

    def structurally_well_formed(values: DependencyView) -> Answer[bool]:
        return Decided(not cast(RegionValidationReport, values["report"]))

    return KernelSemanticDeclarations(
        DerivedProperty(
            selected_region_path,
            _REGION_SEMANTICS,
            EvaluatorSpec(dependencies, select_region),
        ),
        DerivedProperty(
            validation_report_path,
            _REPORT_SEMANTICS,
            EvaluatorSpec((region_ref,), validate_selected_region),
        ),
        Constraint(
            structural_constraint_path,
            EvaluatorSpec((report_ref,), structurally_well_formed),
        ),
    )


def instantiate_kernel(
    engine: Engine, definition: KernelDefinition, point: DesignPoint
) -> Answer[KernelInstance]:
    """Construct a Kernel instance when region, binding, and witness are resolved."""
    region_answer = engine.query_property(point, definition.selected_region_path)
    if not isinstance(region_answer, Decided):
        return region_answer
    binding = point.assignments.get(definition.binding_decision_path)
    if binding is None:
        return Unresolved(
            (
                Finding(
                    FindingKind.BLOCKER,
                    "kernel-binding-unassigned",
                    definition.binding_decision_path,
                    "Kernel instance requires a committed binding",
                ),
            )
        )
    witness_answer = engine.query_property(point, definition.binding_witness_path)
    if not isinstance(witness_answer, Decided):
        return witness_answer
    return Decided(
        KernelInstance(
            definition.id,
            point,
            cast(DataflowRegion, region_answer.value),
            binding,
            witness_answer.value,
        )
    )


def _spec_declaration_paths(spec: DesignSpaceSpec) -> tuple[QualifiedPath, ...]:
    return tuple(
        [field.path for field in spec.problem_schema.fields]
        + [decision.path for decision in spec.decisions]
        + [prop.path for prop in spec.properties]
        + [constraint.path for constraint in spec.constraints]
    )


def assemble_kernel_specs(
    definitions: tuple[KernelDefinition, ...],
    *,
    additions: DesignSpaceSpec = DesignSpaceSpec(),
) -> DesignSpaceSpec:
    """Combine Kernel declarations into one flat specification.

    The function performs only FINN authoring checks.  The returned ordinary
    ``DesignSpaceSpec`` is still validated by the generic engine.
    """
    issues = []
    for duplicate in _duplicate_values(tuple(definition.id for definition in definitions)):
        issues.append(
            KernelAuthoringIssue(
                "kernel-id-duplicate",
                duplicate,
                f"Kernel id {duplicate!r} is duplicated",
            )
        )
    all_specs = tuple(definition.spec for definition in definitions) + (additions,)
    path_values = tuple(str(path) for spec in all_specs for path in _spec_declaration_paths(spec))
    for duplicate in _duplicate_values(path_values):
        issues.append(
            KernelAuthoringIssue(
                "declaration-path-duplicate",
                duplicate,
                f"declaration path {duplicate!r} is duplicated",
            )
        )
    constraint_set_names = tuple(item.name for spec in all_specs for item in spec.constraint_sets)
    for duplicate in _duplicate_values(constraint_set_names):
        issues.append(
            KernelAuthoringIssue(
                "constraint-set-name-duplicate",
                duplicate,
                f"constraint-set name {duplicate!r} is duplicated",
            )
        )
    readiness_names = tuple(item.name for spec in all_specs for item in spec.readiness_profiles)
    for duplicate in _duplicate_values(readiness_names):
        issues.append(
            KernelAuthoringIssue(
                "readiness-profile-name-duplicate",
                duplicate,
                f"readiness-profile name {duplicate!r} is duplicated",
            )
        )
    if issues:
        raise KernelAuthoringError(tuple(issues))
    return DesignSpaceSpec(
        ProblemSchema(tuple(field for spec in all_specs for field in spec.problem_schema.fields)),
        tuple(decision for spec in all_specs for decision in spec.decisions),
        tuple(prop for spec in all_specs for prop in spec.properties),
        tuple(constraint for spec in all_specs for constraint in spec.constraints),
        tuple(group for spec in all_specs for group in spec.constraint_sets),
        tuple(profile for spec in all_specs for profile in spec.readiness_profiles),
    )


def _combined_applicability(
    outer: EvaluatorSpec[Answer[bool]],
    inner: EvaluatorSpec[Answer[bool]] | None,
) -> EvaluatorSpec[Answer[bool]]:
    dependencies = outer.dependencies + (() if inner is None else inner.dependencies)
    names = tuple(dependency.name for dependency in dependencies)
    if len(names) != len(set(names)):
        raise KernelAuthoringError(
            (
                KernelAuthoringIssue(
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


def gate_design_space_spec(
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
                _combined_applicability(applies_if, item.applies_if),
                item.proposal,
            )
            for item in spec.decisions
        ),
        tuple(
            DerivedProperty(
                item.path,
                item.value_semantics,
                item.evaluator,
                _combined_applicability(applies_if, item.applies_if),
            )
            for item in spec.properties
        ),
        tuple(
            Constraint(
                item.path,
                item.evaluator,
                _combined_applicability(applies_if, item.applies_if),
            )
            for item in spec.constraints
        ),
        tuple(ConstraintSet(item.name, item.constraints) for item in spec.constraint_sets),
        spec.readiness_profiles,
    )


__all__ = [
    "BindingDefinition",
    "KernelAuthoringError",
    "KernelAuthoringIssue",
    "KernelDefinition",
    "KernelInstance",
    "KernelSemanticDeclarations",
    "RegionDeclaration",
    "assemble_kernel_specs",
    "build_kernel_semantic_declarations",
    "gate_design_space_spec",
    "instantiate_kernel",
]
