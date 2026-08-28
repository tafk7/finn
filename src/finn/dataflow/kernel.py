# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""FINN Kernel authoring metadata over ordinary flat design-space specs.

This module does not add a Kernel declaration kind to the generic engine.  It
records the domain-level identities and paths shared by concrete Kernel
definitions and provides deterministic flat-spec assembly for larger scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_REGION_SEMANTICS,
    REGION_VALIDATION_REPORT_SEMANTICS,
    AbsenceMode,
    Answer,
    Constraint,
    Decided,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
    as_object_semantics,
    validate_region,
)
from finn.dataflow.region import DataflowRegion
from finn.dataflow.region_validation import RegionValidationReport
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
    duplicate_values,
    gate_spec,
    prefixed,
    rebase_spec,
    spec_declaration_paths,
)

_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_REPORT_SEMANTICS = as_object_semantics(REGION_VALIDATION_REPORT_SEMANTICS)
T = TypeVar("T")


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


# Kernel authoring reuses the generic flat-spec authoring diagnostics.
KernelAuthoringIssue = SpecAuthoringIssue
KernelAuthoringError = SpecAuthoringError
_duplicate_values = duplicate_values


@dataclass(frozen=True)
class KernelDefinition:
    """FINN/domain association around one ordinary Kernel-scoped specification."""

    id: str
    spec: DesignSpaceSpec
    region_declarations: tuple[RegionDeclaration, ...]
    binding_definitions: tuple[BindingDefinition, ...]
    selected_region_path: QualifiedPath
    binding_decision_path: QualifiedPath | None
    binding_selection_path: QualifiedPath | None
    structural_readiness_profile: str
    binding_readiness_profile: str | None

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
        binding_paths = (self.binding_decision_path, self.binding_selection_path)
        if (binding_paths[0] is None) != (binding_paths[1] is None):
            issues.append(
                KernelAuthoringIssue(
                    "binding-path-pair-incomplete",
                    self.id,
                    "binding decision and selection paths must both be present or absent",
                )
            )
        if self.binding_decision_path is None:
            if self.binding_definitions:
                issues.append(
                    KernelAuthoringIssue(
                        "bindingless-kernel-has-bindings",
                        self.id,
                        "a bindingless Kernel must not declare binding alternatives",
                    )
                )
            if self.binding_readiness_profile is not None:
                issues.append(
                    KernelAuthoringIssue(
                        "bindingless-kernel-has-readiness",
                        self.id,
                        "a bindingless Kernel must not declare binding readiness",
                    )
                )
        else:
            if self.binding_selection_path not in declared_properties:
                issues.append(
                    KernelAuthoringIssue(
                        "binding-selection-path-missing",
                        str(self.binding_selection_path),
                        "binding selection path is not a derived property",
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
        profiles = [(self.structural_readiness_profile, "structural-readiness-profile-missing")]
        if self.binding_readiness_profile is not None:
            profiles.append((self.binding_readiness_profile, "binding-readiness-profile-missing"))
        for profile_name, code in profiles:
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

    def place(
        self,
        instance_id: str,
        prefix: QualifiedPath | str,
        *,
        shared_problem_paths: dict[QualifiedPath, QualifiedPath] | None = None,
    ) -> KernelPlacement:
        """Place this definition beneath an owning prefix.

        Problem fields are instance-local unless the caller explicitly maps a
        definition-local field to a shared outer-scope path.
        """
        return _place_kernel_definition(
            self,
            instance_id,
            QualifiedPath.parse(prefix),
            shared_problem_paths or {},
        )


@dataclass(frozen=True)
class KernelPlacement:
    """One path-rebased placement of a reusable Kernel definition."""

    definition_id: str
    instance_id: str
    spec: DesignSpaceSpec
    path_mapping: tuple[tuple[QualifiedPath, QualifiedPath], ...]
    shared_problem_paths: tuple[QualifiedPath, ...]
    region_declarations: tuple[RegionDeclaration, ...]
    binding_definitions: tuple[BindingDefinition, ...]
    selected_region_path: QualifiedPath
    binding_decision_path: QualifiedPath | None
    binding_selection_path: QualifiedPath | None
    structural_readiness_profile: str
    binding_readiness_profile: str | None

    def path(self, definition_local_path: QualifiedPath | str) -> QualifiedPath:
        local = QualifiedPath.parse(definition_local_path)
        mapping = dict(self.path_mapping)
        try:
            return mapping[local]
        except KeyError as exc:
            raise KeyError(f"path {local} does not belong to this Kernel definition") from exc


@dataclass(frozen=True)
class KernelInstance:
    """One evaluated Kernel definition associated with selected semantics."""

    definition_id: str
    instance_id: str
    point: DesignPoint
    region: DataflowRegion
    binding_id: object | None
    binding_selection: object | None


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


KernelDefinitionOrPlacement = KernelDefinition | KernelPlacement


def instantiate_kernel(
    engine: Engine, definition: KernelDefinitionOrPlacement, point: DesignPoint
) -> Answer[KernelInstance]:
    """Construct a resolved selection; feasibility remains a separate query."""
    region_answer = engine.query_property(point, definition.selected_region_path)
    if not isinstance(region_answer, Decided):
        return region_answer
    if definition.binding_decision_path is None:
        return Decided(
            KernelInstance(
                definition.id
                if isinstance(definition, KernelDefinition)
                else definition.definition_id,
                definition.id
                if isinstance(definition, KernelDefinition)
                else definition.instance_id,
                point,
                cast(DataflowRegion, region_answer.value),
                None,
                None,
            )
        )
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
    if definition.binding_selection_path is None:
        raise AssertionError("binding decision requires a binding-selection path")
    selection_answer = engine.query_property(point, definition.binding_selection_path)
    if not isinstance(selection_answer, Decided):
        return selection_answer
    return Decided(
        KernelInstance(
            definition.id if isinstance(definition, KernelDefinition) else definition.definition_id,
            definition.id if isinstance(definition, KernelDefinition) else definition.instance_id,
            point,
            cast(DataflowRegion, region_answer.value),
            binding,
            selection_answer.value,
        )
    )


def _place_kernel_definition(
    definition: KernelDefinition,
    instance_id: str,
    prefix: QualifiedPath,
    shared_problem_paths: dict[QualifiedPath, QualifiedPath],
) -> KernelPlacement:
    if not instance_id:
        raise ValueError("Kernel instance id must not be empty")
    declared_paths = spec_declaration_paths(definition.spec)
    problem_paths = {field.path for field in definition.spec.problem_schema.fields}
    if not set(shared_problem_paths).issubset(problem_paths):
        unknown = sorted(set(shared_problem_paths) - problem_paths)
        raise KernelAuthoringError(
            tuple(
                KernelAuthoringIssue(
                    "shared-problem-path-unknown",
                    str(path),
                    "shared path is not a problem field of this Kernel definition",
                )
                for path in unknown
            )
        )
    path_mapping = {
        path: shared_problem_paths.get(path, prefixed(prefix, path)) for path in declared_paths
    }
    spec = rebase_spec(
        definition.spec,
        path_mapping,
        name_prefix=instance_id,
        dropped_problem_paths=frozenset(shared_problem_paths),
    )
    return KernelPlacement(
        definition.id,
        instance_id,
        spec,
        tuple(sorted(path_mapping.items())),
        tuple(sorted(shared_problem_paths.values())),
        tuple(
            RegionDeclaration(item.id, path_mapping[item.property_path])
            for item in definition.region_declarations
        ),
        definition.binding_definitions,
        path_mapping[definition.selected_region_path],
        None
        if definition.binding_decision_path is None
        else path_mapping[definition.binding_decision_path],
        None
        if definition.binding_selection_path is None
        else path_mapping[definition.binding_selection_path],
        f"{instance_id}.{definition.structural_readiness_profile}",
        None
        if definition.binding_readiness_profile is None
        else f"{instance_id}.{definition.binding_readiness_profile}",
    )


def assemble_kernel_specs(
    definitions: tuple[KernelDefinitionOrPlacement, ...],
    *,
    additions: DesignSpaceSpec = DesignSpaceSpec(),
) -> DesignSpaceSpec:
    """Combine Kernel declarations into one flat specification.

    The function performs only FINN authoring checks.  The returned ordinary
    ``DesignSpaceSpec`` is still validated by the generic engine.
    """
    identities = tuple(
        definition.id if isinstance(definition, KernelDefinition) else definition.instance_id
        for definition in definitions
    )
    issues = [
        KernelAuthoringIssue(
            "kernel-id-duplicate",
            duplicate,
            f"Kernel id {duplicate!r} is duplicated",
        )
        for duplicate in duplicate_values(identities)
    ]
    all_specs = tuple(definition.spec for definition in definitions) + (additions,)
    try:
        assembled = assemble_specs(all_specs)
    except SpecAuthoringError as exc:
        raise KernelAuthoringError(tuple(issues) + exc.issues) from exc
    if issues:
        raise KernelAuthoringError(tuple(issues))
    return assembled


def gate_design_space_spec(
    spec: DesignSpaceSpec,
    applies_if: EvaluatorSpec[Answer[bool]],
) -> DesignSpaceSpec:
    """Compose an outer scope's applicability into every executable declaration."""
    return gate_spec(spec, applies_if)


__all__ = [
    "BindingDefinition",
    "KernelAuthoringError",
    "KernelAuthoringIssue",
    "KernelDefinition",
    "KernelInstance",
    "KernelPlacement",
    "KernelDefinitionOrPlacement",
    "KernelSemanticDeclarations",
    "RegionDeclaration",
    "assemble_kernel_specs",
    "build_kernel_semantic_declarations",
    "gate_design_space_spec",
    "instantiate_kernel",
]
