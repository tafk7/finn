# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Source-free capture of accepted local physical capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from typing import Any, cast

from finn.kernels._engine import (
    Absent,
    Answer,
    Decided,
    DependencyKind,
    DependencyRef,
    Finding,
    FindingKind,
    QualifiedPath,
)
from finn.kernels.artifacts.build import ModuleBuildRequirements, module_build_fingerprint
from finn.parked.dataflow.model.identity import (
    ImplementationIdentity,
    comparison_type_identity,
    implementation_identity,
)
from finn.parked.dataflow.logical_values.composition import ImplementationPath
from finn.parked.dataflow.model.physical.authoring import PhysicallyUnsupported
from finn.parked.dataflow.model.physical.interface import KernelRealizationFacts, KernelStreamBinding
from finn.kernels.space.declarations import (
    ConstraintGroup,
    Projection,
    Space,
    ValueSource,
    declared_members,
    semantics_for,
)
from finn.kernels.space.occurrence import ProjectionAssessment, layer_runtime


@dataclass(frozen=True)
class CapturedDependency:
    kind: str
    path: str
    value: object


@dataclass(frozen=True)
class LocalPhysicalCapture:
    """Model-free physical evidence for one projected implementation point."""

    implementation: ImplementationIdentity
    occurrence_path: ImplementationPath
    occurrence_token: int
    dependencies: tuple[CapturedDependency, ...]
    point_fingerprint: str
    physical_fingerprint: str
    requirements: ModuleBuildRequirements
    physical: object


class PhysicalCaptureError(ValueError):
    """An accepted local codegen capability could not be captured."""

    def __init__(self, message: str, findings: tuple[Finding, ...] = ()) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


def _digest(*values: object) -> str:
    return hashlib.sha256("\n".join(repr(value) for value in values).encode("utf-8")).hexdigest()


def _occurrence_identity(implementation: object) -> tuple[ImplementationPath, int]:
    if not isinstance(implementation, Space):
        raise TypeError("local physical capture requires a Space occurrence")
    runtime = layer_runtime(implementation)
    return ImplementationPath(tuple(runtime.compiled.namespace.split("."))), id(runtime.engine)


def canonical_dependency_value(value: object) -> object:
    """Canonical comparison value with explicit pre-KP type relocation."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, QualifiedPath):
        return {"path": value.value}
    if isinstance(value, Enum):
        return {
            "enum": comparison_type_identity(value),
            "value": canonical_dependency_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": comparison_type_identity(value),
            "fields": tuple(
                (item.name, canonical_dependency_value(getattr(value, item.name)))
                for item in fields(value)
                if item.compare
            ),
        }
    if isinstance(value, Mapping):
        return {
            "mapping": tuple(
                sorted(
                    (
                        repr(canonical_dependency_value(key)),
                        canonical_dependency_value(item),
                    )
                    for key, item in value.items()
                )
            )
        }
    if isinstance(value, (tuple, list)):
        return tuple(canonical_dependency_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return {"set": tuple(sorted(repr(canonical_dependency_value(item)) for item in value))}
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return {"named": comparison_type_identity(value), "name": name}
    return {"typed_repr": comparison_type_identity(value), "value": repr(value)}


def _canonical_answer(answer: Answer[object]) -> object:
    if isinstance(answer, Decided):
        return {"decided": canonical_dependency_value(answer.value)}
    return {
        "absent" if isinstance(answer, Absent) else "unresolved": tuple(
            (
                finding.kind.value,
                finding.code,
                finding.path.value,
                tuple((name, canonical_dependency_value(value)) for name, value in finding.values),
            )
            for finding in answer.findings
        )
    }


def _physical_dependency_snapshot(implementation: Space) -> tuple[CapturedDependency, ...]:
    return capture_assessment_dependencies(implementation, "physical")


def capture_assessment_dependencies(
    implementation: Space, assessment: str | Projection[Any] | ConstraintGroup | DependencyRef
) -> tuple[CapturedDependency, ...]:
    """Capture the actual scoped dependency closure of one existing assessment.

    Codegen capture requests only its physical Projection. A consumer that also
    assesses a narrow interface or constraint group can account for those premises
    separately, without adding them to local codegen or evaluating a full graph.
    """
    from finn.kernels.space.compiler import _Ref, answer_for  # noqa: PLC0415

    reference_subject = assessment if isinstance(assessment, DependencyRef) else None
    name = (
        ""
        if reference_subject is not None
        else assessment
        if isinstance(assessment, str)
        else next(
            name
            for name, declaration in declared_members(type(implementation))
            if declaration is assessment
        )
    )
    declaration = getattr(type(implementation), name, None)
    if reference_subject is None and not isinstance(declaration, (Projection, ConstraintGroup)):
        projected: ProjectionAssessment[Any] = implementation.assess_view(name)
        return (
            CapturedDependency(
                "capability-output",
                f"{type(implementation).__name__}.{name}",
                _canonical_answer(cast("Answer[object]", projected.accepted_answer)),
            ),
        )
    runtime = layer_runtime(implementation)
    space = runtime.point.design_space
    captured: dict[tuple[str, str], CapturedDependency] = {}
    visiting: set[tuple[DependencyKind, QualifiedPath]] = set()

    def record(kind: str, path: QualifiedPath, answer: Answer[object]) -> None:
        captured[(kind, path.value)] = CapturedDependency(
            kind, path.value, _canonical_answer(answer)
        )

    def visit(reference: DependencyRef) -> None:
        key = (reference.kind, reference.path)
        if key in visiting:
            return
        visiting.add(key)
        path = reference.path
        if reference.kind is DependencyKind.PROBLEM:
            record(
                "problem",
                path,
                Decided(runtime.point.problem[path]) if path in runtime.point.problem else Absent(),
            )
            return
        if reference.kind is DependencyKind.DECISION:
            decision = space.decisions[path]
            if decision.applies_if is not None:
                for dependency in decision.applies_if.dependencies:
                    visit(dependency)
                applies = runtime.engine._query_applicability(runtime.point, path)
                record("applicability", path, applies)
                if not isinstance(applies, Decided) or not applies.value:
                    return
            for dependency in decision.domain.dependencies:
                visit(dependency)
            record(
                "decision",
                path,
                answer_for(
                    runtime.engine,
                    runtime.point,
                    _Ref(path, DependencyKind.DECISION, decision.value_semantics),
                ),
            )
            return
        property_declaration = space.properties.get(path)
        if property_declaration is not None:
            if property_declaration.applies_if is not None:
                for dependency in property_declaration.applies_if.dependencies:
                    visit(dependency)
                applies = runtime.engine._query_applicability(runtime.point, path)
                record("applicability", path, applies)
                if not isinstance(applies, Decided) or not applies.value:
                    return
            for dependency in property_declaration.evaluator.dependencies:
                visit(dependency)
            record("property", path, runtime.engine.query_property(runtime.point, path))
            return
        constraint = space.constraints.get(path)
        if constraint is None:
            raise ValueError(f"assessment dependency {path} is not declared")
        if constraint.applies_if is not None:
            for dependency in constraint.applies_if.dependencies:
                visit(dependency)
            applies = runtime.engine._query_applicability(runtime.point, path)
            record("applicability", path, applies)
            if not isinstance(applies, Decided) or not applies.value:
                return
        for dependency in constraint.evaluator.dependencies:
            visit(dependency)
        record(
            "constraint",
            path,
            cast(Any, runtime.engine.evaluate_constraints(runtime.point, (path,)).answers[path]),
        )

    if reference_subject is not None:
        visit(reference_subject)
        constraint_paths = set()
    elif isinstance(declaration, ConstraintGroup):
        constraint_paths = set(implementation.assess(declaration).answers)
    else:
        compiled = runtime.compiled.projection(name)
        for dependency in compiled.applicability.dependencies if compiled.applicability else ():
            visit(dependency)
        visit(
            DependencyRef(
                "output", compiled.output.path, compiled.output.kind, compiled.output.semantics
            )
        )
        readiness = space.readiness_profiles[compiled.readiness_profile]
        for path in readiness.decisions:
            visit(
                DependencyRef(
                    "readiness",
                    path,
                    DependencyKind.DECISION,
                    space.decisions[path].value_semantics,
                )
            )
        for path in readiness.properties:
            visit(
                DependencyRef(
                    "readiness",
                    path,
                    DependencyKind.PROPERTY,
                    space.properties[path].value_semantics,
                )
            )
        constraint_paths = set(readiness.constraints)
        for group in compiled.constraint_sets:
            constraint_paths.update(space.constraint_sets[group])
    for path in sorted(constraint_paths):
        visit(DependencyRef("constraint", path, DependencyKind.CONSTRAINT, semantics_for(bool)))
    visited_paths = {path for _kind, path in captured}

    def implementation_dependencies(compiled_space: object) -> None:
        namespace = getattr(compiled_space, "namespace", "")
        owner = getattr(compiled_space, "owner", None)
        if (
            namespace
            and isinstance(owner, type)
            and any(path == namespace or path.startswith(f"{namespace}.") for path in visited_paths)
        ):
            family = getattr(owner, "id", "")
            version = getattr(owner, "version", "")
            if isinstance(family, str) and family and isinstance(version, str) and version:
                captured[("implementation", namespace)] = CapturedDependency(
                    "implementation", namespace, {"family": family, "version": version}
                )
        for _name, child in getattr(compiled_space, "children", ()):
            implementation_dependencies(child)
        for _name, branch in getattr(compiled_space, "branches", ()):
            for case in branch.cases:
                implementation_dependencies(case.compiled)

    implementation_dependencies(runtime.compiled)
    return tuple(captured[key] for key in sorted(captured))


def capture_local_physical(implementation: object) -> LocalPhysicalCapture:
    if not isinstance(implementation, Space):
        raise TypeError("local physical capture requires a Space occurrence")
    assessment: ProjectionAssessment[Any] = implementation.assess_view("physical")
    if not isinstance(assessment.accepted_answer, Decided):
        raise PhysicalCaptureError(
            "local physical capability is not accepted",
            tuple(assessment.accepted_answer.findings),
        )
    physical: object = assessment.accepted_answer.value
    requirements = (
        physical
        if isinstance(physical, ModuleBuildRequirements)
        else getattr(physical, "requirements", None)
    )
    if not isinstance(requirements, ModuleBuildRequirements):
        raise TypeError("physical capability does not expose ModuleBuildRequirements")
    path, token = _occurrence_identity(implementation)
    identity = implementation_identity(implementation)
    dependencies = _physical_dependency_snapshot(implementation)
    physical_fingerprint = module_build_fingerprint(requirements)
    point_fingerprint = _digest(identity, path, dependencies)
    return LocalPhysicalCapture(
        identity,
        path,
        token,
        dependencies,
        point_fingerprint,
        physical_fingerprint,
        requirements,
        physical,
    )


def kernel_physical_refusal(kernel: Space, reason: str) -> Absent:
    namespace = layer_runtime(kernel).compiled.namespace
    return Absent(
        (
            Finding(
                FindingKind.REJECTION,
                "kernel-physically-unsupported",
                QualifiedPath(f"{namespace}.physical"),
                reason,
                (("kernel", getattr(type(kernel), "id", "") or type(kernel).__name__),),
            ),
        )
    )


def capture_kernel_realization(kernel: Space) -> KernelRealizationFacts:
    physical: Answer[object] = kernel.assess_view("physical").accepted_answer
    if not isinstance(physical, Decided):
        raise PhysicallyUnsupported(f"Kernel physical projection is not accepted: {physical}")
    if not isinstance(physical.value, ModuleBuildRequirements):
        raise PhysicallyUnsupported("Kernel physical projection returned the wrong value")
    streams_declaration = getattr(type(kernel), "physical_streams", None)
    if not isinstance(streams_declaration, ValueSource):
        raise PhysicallyUnsupported("Kernel has no physical stream-binding capability")
    bindings = kernel.answer(streams_declaration)
    if not isinstance(bindings, Decided):
        raise PhysicallyUnsupported(f"Kernel stream bindings are not available: {bindings}")
    return KernelRealizationFacts(
        physical.value, cast("tuple[KernelStreamBinding, ...]", bindings.value)
    )


def selected_child_realization(kernel: Space, role: str) -> Answer[KernelRealizationFacts]:
    selected = kernel.child(role)  # type: ignore[attr-defined]
    if not isinstance(selected, Decided):
        return cast("Answer[KernelRealizationFacts]", selected)
    child = selected.value
    if not isinstance(child, Space):
        return cast(
            "Answer[KernelRealizationFacts]",
            kernel_physical_refusal(kernel, f"child {role!r} is not a Space occurrence"),
        )
    physical: Answer[object] = child.assess_view("physical").accepted_answer
    if not isinstance(physical, Decided):
        return cast("Answer[KernelRealizationFacts]", physical)
    if not isinstance(physical.value, ModuleBuildRequirements):
        return cast(
            "Answer[KernelRealizationFacts]",
            kernel_physical_refusal(kernel, f"child {role!r} physical capability is not a module"),
        )
    streams_declaration = getattr(type(child), "physical_streams", None)
    if not isinstance(streams_declaration, ValueSource):
        return cast(
            "Answer[KernelRealizationFacts]",
            kernel_physical_refusal(
                kernel, f"child {role!r} has no physical stream-binding capability"
            ),
        )
    streams = child.answer(streams_declaration)
    if not isinstance(streams, Decided):
        return cast("Answer[KernelRealizationFacts]", streams)
    try:
        return Decided(
            KernelRealizationFacts(
                physical.value, cast("tuple[KernelStreamBinding, ...]", streams.value)
            )
        )
    except (PhysicallyUnsupported, TypeError, ValueError) as error:
        return cast("Answer[KernelRealizationFacts]", kernel_physical_refusal(kernel, str(error)))


__all__ = [
    "CapturedDependency",
    "LocalPhysicalCapture",
    "PhysicalCaptureError",
    "canonical_dependency_value",
    "capture_kernel_realization",
    "capture_local_physical",
    "capture_assessment_dependencies",
    "kernel_physical_refusal",
    "selected_child_realization",
]
