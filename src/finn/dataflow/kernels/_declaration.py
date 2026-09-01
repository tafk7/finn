# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private compiled declarations for physical Kernel authoring.

This leaf deliberately knows only the immutable records produced by a Kernel
scope.  Keeping it separate lets the authoring compiler and configured-Kernel
runtime depend in one direction without making the compiled declaration a
public contributor concept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, cast

from finn.dataflow.authoring.scope import ConstraintRef, Ref
from finn.dataflow.design import (
    DependencyKind,
    DependencyRef,
    DesignSpaceSpec,
    QualifiedPath,
    ValueSemantics,
)
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    duplicate_values,
)

if TYPE_CHECKING:
    from finn.dataflow.kernels.kernel import (
        CoveragePattern,
        Kernel,
        KernelParameter,
        SourceFile,
    )

Handles = TypeVar("Handles")


@dataclass(frozen=True)
class CompiledKernelDeclaration:
    """Everything one physical Kernel owns locally, as engine declarations."""

    id: str
    version: str
    namespace: str
    spec: DesignSpaceSpec
    coverage: CoveragePattern
    parameters: tuple[KernelParameter, ...] = ()
    coverage_constraints: tuple[QualifiedPath, ...] = ()
    sources: tuple[SourceFile, ...] = ()
    owner: type[Kernel] | None = None
    handles: object = field(default=None, repr=False, compare=False)
    decision_handles: tuple[Ref[object], ...] = field(default=(), repr=False, compare=False)
    constraint_handles: tuple[ConstraintRef, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        issues: list[SpecAuthoringIssue] = []
        if not self.id:
            issues.append(
                SpecAuthoringIssue("hardware-kernel-id-empty", "kernel", "Kernel id is empty")
            )
        if not self.version:
            issues.append(
                SpecAuthoringIssue("hardware-kernel-version-empty", self.id, "version is empty")
            )
        declared = {item.path for item in self.spec.constraints}
        for path in self.coverage_constraints:
            if path not in declared:
                issues.append(
                    SpecAuthoringIssue(
                        "hardware-coverage-constraint-missing",
                        str(path),
                        "the Kernel does not declare this coverage constraint",
                    )
                )
        for duplicate in duplicate_values(tuple(item.name for item in self.parameters)):
            issues.append(
                SpecAuthoringIssue(
                    "hardware-parameter-duplicate",
                    f"{self.id}.{duplicate}",
                    f"physical parameter {duplicate!r} is declared twice",
                )
            )
        issues.extend(self._local_reference_issues())
        if issues:
            raise SpecAuthoringError(tuple(issues))

    @property
    def _owned_paths(self) -> set[QualifiedPath]:
        return {
            *(item.path for item in self.spec.decisions),
            *(item.path for item in self.spec.properties),
        }

    def _is_local(self, path: QualifiedPath) -> bool:
        return str(path).startswith((f"{self.namespace}.", f"semantic.{self.namespace}."))

    def _local_reference_issues(self) -> list[SpecAuthoringIssue]:
        owned = self._owned_paths
        issues: list[SpecAuthoringIssue] = []
        for label, reference in self.references:
            if self._is_local(reference.path) and reference.path not in owned:
                issues.append(
                    SpecAuthoringIssue(
                        "hardware-reference-undeclared",
                        str(reference.path),
                        f"{label} names a local path this Kernel does not declare",
                    )
                )
        return issues

    @property
    def references(self) -> tuple[tuple[str, Ref[object]], ...]:
        referenced: list[tuple[str, Ref[object]]] = []
        for coverage in self.coverage.regions:
            referenced.append(
                (f"coverage {coverage.role!r} Region", cast("Ref[object]", coverage.region))
            )
            referenced.append(
                (
                    f"coverage {coverage.role!r} computation",
                    cast("Ref[object]", coverage.computation),
                )
            )
        for edge in self.coverage.edges:
            referenced.append((f"edge {edge.role!r} Network", cast("Ref[object]", edge.network)))
        for parameter in self.parameters:
            if parameter.source is not None:
                referenced.append((f"parameter {parameter.name!r}", parameter.source))
        return tuple(referenced)

    @property
    def imported_decisions(self) -> tuple[QualifiedPath, ...]:
        owned = {item.path for item in self.spec.decisions}
        found = [
            path
            for path, kind in self._read_declarations()
            if kind is DependencyKind.DECISION and path not in owned
        ]
        return tuple(dict.fromkeys(found))

    def _read_declarations(self) -> list[tuple[QualifiedPath, DependencyKind]]:
        groups: list[tuple[DependencyRef, ...]] = []
        for decision in self.spec.decisions:
            groups.append(decision.domain.dependencies)
            if decision.applies_if is not None:
                groups.append(decision.applies_if.dependencies)
        for item in self.spec.properties:
            groups.append(item.evaluator.dependencies)
            if item.applies_if is not None:
                groups.append(item.applies_if.dependencies)
        for constraint in self.spec.constraints:
            groups.append(constraint.evaluator.dependencies)
            if constraint.applies_if is not None:
                groups.append(constraint.applies_if.dependencies)
        read = [(item.path, item.kind) for group in groups for item in group]
        read.extend((item.path, item.kind) for _, item in self.references)
        return read

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    def parameter(self, name: str) -> KernelParameter | None:
        return next((item for item in self.parameters if item.name == name), None)

    def typed_handles(self, expected: type[Handles]) -> Handles:
        """Return author-declared choice handles without exposing the compiled spec."""

        if not isinstance(self.handles, expected):
            actual = type(self.handles).__name__
            raise TypeError(f"{self.id} declares {actual} handles, not {expected.__name__}")
        return self.handles


def check_declared_references(
    specification: DesignSpaceSpec,
    declarations: Sequence[CompiledKernelDeclaration],
) -> None:
    """Refuse Kernel references the assembled design space does not honour."""

    declared: dict[DependencyKind, dict[QualifiedPath, ValueSemantics[object]]] = {
        DependencyKind.PROBLEM: {
            item.path: item.value_semantics for item in specification.problem_schema.fields
        },
        DependencyKind.DECISION: {
            item.path: item.value_semantics for item in specification.decisions
        },
        DependencyKind.PROPERTY: {
            item.path: item.value_semantics for item in specification.properties
        },
    }
    issues: list[SpecAuthoringIssue] = []
    for declaration in declarations:
        for label, reference in declaration.references:
            where = f"{declaration.id}: {label}"
            matching = declared[reference.kind]
            if reference.path in matching:
                semantics = matching[reference.path]
                if not reference.semantics.is_compatible_with(semantics):
                    issues.append(
                        SpecAuthoringIssue(
                            "hardware-reference-wrong-type",
                            str(reference.path),
                            f"{where} reads it as {reference.semantics.name}, but it is "
                            f"declared {semantics.name}",
                        )
                    )
                continue
            elsewhere = tuple(
                kind.value for kind, paths in declared.items() if reference.path in paths
            )
            issues.append(
                SpecAuthoringIssue(
                    "hardware-reference-wrong-kind"
                    if elsewhere
                    else "hardware-reference-not-assembled",
                    str(reference.path),
                    f"{where} reads it as a {reference.kind.value}, but the design space "
                    f"declares it as {', '.join(elsewhere)}"
                    if elsewhere
                    else f"{where} names a path the design space does not declare",
                )
            )
    if issues:
        raise SpecAuthoringError(tuple(issues))


__all__ = ["CompiledKernelDeclaration", "check_declared_references"]
