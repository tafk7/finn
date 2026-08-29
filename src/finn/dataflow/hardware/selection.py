# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Choosing between physical Kernels that cover the same semantics.

A selection exists only when there is a real choice to make: several Kernels
cover one design point and a policy, a measurement, or a target requirement has
to pick.  When exactly one Kernel covers a point, there is no decision and none
is invented -- ``bind_hardware_kernel`` is called directly and the binding is
derived.  A gratuitous decision would make the design space claim a degree of
freedom that does not exist.

Selecting a physical Kernel never changes the Region or Network it covers.
That is the property the whole layer rests on, and it is what the fifth forcing
case checks: two alternatives, one committed choice, and equal semantics either
way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from finn.dataflow.design import (
    Answer,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DesignPoint,
    DesignSpaceSpec,
    Engine,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.hardware.kernel import (
    BoundRegion,
    HardwareKernelDeclaration,
    KernelBinding,
    bind_hardware_kernel,
)
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
    duplicate_values,
    gate_spec,
)

HARDWARE_KERNEL_ID_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(str, name="HardwareKernelId")
)


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


@dataclass(frozen=True)
class HardwareKernelSelection:
    """One pool of physical Kernels covering equal semantics.

    Every member must cover the same roles.  A pool whose members covered
    different shapes would not be a choice between implementations of one
    thing; it would be two different bindings wearing one decision.
    """

    name: str
    kernels: tuple[HardwareKernelDeclaration, ...]
    applies_if: EvaluatorSpec[Answer[bool]] | None = None

    def __post_init__(self) -> None:
        issues: list[SpecAuthoringIssue] = []
        if not self.name:
            issues.append(
                SpecAuthoringIssue(
                    "hardware-selection-name-empty", "selection", "selection name is empty"
                )
            )
        if not self.kernels:
            issues.append(
                SpecAuthoringIssue(
                    "hardware-selection-pool-empty",
                    self.name,
                    "a physical Kernel selection must declare at least one candidate",
                )
            )
        for duplicate in duplicate_values(tuple(item.id for item in self.kernels)):
            issues.append(
                SpecAuthoringIssue(
                    "hardware-kernel-id-duplicate",
                    f"{self.name}.{duplicate}",
                    f"Kernel id {duplicate!r} is duplicated in this pool",
                )
            )
        if self.kernels:
            shapes = {
                (item.coverage.region_roles, item.coverage.edge_roles) for item in self.kernels
            }
            if len(shapes) > 1:
                issues.append(
                    SpecAuthoringIssue(
                        "hardware-selection-coverage-differs",
                        self.name,
                        "pool members must cover the same Region and edge roles",
                    )
                )
        if issues:
            raise SpecAuthoringError(tuple(issues))

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.kernels)

    @property
    def kernel_path(self) -> QualifiedPath:
        return QualifiedPath(f"{self.name}.hardware_kernel")

    @property
    def coverage_constraints(self) -> tuple[QualifiedPath, ...]:
        return tuple(
            dict.fromkeys(path for item in self.kernels for path in item.coverage_constraints)
        )

    def kernel(self, kernel_id: str) -> HardwareKernelDeclaration:
        for candidate in self.kernels:
            if candidate.id == kernel_id:
                return candidate
        raise KeyError(f"{kernel_id!r} is not a member of the {self.name!r} hardware pool")

    def kernel_applies(self, kernel_id: str) -> EvaluatorSpec[Answer[bool]]:
        """The applicability of declarations owned by one pool member."""

        reference = DependencyRef.decision(
            f"{self.name}.selected", self.kernel_path, HARDWARE_KERNEL_ID_SEMANTICS
        )
        outer = self.applies_if

        def evaluate(values: DependencyView) -> Answer[bool]:
            if outer is not None:
                answer = outer.evaluator(
                    DependencyView({item.name: values[item.name] for item in outer.dependencies})
                )
                if not isinstance(answer, Decided) or not answer.value:
                    return answer
            return Decided(values[reference.name] == kernel_id)

        dependencies = (*outer.dependencies, reference) if outer else (reference,)
        return EvaluatorSpec(dependencies, evaluate)

    def build_spec(self) -> DesignSpaceSpec:
        """Assemble this pool into one ordinary flat specification."""

        gated = tuple(gate_spec(item.spec, self.kernel_applies(item.id)) for item in self.kernels)
        own = DesignSpaceSpec(
            decisions=(
                Decision(
                    self.kernel_path,
                    HARDWARE_KERNEL_ID_SEMANTICS,
                    _finite_domain(cast("tuple[object, ...]", self.kernel_ids)),
                    applies_if=self.applies_if,
                ),
            ),
        )
        return assemble_specs((*gated, own))

    def selected(self, engine: Engine, point: DesignPoint) -> Answer[str]:
        """The committed physical Kernel id, if one is assigned."""

        if self.kernel_path not in point.assignments:
            return Unresolved(
                (
                    Finding(
                        FindingKind.LIMITATION,
                        "hardware-kernel-unselected",
                        self.kernel_path,
                        f"no physical Kernel is committed for {self.name}",
                    ),
                )
            )
        return Decided(cast(str, point.assignments[self.kernel_path]))

    def bind(
        self,
        engine: Engine,
        point: DesignPoint,
        regions: Mapping[str, BoundRegion],
        edges: Mapping[str, str] | None = None,
    ) -> Answer[KernelBinding]:
        """Bind whichever member this point committed to."""

        chosen = self.selected(engine, point)
        if not isinstance(chosen, Decided):
            return Unresolved(chosen.findings)
        return bind_hardware_kernel(engine, self.kernel(chosen.value), point, regions, edges)


__all__ = [
    "HARDWARE_KERNEL_ID_SEMANTICS",
    "HardwareKernelSelection",
]
