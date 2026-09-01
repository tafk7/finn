# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped engine authoring for one ``Kernel`` subclass.

``KernelScope`` is the scope a hardware author is handed.  Like
``KernelDesign`` it owns no problem namespace -- a Kernel never reads a
``ModelWrapper``, an ONNX node, or a build configuration.  Everything it knows
about the outside arrives as typed handles the covered semantics wired in,
reachable as ``design.inputs``.

Unlike ``KernelDesign`` it declares no Region, no demand, and no export.  Those
are semantic, and a physical Kernel that derived one would be re-deciding the
logical dataflow it exists to implement.  What it declares instead is coverage,
local physical choices, derived physical parameters, coverage conditions, and a
source manifest.

The scope refuses two things outright, because both are the boundary this whole
layer exists to draw: reading a problem field the covered semantics did not
wire in, and declaring a physical parameter whose value has no declaration
behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Generic, TypeVar

from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Dependencies,
    DomainFactory,
    Ref,
    Scope,
    T,
)
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design import Answer, DesignSpaceSpec, EvaluatorSpec, ValueSemantics
from finn.dataflow.kernels._declaration import CompiledKernelDeclaration
from finn.dataflow.kernels.kernel import (
    CoveragePattern,
    EdgeCoverage,
    Kernel,
    KernelParameter,
    RegionCoverage,
    SourceFile,
)
from finn.dataflow.network import DataflowNetwork
from finn.dataflow.region import DataflowRegion
from finn.dataflow.spec_algebra import gate_spec

In = TypeVar("In")

#: The constraint-set name coverage conditions register into.  The declaration
#: reads it back rather than taking a hand-maintained path list.
COVERAGE = "coverage"


class KernelScope(Scope, Generic[In]):
    """One physical Kernel's authoring namespace over wired-in typed inputs."""

    def __init__(self, namespace: str, inputs: In) -> None:
        super().__init__(namespace)
        self.inputs = inputs
        self._regions: list[RegionCoverage] = []
        self._edges: list[EdgeCoverage] = []
        self._sealed = False
        self._parameters: list[KernelParameter] = []
        self._sources: list[SourceFile] = []

    # -- coverage ----------------------------------------------------------

    def covers_region(
        self,
        role: str,
        *,
        region: Ref[DataflowRegion],
        computation: Ref[ComputationContract],
        implements: ComputationContract,
        description: str = "",
    ) -> None:
        """Declare one Region this Kernel realizes, by naming its declaration.

        ``region`` and ``computation`` are handles the covered semantics wired
        in, so the Kernel is claiming a specific Region rather than a shape.
        ``implements`` is its own statement of what it computes, checked at
        binding against what that Region requires -- because equal traffic does
        not imply equal arithmetic.
        """

        if self._sealed:
            raise AuthoringError(f"{self.namespace} already sealed its coverage")
        if any(item.role == role for item in self._regions):
            raise AuthoringError(f"{self.namespace} covers Region role {role!r} twice")
        self._regions.append(RegionCoverage(role, region, computation, implements, description))

    def absorbs_edge(
        self,
        role: str,
        *,
        network: Ref[DataflowNetwork],
        source_role: str,
        sink_role: str,
        description: str = "",
    ) -> None:
        """Declare one connection this Kernel implements as internal wiring.

        Absorbing the edge is what makes a fused Kernel fused.  Naming the two
        roles it runs between lets binding check the selected Network really has
        that connection, in that direction -- an id alone would be a claim.
        """

        if self._sealed:
            raise AuthoringError(f"{self.namespace} already sealed its coverage")
        if any(item.role == role for item in self._edges):
            raise AuthoringError(f"{self.namespace} absorbs edge role {role!r} twice")
        self._edges.append(EdgeCoverage(role, network, source_role, sink_role, description))

    def coverage_constraint(
        self,
        name: str,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> ConstraintRef:
        """A condition on whether this Kernel can realize the point as configured.

        Every condition a physical Kernel owns is one of these.  There is no
        source-admission tier here: whether the *source* can be expressed at all
        was settled upstream, by the Region alternative this Kernel covers.
        """

        return self.constraint(
            name,
            dependencies=dependencies,
            evaluate=evaluate,
            applies_if=applies_if,
            sets=(COVERAGE,),
        )

    # -- physical values ---------------------------------------------------

    def choice(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: DomainFactory,
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[T]:
        """Declare one physical choice this Kernel owns locally.

        Pumping, pipeline depth, memory primitive: things that change the
        hardware without changing what the hardware computes.  A choice that
        would move a beat is a Region question and does not belong here.
        """

        return self.decision(name, value_type, domain=domain, applies_if=applies_if)

    def parameter(self, name: str, source: Ref[object]) -> KernelParameter:
        """Declare one HDL parameter, driven from an existing declaration.

        Where the value comes from is read off the handle, not restated: a
        problem field, a committed decision, or a derived property.  That is
        what makes the table an audit rather than a second set of formulas.
        """

        return self._remember_parameter(KernelParameter(name, source))

    def constant(self, name: str, value: object, *, why: str) -> KernelParameter:
        """Declare one HDL parameter this Kernel fixes, and say why it may.

        A constant is the one value with no declaration behind it, so it is the
        one that has to argue for itself.  ``why`` is required for that reason.
        """

        if not why:
            raise AuthoringError(f"{self.namespace}.{name} is a constant and must say why")
        return self._remember_parameter(KernelParameter(name, None, value, why))

    def _remember_parameter(self, parameter: KernelParameter) -> KernelParameter:
        if any(item.name == parameter.name for item in self._parameters):
            raise AuthoringError(f"{self.namespace} declares parameter {parameter.name!r} twice")
        self._parameters.append(parameter)
        return parameter

    # -- sources -----------------------------------------------------------

    def source(self, root: str, *paths: str) -> None:
        """Declare HDL this Kernel compiles, in compile order, under one root.

        The root is a name -- ``finn``, ``finnlib`` -- not a location.  Where
        that root resolves to is a property of the checkout, which a Kernel has
        no business knowing.
        """

        for path in paths:
            self._sources.append(SourceFile(root, path))

    # -- output ------------------------------------------------------------

    @property
    def declared_coverage(self) -> CoveragePattern:
        if not self._regions:
            raise AuthoringError(f"{self.namespace} covers no Region")
        self._sealed = True
        return CoveragePattern(tuple(self._regions), tuple(self._edges))

    @property
    def declared_parameters(self) -> tuple[KernelParameter, ...]:
        return tuple(self._parameters)

    @property
    def declared_sources(self) -> tuple[SourceFile, ...]:
        return tuple(self._sources)

    def spec(self) -> DesignSpaceSpec:
        """The Kernel's declarations, without its role set.

        ``COVERAGE`` is how this scope records which constraints are coverage
        conditions; the declaration turns that into a real constraint set under
        a name the assembly owns.  Leaving it here would collide the moment two
        Kernels were assembled together.
        """

        declared = super().spec()
        return replace(
            declared,
            constraint_sets=tuple(
                item for item in declared.constraint_sets if item.name != COVERAGE
            ),
        )


def declare_kernel(
    kernel: type[Kernel],
    namespace: str,
    inputs: object,
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> tuple[CompiledKernelDeclaration, KernelScope[object]]:
    """Run one ``Kernel`` subclass's ``define_design`` under a namespace.

    Returns the scope alongside the declarations, so an assembly that must wire
    one Kernel's choice into another can reach the handle rather than rebuild
    its path.

    ``applies_if`` gates every declaration the Kernel makes.  A Kernel that
    covers semantics this point did not select should contribute nothing to the
    design space -- not an unresolved decision, and not a coverage constraint
    asking about hardware nobody wants.
    """

    if not kernel.id:
        raise AuthoringError(f"{kernel.__name__} must set a Kernel id")
    design: KernelScope[object] = KernelScope(namespace, inputs)
    handles = kernel.define_design(design)
    declared = design.spec()
    spec = gate_spec(declared, applies_if) if applies_if is not None else declared
    return CompiledKernelDeclaration(
        kernel.id,
        kernel.version,
        namespace,
        spec,
        design.declared_coverage,
        design.declared_parameters,
        tuple(item.path for item in design.constraints_in(COVERAGE)),
        design.declared_sources,
        kernel,
        handles,
        design.decision_handles,
        design.constraint_handles,
    ), design


def kernel_namespace(owner: str, kernel_id: str) -> str:
    """The namespace one physical Kernel owns inside one assembly."""

    if not owner or not kernel_id:
        raise AuthoringError("a hardware namespace needs both an owner and a Kernel id")
    return f"{owner}.{kernel_id}"


__all__ = [
    "COVERAGE",
    "KernelScope",
    "declare_kernel",
    "kernel_namespace",
]
