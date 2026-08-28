# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped engine authoring for one ``Kernel`` subclass.

``KernelDesign`` is the scope a Kernel author is handed.  Unlike ``OpDesign``
it owns no problem namespace: a Kernel never reads a ``ModelWrapper``, an ONNX
node, or a build configuration.  Everything it knows about the outside arrives
as typed handles its owning operation wired in, reachable as ``design.inputs``.

The scope also carries the two things a bare ``Scope`` cannot express about a
Kernel: which of its properties is *the* Region, and which of its constraints
are source-admission questions rather than feasibility ones.  Both are declared,
not inferred -- what a constraint reads is a different question from whose
condition it is -- but a declared source constraint is checked against what it
reads, because a source question that consults the target or a local decision
is not a source question.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Generic, TypeVar

from finn.dataflow.authoring.op_design import BUILD_OWNED, ProblemProvenance
from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Dependencies,
    DomainFactory,
    Ref,
    Scope,
    T,
)
from finn.dataflow.region import DataflowRegion, Port
from finn.dataflow.kernels import (
    Kernel,
    KernelDeclaration,
    KernelDemand,
    KernelExport,
    KernelProvider,
)
from finn.dataflow.design import (
    Answer,
    DATAFLOW_REGION_SEMANTICS,
    DependencyKind,
    DesignSpaceSpec,
    EvaluatorSpec,
    QualifiedPath,
    ValueSemantics,
)

In = TypeVar("In")

#: Constraint-set names a Kernel registers into.  ``KernelSelection`` reads
#: them back rather than taking two hand-maintained path lists.
SOURCE_ADMISSION = "source_admission"
FEASIBILITY = "feasibility"


class KernelDesign(Scope, Generic[In]):
    """One Kernel's authoring namespace over wired-in typed inputs."""

    def __init__(
        self,
        namespace: str,
        inputs: In,
        *,
        provenance: ProblemProvenance | None = None,
    ) -> None:
        super().__init__(namespace)
        self.inputs = inputs
        self._provenance = provenance
        self._region: QualifiedPath | None = None
        self._demands: list[tuple[str, QualifiedPath]] = []
        self._exports: list[tuple[str, QualifiedPath, ValueSemantics[object]]] = []
        self._providers: list[tuple[str, str]] = []

    # -- the Region --------------------------------------------------------

    def region(
        self,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        name: str = "region",
    ) -> Ref[DataflowRegion]:
        """Declare the one complete Region this Kernel derives."""

        if self._region is not None:
            raise AuthoringError(f"{self.namespace} already declares a Region at {self._region}")
        handle = self.derived(
            name, DATAFLOW_REGION_SEMANTICS, dependencies=dependencies, evaluate=evaluate
        )
        self._region = handle.path
        return handle

    # -- boundary --------------------------------------------------------

    def demand(
        self,
        interface: str,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        name: str | None = None,
    ) -> Ref[Port]:
        """Declare one parameter interface this Kernel needs supplied."""

        if any(item[0] == interface for item in self._demands):
            raise AuthoringError(f"{self.namespace} demands {interface!r} twice")
        handle = self.derived(
            name or f"{interface}_port", Port, dependencies=dependencies, evaluate=evaluate
        )
        self._demands.append((interface, handle.path))
        return handle

    def export(self, name: str, value: Ref[object]) -> None:
        """Present an already-declared property under an assembly-facing name."""

        if any(item[0] == name for item in self._exports):
            raise AuthoringError(f"{self.namespace} exports {name!r} twice")
        if value.kind is not DependencyKind.PROPERTY:
            raise AuthoringError(
                f"{self.namespace} cannot export {value.path}: an export presents one path for a "
                f"value every pool member derives its own way, so it must be a derived property"
            )
        self._exports.append((name, value.path, value.semantics))

    def provider(self, provider_id: str, *, version: str = "1") -> None:
        """Record one mechanism that can realize this Kernel."""

        self._providers.append((provider_id, version))

    # -- constraints -------------------------------------------------------

    def feasibility_constraint(
        self,
        name: str,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> ConstraintRef:
        """A condition on whether this Kernel can be built as configured."""

        return self.constraint(
            name,
            dependencies=dependencies,
            evaluate=evaluate,
            applies_if=applies_if,
            sets=(FEASIBILITY,),
        )

    def source_constraint(
        self,
        name: str,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> ConstraintRef:
        """A condition on whether this Kernel may serve the source at all.

        Registered into both sets: a Kernel the source rules out is not
        feasible either, and stating that once removes the duplicate path
        lists the two sets used to be maintained as.
        """

        self._check_source_dependencies(name, dependencies, applies_if)
        return self.constraint(
            name,
            dependencies=dependencies,
            evaluate=evaluate,
            applies_if=applies_if,
            sets=(SOURCE_ADMISSION, FEASIBILITY),
        )

    def _check_source_dependencies(
        self,
        name: str,
        dependencies: Dependencies,
        applies_if: EvaluatorSpec[Answer[bool]] | None,
    ) -> None:
        """Source admission is asked before anything is decided or derived.

        A constraint that reads a decision, a derived property, or a target or
        build fact cannot be answered then, so declaring it a source question
        is an authoring error rather than a value the engine will later find
        unresolved.
        """

        offenders: list[str] = []
        paths: list[QualifiedPath] = []
        for ref in dependencies.values():
            if ref.kind is not DependencyKind.PROBLEM:
                offenders.append(f"{ref.path} is a {ref.kind.value}")
            else:
                paths.append(ref.path)
        if applies_if is not None:
            for item in applies_if.dependencies:
                if item.kind is not DependencyKind.PROBLEM:
                    offenders.append(f"{item.path} is a {item.kind.value}")
                else:
                    paths.append(item.path)
        if self._provenance is not None:
            for path in paths:
                kind = self._provenance.kind_of(path)
                if kind is not None and kind in BUILD_OWNED:
                    offenders.append(f"{path} is {kind.value}")
        if offenders:
            raise AuthoringError(
                f"{self.constraint_path(name)} is declared a source constraint but reads "
                f"{sorted(set(offenders))}; source admission is answered from graph facts alone"
            )

    # -- decisions ---------------------------------------------------------

    def choice(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: DomainFactory,
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[T]:
        """Declare one choice this Kernel owns locally.

        An alias for ``Scope.decision`` that reads correctly at a Kernel: the
        Kernel is not deciding *which* Kernel, it is configuring itself.
        """

        return self.decision(name, value_type, domain=domain, applies_if=applies_if)

    # -- output ------------------------------------------------------------

    @property
    def region_path(self) -> QualifiedPath:
        if self._region is None:
            raise AuthoringError(f"{self.namespace} declares no Region")
        return self._region

    @property
    def declared_demands(self) -> tuple[tuple[str, QualifiedPath], ...]:
        return tuple(self._demands)

    @property
    def declared_exports(self) -> tuple[tuple[str, QualifiedPath, ValueSemantics[object]], ...]:
        return tuple(self._exports)

    @property
    def declared_providers(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._providers)

    def constraint_paths(self, group: str) -> tuple[QualifiedPath, ...]:
        return tuple(item.path for item in self.constraints_in(group))

    def spec(self) -> DesignSpaceSpec:
        """The Kernel's declarations, without its two role sets.

        ``SOURCE_ADMISSION`` and ``FEASIBILITY`` are how this scope records
        which role each constraint plays; the pool is what turns those roles
        into real constraint sets, under names it owns.  Leaving them in the
        Kernel's own spec would collide the moment two pool members were
        assembled together.
        """

        declared = super().spec()
        return replace(
            declared,
            constraint_sets=tuple(
                item
                for item in declared.constraint_sets
                if item.name not in (SOURCE_ADMISSION, FEASIBILITY)
            ),
        )


def declare_kernel(
    kernel: type[Kernel],
    namespace: str,
    inputs: object,
    *,
    provenance: ProblemProvenance | None = None,
) -> KernelDeclaration:
    """Declare one Kernel and keep only its declarations."""

    return declare_kernel_design(kernel, namespace, inputs, provenance=provenance)[0]


def declare_kernel_design(
    kernel: type[Kernel],
    namespace: str,
    inputs: object,
    *,
    provenance: ProblemProvenance | None = None,
) -> tuple[KernelDeclaration, KernelDesign[object]]:
    """Run one Kernel subclass's ``define_design`` under ``namespace``.

    Returns the scope alongside the declarations, so an operation that must
    wire one Kernel's choice into another -- a downstream Kernel's folding
    determining what an upstream one has to present -- can reach the handle
    rather than rebuild its path.

    The operation that owns the pool calls this, because the operation is what
    knows the namespace and the wired inputs.  Placing the same subclass twice
    under different namespaces yields two non-colliding declaration sets over
    one authored design.
    """

    if not kernel.id:
        raise AuthoringError(f"{kernel.__name__} must set a Kernel id")
    design: KernelDesign[object] = KernelDesign(namespace, inputs, provenance=provenance)
    kernel.define_design(design)
    return KernelDeclaration(
        kernel.id,
        kernel.version,
        design.spec(),
        design.region_path,
        design.constraint_paths(FEASIBILITY),
        design.constraint_paths(SOURCE_ADMISSION),
        tuple(KernelDemand(name, path) for name, path in design.declared_demands),
        tuple(
            KernelExport(name, path, semantics) for name, path, semantics in design.declared_exports
        ),
        tuple(
            KernelProvider(provider_id, kernel.id, version)
            for provider_id, version in design.declared_providers
        ),
        kernel,
    ), design


def kernel_namespace(pool: str, kernel_id: str) -> str:
    """The namespace one Kernel owns inside one pool."""

    if not pool or not kernel_id:
        raise AuthoringError("a Kernel namespace needs both a pool and a Kernel id")
    return f"{pool}.{kernel_id}"


__all__ = [
    "FEASIBILITY",
    "SOURCE_ADMISSION",
    "KernelDesign",
    "declare_kernel",
    "declare_kernel_design",
    "kernel_namespace",
]
