# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Private lowering support for one physical ``Kernel`` subclass.

``KernelScope`` binds immutable class declarations. Legacy tests also exercise
its callback bridge directly. Like the removed public ``KernelDesign`` API it
owns no problem namespace -- a Kernel never reads a
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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Generic, TypeVar, cast

from finn.dataflow.authoring.scope import (
    AuthoringError,
    ConstraintRef,
    Dependencies,
    DomainFactory,
    Ref,
    Scope,
    T,
)
from finn.dataflow.authoring.declarations import (
    DeclarationGroup,
    DeclarationLayer,
    DeclarationTemplate,
    compile_class_declarations,
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


@dataclass(frozen=True, slots=True)
class KernelInput:
    """Reference to one non-engine value in a Kernel's typed input bundle."""

    name: str


@dataclass(frozen=True, slots=True)
class RegionClaim:
    role: str | KernelInput
    region: DeclarationTemplate[Any]
    computation: DeclarationTemplate[Any]
    implements: ComputationContract
    description: str = ""


@dataclass(frozen=True, slots=True)
class Covers(DeclarationGroup):
    regions: tuple[RegionClaim, ...]
    layers = frozenset({DeclarationLayer.KERNEL})

    def __init__(self, *regions: RegionClaim) -> None:
        if not regions:
            raise ValueError("Kernel coverage requires at least one Region")
        object.__setattr__(self, "regions", tuple(regions))

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class EdgeClaim(DeclarationGroup):
    role: str
    network: DeclarationTemplate[Any]
    source_role: str
    sink_role: str
    description: str = ""
    layers = frozenset({DeclarationLayer.KERNEL})

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class Parameter(DeclarationGroup):
    name: str
    source: DeclarationTemplate[Any]
    layers = frozenset({DeclarationLayer.KERNEL})

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class Constant(DeclarationGroup):
    name: str
    value: object
    why: str
    layers = frozenset({DeclarationLayer.KERNEL})

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class Sources(DeclarationGroup):
    root: str
    paths: tuple[str, ...]
    layers = frozenset({DeclarationLayer.KERNEL})

    def __init__(self, root: str, *paths: str) -> None:
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "paths", tuple(paths))

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


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


def _kernel_input_refs(inputs: object) -> Mapping[str, Ref[object]]:
    if isinstance(inputs, Mapping):
        values = dict(inputs)
    elif is_dataclass(inputs) and not isinstance(inputs, type):
        values = {item.name: getattr(inputs, item.name) for item in fields(inputs)}
    else:
        raise AuthoringError("class-authored Kernel inputs must be a mapping or dataclass")
    return cast(
        "Mapping[str, Ref[object]]",
        {name: value for name, value in values.items() if isinstance(value, Ref)},
    )


def _kernel_input_values(inputs: object) -> Mapping[str, object]:
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if is_dataclass(inputs) and not isinstance(inputs, type):
        return {item.name: getattr(inputs, item.name) for item in fields(inputs)}
    raise AuthoringError("class-authored Kernel inputs must be a mapping or dataclass")


def _kernel_ref(template: DeclarationTemplate[Any], compiled: object) -> Ref[object]:
    declarations = cast(Any, compiled)
    member = declarations.template_members.get(id(template))
    if member is None:
        raise AuthoringError("a Kernel declaration references an undeclared class member")
    return cast("Ref[object]", declarations.ref(member))


def compile_kernel_class(
    kernel: type[Kernel],
    namespace: str,
    inputs: object,
    *,
    applies_if: EvaluatorSpec[Answer[bool]] | None = None,
) -> tuple[CompiledKernelDeclaration, KernelScope[object]]:
    """Lower one direct Kernel class through a private ``KernelScope``."""

    scope: KernelScope[object] = KernelScope(namespace, inputs)
    input_values = _kernel_input_values(inputs)
    compiled = compile_class_declarations(
        kernel,
        layer=DeclarationLayer.KERNEL,
        namespace=namespace,
        imports=_kernel_input_refs(inputs),
        scope=scope,
    )
    for group in compiled.groups.values():
        if isinstance(group, Covers):
            for claim in group.regions:
                scope.covers_region(
                    cast(str, input_values[claim.role.name])
                    if isinstance(claim.role, KernelInput)
                    else claim.role,
                    region=cast("Ref[DataflowRegion]", _kernel_ref(claim.region, compiled)),
                    computation=cast(
                        "Ref[ComputationContract]", _kernel_ref(claim.computation, compiled)
                    ),
                    implements=claim.implements,
                    description=claim.description,
                )
        elif isinstance(group, EdgeClaim):
            scope.absorbs_edge(
                group.role,
                network=cast("Ref[DataflowNetwork]", _kernel_ref(group.network, compiled)),
                source_role=group.source_role,
                sink_role=group.sink_role,
                description=group.description,
            )
        elif isinstance(group, Parameter):
            scope.parameter(group.name, _kernel_ref(group.source, compiled))
        elif isinstance(group, Constant):
            scope.constant(group.name, group.value, why=group.why)
        elif isinstance(group, Sources):
            scope.source(group.root, *group.paths)

    declared = scope.spec()
    spec = gate_spec(declared, applies_if) if applies_if is not None else declared
    handle_factory = getattr(kernel, "compiled_handles", None)
    handles = handle_factory(compiled.members) if callable(handle_factory) else compiled.members
    return CompiledKernelDeclaration(
        kernel.id,
        kernel.version,
        namespace,
        spec,
        scope.declared_coverage,
        scope.declared_parameters,
        tuple(item.path for item in scope.constraints_in(COVERAGE)),
        scope.declared_sources,
        kernel,
        handles,
        scope.decision_handles,
        scope.constraint_handles,
    ), scope


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
    if kernel.uses_class_authoring:
        return compile_kernel_class(kernel, namespace, inputs, applies_if=applies_if)
    design: KernelScope[object] = KernelScope(namespace, inputs)
    define = getattr(kernel, "define_design", None)
    if not callable(define):
        raise AuthoringError(f"{kernel.__name__} has no class-local declarations")
    handles = define(design)
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
    "Constant",
    "Covers",
    "EdgeClaim",
    "KernelScope",
    "KernelInput",
    "Parameter",
    "RegionClaim",
    "Sources",
    "compile_kernel_class",
    "declare_kernel",
    "kernel_namespace",
]
