# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Authoring for one parametric :class:`DataflowDesign`.

The canonical Region and Network objects deliberately know nothing about the
design-space engine or physical Kernels. This module composes those existing
values into the compiler-facing hierarchy:

``DataflowDesign -> one flat DataflowNetwork -> named Kernel placements``.

Everything declared here is still an ordinary engine declaration. A design
choice, a folding choice, a conditional supplier, and a Kernel-local choice
therefore remain separate coordinates in one flat ``DesignSpaceSpec``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from finn.dataflow.authoring.scope import (
    AuthoringError,
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
    Derived,
    compile_class_declarations,
)
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design import (
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    Answer,
    DependencyKind,
    DesignSpaceSpec,
    EvaluatorSpec,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels.authoring import declare_kernel
from finn.dataflow.kernels._declaration import CompiledKernelDeclaration
from finn.dataflow.kernels.kernel import Kernel
from finn.dataflow.kernels.selection import (
    KERNEL_ID_SEMANTICS,
    KernelCandidateSelection,
)
from finn.dataflow.network import BoundaryContract, DataflowNetwork, NetworkNode, RegionEndpoint
from finn.dataflow.region import DataflowRegion, InputInterface
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
)

In = TypeVar("In")

if TYPE_CHECKING:
    from finn.dataflow.authoring.composition import PhysicalCompositionContext

_COMPUTATION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(ComputationContract, name="ComputationContract")
)
_INPUT_INTERFACE_SEMANTICS = ValueSemantics.immutable_nominal(InputInterface, name="InputInterface")


@dataclass(frozen=True)
class DesignNode:
    """One stable Network role and the semantic declarations behind it."""

    role: str
    node_id: str
    region: Ref[DataflowRegion]
    computation: Ref[ComputationContract]

    def __post_init__(self) -> None:
        if not self.role or not self.node_id:
            raise ValueError("a design node needs a role and a Network node id")
        for name, handle in (("region", self.region), ("computation", self.computation)):
            if handle.kind is not DependencyKind.PROPERTY:
                raise ValueError(f"a design node {name} must be a derived property")


@dataclass(frozen=True)
class DesignEdge:
    """One stable edge role used by a placement that absorbs the edge."""

    role: str
    edge_id: str
    network: Ref[DataflowNetwork]
    source_role: str
    sink_role: str

    def __post_init__(self) -> None:
        if not all((self.role, self.edge_id, self.source_role, self.sink_role)):
            raise ValueError("a design edge needs a role, id, source role, and sink role")
        if self.network.kind is not DependencyKind.PROPERTY:
            raise ValueError("a design edge must name a derived Network property")


@dataclass(frozen=True)
class DesignInput:
    """One logical boundary mapped to an Operation source operand."""

    source_operand: str
    boundary_id: str
    consumer: Ref[InputInterface]

    def __post_init__(self) -> None:
        if not self.source_operand or not self.boundary_id:
            raise ValueError("a design input needs a source operand and boundary id")
        if self.consumer.kind is not DependencyKind.PROPERTY:
            raise ValueError("a design input consumer must be a derived InputInterface")


@dataclass(frozen=True)
class PlacementSelection:
    """The active Kernel identity for a placement, or no physical candidate."""

    kernel_id: str | None


PLACEMENT_SELECTION_SEMANTICS = ValueSemantics.immutable_nominal(
    PlacementSelection, name="PlacementSelection"
)


def _selected_placement(kernel_id: str) -> PlacementSelection:
    return PlacementSelection(kernel_id)


def _fixed_placement(kernel_id: str | None) -> Callable[[], object]:
    def selected() -> object:
        return PlacementSelection(kernel_id)

    return selected


@dataclass(frozen=True)
class KernelPlacement:
    """Compiled metadata for one named physical placement."""

    name: str
    nodes: tuple[DesignNode, ...]
    edges: tuple[DesignEdge, ...]
    candidates: tuple[CompiledKernelDeclaration, ...]
    selected_kernel: Ref[PlacementSelection]
    kernel_choice: Ref[str] | None = field(default=None, repr=False, compare=False)

    def candidate(self, kernel_id: str) -> CompiledKernelDeclaration:
        for candidate in self.candidates:
            if candidate.id == kernel_id:
                return candidate
        raise KeyError(f"{kernel_id!r} is not a candidate for placement {self.name!r}")

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        return tuple(candidate.id for candidate in self.candidates)


class DataflowDesign:
    """One contributor-authored logical Network and Kernel partition."""

    id: str = ""
    version: str = "1"
    uses_class_authoring: bool = False


@dataclass(frozen=True, slots=True)
class DesignPortRef:
    """Unbound reference to one port of a class-declared Region."""

    region: Region
    port_id: str
    direction: str


@dataclass(frozen=True, slots=True, eq=False)
class Region(DeclarationGroup):
    """One class-local Region declaration and its computation requirement."""

    node_id: str
    role: str
    construct: Callable[..., DataflowRegion]
    dependencies: tuple[DeclarationTemplate[Any], ...]
    computation_contract: ComputationContract
    region: Derived[DataflowRegion]
    computation: Derived[ComputationContract]
    layers = frozenset({DeclarationLayer.DESIGN})

    def __init__(
        self,
        *,
        node_id: str,
        construct: Callable[..., DataflowRegion],
        dependencies: Sequence[DeclarationTemplate[Any]],
        computation: ComputationContract,
        role: str | None = None,
    ) -> None:
        if not node_id:
            raise ValueError("a Region declaration needs a node id")
        resolved_role = node_id if role is None else role
        if not resolved_role:
            raise ValueError("a Region declaration needs a role")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "role", resolved_role)
        object.__setattr__(self, "construct", construct)
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "computation_contract", computation)
        object.__setattr__(
            self,
            "region",
            Derived(
                DATAFLOW_REGION_SEMANTICS,
                dependencies,
                construct,
                stable_name=f"{resolved_role}.region",
            ),
        )
        object.__setattr__(
            self,
            "computation",
            Derived(
                ComputationContract,
                (),
                lambda: computation,
                stable_name=f"{resolved_role}.computation",
            ),
        )

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        return (
            (f"{member_name}.region", self.region),
            (f"{member_name}.computation", self.computation),
        )

    def input(self, port_id: str) -> DesignPortRef:
        return DesignPortRef(self, port_id, "input")

    def output(self, port_id: str) -> DesignPortRef:
        return DesignPortRef(self, port_id, "output")


@dataclass(frozen=True, slots=True, eq=False)
class Network(Derived[DataflowNetwork]):
    """The one flat Network property owned by a Design class."""

    nodes: tuple[Region, ...] = ()

    def __init__(
        self,
        *nodes: Region,
        construct: Callable[..., DataflowNetwork] | None = None,
        stable_name: str = "network",
    ) -> None:
        if not nodes:
            raise ValueError("a Network declaration needs at least one Region")
        node_tuple = tuple(nodes)
        if construct is None:
            if len(node_tuple) != 1:
                raise ValueError("only a one-Region Network can omit a constructor")
            only = node_tuple[0]

            def singleton(region: object) -> object:
                return singleton_network(only.node_id, cast(DataflowRegion, region))

            evaluate: Callable[..., object] = singleton
        else:
            evaluate = construct
        Derived.__init__(
            self,
            DATAFLOW_NETWORK_SEMANTICS,
            tuple(cast("DeclarationTemplate[Any]", item.region) for item in node_tuple),
            evaluate,
            stable_name=stable_name,
            layers=(DeclarationLayer.DESIGN,),
        )
        object.__setattr__(self, "nodes", node_tuple)


@dataclass(frozen=True, slots=True)
class Connection(DeclarationGroup):
    """Named Network edge used when a placement absorbs the connection."""

    role: str
    edge_id: str
    source: DesignPortRef
    sink: DesignPortRef
    layers = frozenset({DeclarationLayer.DESIGN})

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class SourceInput(DeclarationGroup):
    source_operand: str
    destination: DesignPortRef
    boundary_id: str
    coordinates: object | None = None
    supply_eligible: bool = False
    layers = frozenset({DeclarationLayer.DESIGN})

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class Kernels(DeclarationGroup):
    name: str | None
    covers: tuple[Region, ...]
    candidates: tuple[type[Kernel], ...]
    inputs: object
    absorbs: tuple[Connection, ...] = ()
    layers = frozenset({DeclarationLayer.DESIGN})

    def __init__(
        self,
        *,
        covers: Sequence[Region],
        candidates: Sequence[type[Kernel]],
        inputs: object,
        absorbs: Sequence[Connection] = (),
        name: str | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "covers", tuple(covers))
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "absorbs", tuple(absorbs))

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


@dataclass(frozen=True, slots=True)
class PhysicalComposition(DeclarationGroup):
    composer: Callable[[PhysicalCompositionContext], object] | str
    inputs: Mapping[str, DeclarationTemplate[Any]]
    layers = frozenset({DeclarationLayer.DESIGN})

    def __init__(
        self,
        composer: Callable[[PhysicalCompositionContext], object] | str,
        *,
        inputs: Mapping[str, DeclarationTemplate[Any]] = MappingProxyType({}),
    ) -> None:
        if not callable(composer) and (not isinstance(composer, str) or ":" not in composer):
            raise ValueError("a composer must be callable or a 'module:name' reference")
        object.__setattr__(self, "composer", composer)
        object.__setattr__(self, "inputs", MappingProxyType(dict(inputs)))

    def declaration_items(
        self, member_name: str
    ) -> tuple[tuple[str, DeclarationTemplate[Any]], ...]:
        del member_name
        return ()


class DataflowDesignScope(Scope, Generic[In]):
    """Private lowering scope for one ``DataflowDesign`` class."""

    def __init__(self, namespace: str, inputs: In) -> None:
        super().__init__(namespace)
        self.inputs = inputs
        self._network: Ref[DataflowNetwork] | None = None
        self._nodes: dict[str, DesignNode] = {}
        self._placements: list[KernelPlacement] = []
        self._inputs: dict[str, DesignInput] = {}
        self._physical_specs: list[DesignSpaceSpec] = []
        self._hardware: list[CompiledKernelDeclaration] = []

    def choice(
        self,
        name: str,
        value_type: type[T] | ValueSemantics[T],
        *,
        domain: DomainFactory,
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[T]:
        return self.decision(name, value_type, domain=domain, applies_if=applies_if)

    def region(
        self,
        name: str,
        *,
        node_id: str,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        computation: ComputationContract | Ref[ComputationContract],
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> DesignNode:
        """Declare and register one Region node under a stable design role."""

        region = self.derived(
            f"{name}.region",
            DATAFLOW_REGION_SEMANTICS,
            dependencies=dependencies,
            evaluate=evaluate,
            applies_if=applies_if,
        )
        required: Ref[ComputationContract]
        if isinstance(computation, Ref):
            required = computation
        else:
            required = cast(
                "Ref[ComputationContract]",
                self.derived(
                    f"{name}.computation",
                    _COMPUTATION_SEMANTICS,
                    dependencies={},
                    evaluate=lambda: computation,
                    applies_if=applies_if,
                ),
            )
        return self.node(name, node_id=node_id, region=region, computation=required)

    def node(
        self,
        role: str,
        *,
        node_id: str,
        region: Ref[DataflowRegion],
        computation: Ref[ComputationContract],
    ) -> DesignNode:
        """Register shared semantic declarations as one role in this design."""

        if role in self._nodes:
            raise AuthoringError(f"{self.namespace} declares node role {role!r} twice")
        if any(item.node_id == node_id for item in self._nodes.values()):
            raise AuthoringError(f"{self.namespace} uses Network node id {node_id!r} twice")
        node = DesignNode(role, node_id, region, computation)
        self._nodes[role] = node
        return node

    def input_interface(
        self,
        name: str,
        node: DesignNode,
        port_id: str,
        *,
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[InputInterface]:
        """Declare the complete consumer contract for one mapped input."""

        return self.derived(
            name,
            _INPUT_INTERFACE_SEMANTICS,
            dependencies={"region": node.region},
            evaluate=lambda region: cast(DataflowRegion, region).input_interface(port_id),
            applies_if=applies_if,
        )

    def map_input(
        self,
        source_operand: str,
        *,
        boundary_id: str,
        consumer: Ref[InputInterface],
    ) -> DesignInput:
        """Map one Operation source operand onto this design's logical input."""

        if source_operand in self._inputs:
            raise AuthoringError(
                f"{self.namespace} maps source operand {source_operand!r} more than once"
            )
        mapping = DesignInput(source_operand, boundary_id, consumer)
        self._inputs[source_operand] = mapping
        return mapping

    def network(
        self,
        *,
        dependencies: Dependencies,
        evaluate: Callable[..., object],
        name: str = "network",
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> Ref[DataflowNetwork]:
        """Declare the one core flat Network this design exposes."""

        if self._network is not None:
            raise AuthoringError(f"{self.namespace} already declares a Network")
        network = self.derived(
            name,
            DATAFLOW_NETWORK_SEMANTICS,
            dependencies=dependencies,
            evaluate=evaluate,
            applies_if=applies_if,
        )
        self._network = network
        return network

    def use_network(self, network: Ref[DataflowNetwork]) -> Ref[DataflowNetwork]:
        """Expose a Network declaration shared with another design."""

        if self._network is not None:
            raise AuthoringError(f"{self.namespace} already declares a Network")
        if network.kind is not DependencyKind.PROPERTY:
            raise AuthoringError("a DataflowDesign Network must be a derived property")
        self._network = network
        return network

    def singleton_network(
        self,
        node: DesignNode,
        *,
        name: str = "network",
    ) -> Ref[DataflowNetwork]:
        """Lift one Region into a one-node Network with complete boundaries."""

        return self.network(
            name=name,
            dependencies={"region": node.region},
            evaluate=lambda region: singleton_network(node.node_id, cast(DataflowRegion, region)),
        )

    def edge(
        self,
        role: str,
        *,
        edge_id: str,
        source: DesignNode,
        sink: DesignNode,
        network: Ref[DataflowNetwork] | None = None,
    ) -> DesignEdge:
        """Name one Network edge for placement absorption."""

        return DesignEdge(
            role,
            edge_id,
            network or self.network_ref,
            source.role,
            sink.role,
        )

    def kernels(
        self,
        name: str,
        *,
        covers: Sequence[DesignNode],
        candidates: Sequence[type[Kernel]],
        inputs: object | None = None,
        absorbs: Sequence[DesignEdge] = (),
        applies_if: EvaluatorSpec[Answer[bool]] | None = None,
    ) -> KernelPlacement:
        """Declare one placement and its zero, one, or many Kernel candidates."""

        if not name:
            raise AuthoringError("a Kernel placement must be named")
        if any(item.name == name for item in self._placements):
            raise AuthoringError(f"{self.namespace} declares placement {name!r} twice")
        nodes = tuple(covers)
        edges = tuple(absorbs)
        if not nodes:
            raise AuthoringError(f"{self.namespace}.{name} covers no Region")
        if len({item.role for item in nodes}) != len(nodes):
            raise AuthoringError(f"{self.namespace}.{name} repeats a Region role")
        if len({item.role for item in edges}) != len(edges):
            raise AuthoringError(f"{self.namespace}.{name} repeats an edge role")

        candidate_types = tuple(candidates)
        declarations = tuple(
            declare_kernel(
                kernel,
                f"{self.namespace}.{name}.{kernel.id}",
                self.inputs if inputs is None else inputs,
                applies_if=applies_if if len(candidate_types) == 1 else None,
            )[0]
            for kernel in candidate_types
        )
        self._check_candidate_coverage(name, nodes, edges, declarations)

        selected_dependencies: Dependencies = {}
        kernel_choice: Ref[str] | None = None
        physical_spec: DesignSpaceSpec | None = None
        if len(declarations) > 1:
            selection = KernelCandidateSelection(
                f"{self.namespace}.{name}", declarations, applies_if=applies_if
            )
            physical_spec = selection.build_spec()
            kernel_choice = Ref(
                selection.kernel_path,
                DependencyKind.DECISION,
                KERNEL_ID_SEMANTICS,
            )
            selected_dependencies = {"kernel_id": kernel_choice}
            derive_selection: Callable[..., object] = _selected_placement
        elif declarations:
            physical_spec = declarations[0].spec
            derive_selection = _fixed_placement(declarations[0].id)
        else:
            derive_selection = _fixed_placement(None)

        selected = cast(
            "Ref[PlacementSelection]",
            self.derived(
                f"{name}.selected_kernel",
                PLACEMENT_SELECTION_SEMANTICS,
                dependencies=selected_dependencies,
                evaluate=derive_selection,
                applies_if=applies_if,
            ),
        )
        placement = KernelPlacement(
            name,
            nodes,
            edges,
            declarations,
            selected,
            kernel_choice,
        )
        self._placements.append(placement)
        self._hardware.extend(declarations)
        if physical_spec is not None:
            self._physical_specs.append(physical_spec)
        return placement

    def _check_candidate_coverage(
        self,
        placement: str,
        nodes: tuple[DesignNode, ...],
        edges: tuple[DesignEdge, ...],
        candidates: tuple[CompiledKernelDeclaration, ...],
    ) -> None:
        expected_nodes = {item.role: item for item in nodes}
        expected_edges = {item.role: item for item in edges}
        issues: list[SpecAuthoringIssue] = []
        for candidate in candidates:
            actual_nodes = {item.role: item for item in candidate.coverage.regions}
            actual_edges = {item.role: item for item in candidate.coverage.edges}
            if set(actual_nodes) != set(expected_nodes):
                issues.append(
                    SpecAuthoringIssue(
                        "design-placement-region-coverage-differs",
                        f"{self.namespace}.{placement}.{candidate.id}",
                        "candidate Region roles do not equal the placement coverage",
                    )
                )
            for role in set(actual_nodes) & set(expected_nodes):
                actual = actual_nodes[role]
                expected = expected_nodes[role]
                if (
                    actual.region.path != expected.region.path
                    or actual.computation.path != expected.computation.path
                ):
                    issues.append(
                        SpecAuthoringIssue(
                            "design-placement-region-declaration-differs",
                            f"{self.namespace}.{placement}.{candidate.id}.{role}",
                            "candidate does not cover the Region declarations assigned "
                            "to this placement",
                        )
                    )
            if set(actual_edges) != set(expected_edges):
                issues.append(
                    SpecAuthoringIssue(
                        "design-placement-edge-coverage-differs",
                        f"{self.namespace}.{placement}.{candidate.id}",
                        "candidate edge roles do not equal the placement absorption",
                    )
                )
            for role in set(actual_edges) & set(expected_edges):
                actual_edge = actual_edges[role]
                expected_edge = expected_edges[role]
                if (
                    actual_edge.network.path != expected_edge.network.path
                    or actual_edge.source_role != expected_edge.source_role
                    or actual_edge.sink_role != expected_edge.sink_role
                ):
                    issues.append(
                        SpecAuthoringIssue(
                            "design-placement-edge-declaration-differs",
                            f"{self.namespace}.{placement}.{candidate.id}.{role}",
                            "candidate does not absorb the edge assigned to this placement",
                        )
                    )
        if issues:
            raise SpecAuthoringError(tuple(issues))

    @property
    def network_ref(self) -> Ref[DataflowNetwork]:
        if self._network is None:
            raise AuthoringError(f"{self.namespace} declares no Network")
        return self._network

    def _replace_network(self, network: Ref[DataflowNetwork]) -> None:
        self._network = network

    @property
    def nodes(self) -> tuple[DesignNode, ...]:
        return tuple(self._nodes.values())

    @property
    def placements(self) -> tuple[KernelPlacement, ...]:
        return tuple(self._placements)

    @property
    def input_mappings(self) -> tuple[DesignInput, ...]:
        return tuple(self._inputs.values())

    def input_mapping(self, source_operand: str) -> DesignInput | None:
        return self._inputs.get(source_operand)

    @property
    def hardware_declarations(self) -> tuple[CompiledKernelDeclaration, ...]:
        return tuple(self._hardware)

    def spec(self) -> DesignSpaceSpec:
        return assemble_specs((super().spec(), *self._physical_specs))


def _input_refs(inputs: object) -> Mapping[str, Ref[object]]:
    if isinstance(inputs, Mapping):
        values = dict(inputs)
    elif is_dataclass(inputs) and not isinstance(inputs, type):
        values = {item.name: getattr(inputs, item.name) for item in fields(inputs)}
    else:
        raise AuthoringError("class-authored Design inputs must be a mapping or dataclass")
    if not all(isinstance(value, Ref) for value in values.values()):
        raise AuthoringError("every class-authored Design input must be a typed Ref")
    return cast("Mapping[str, Ref[object]]", values)


def _resolve_design_value(
    value: object,
    compiled: object,
    nodes: Mapping[int, DesignNode],
    edges: Mapping[int, DesignEdge],
) -> object:
    members = cast(Any, compiled)
    if isinstance(value, DeclarationTemplate):
        member = members.template_members.get(id(value))
        if member is None:
            raise AuthoringError("a Design value references an undeclared class member")
        return members.ref(member)
    if isinstance(value, Region):
        return nodes[id(value)]
    if isinstance(value, Connection):
        return edges[id(value)]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                item.name: _resolve_design_value(getattr(value, item.name), compiled, nodes, edges)
                for item in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(_resolve_design_value(item, compiled, nodes, edges) for item in value)
    if isinstance(value, list):
        return [_resolve_design_value(item, compiled, nodes, edges) for item in value]
    if isinstance(value, Mapping):
        return {
            _resolve_design_value(key, compiled, nodes, edges): _resolve_design_value(
                item, compiled, nodes, edges
            )
            for key, item in value.items()
        }
    return value


def compile_dataflow_design_class(
    design: type[DataflowDesign],
    namespace: str,
    inputs: object,
    *,
    input_supplies: Sequence[object] = (),
) -> tuple[object, DataflowDesignScope[object]]:
    """Lower one direct Design class through the existing private collector."""

    from finn.dataflow.authoring.inventory import (  # noqa: PLC0415 - cycle boundary
        DataflowDesignDeclaration,
    )

    scope: DataflowDesignScope[object] = DataflowDesignScope(namespace, inputs)
    compiled = compile_class_declarations(
        design,
        layer=DeclarationLayer.DESIGN,
        namespace=namespace,
        imports=_input_refs(inputs),
        scope=scope,
    )
    nodes: dict[int, DesignNode] = {}
    for member_name, group in compiled.groups.items():
        if not isinstance(group, Region):
            continue
        node = scope.node(
            group.role,
            node_id=group.node_id,
            region=cast("Ref[DataflowRegion]", compiled.ref(f"{member_name}.region")),
            computation=cast(
                "Ref[ComputationContract]", compiled.ref(f"{member_name}.computation")
            ),
        )
        nodes[id(group)] = node

    network_members = tuple(
        item
        for item in compiled.members
        if isinstance(getattr(design, item.split(".")[0], None), Network)
    )
    if len(network_members) != 1:
        raise AuthoringError(f"{design.__name__} must declare exactly one Network")
    scope.use_network(cast("Ref[DataflowNetwork]", compiled.ref(network_members[0])))

    edges: dict[int, DesignEdge] = {}
    for member_name, group in compiled.groups.items():
        if not isinstance(group, Connection):
            continue
        edge = scope.edge(
            group.role or member_name,
            edge_id=group.edge_id,
            source=nodes[id(group.source.region)],
            sink=nodes[id(group.sink.region)],
        )
        edges[id(group)] = edge

    for member_name, group in compiled.groups.items():
        if not isinstance(group, SourceInput):
            continue
        node = nodes[id(group.destination.region)]
        consumer = scope.input_interface(
            f"{member_name}.consumer",
            node,
            group.destination.port_id,
        )
        scope.map_input(group.source_operand, boundary_id=group.boundary_id, consumer=consumer)

    for member_name, group in compiled.groups.items():
        if not isinstance(group, Kernels):
            continue
        scope.kernels(
            group.name or member_name,
            covers=tuple(nodes[id(item)] for item in group.covers),
            candidates=group.candidates,
            inputs=_resolve_design_value(group.inputs, compiled, nodes, edges),
            absorbs=tuple(edges[id(item)] for item in group.absorbs),
        )

    compositions = tuple(
        group for group in compiled.groups.values() if isinstance(group, PhysicalComposition)
    )
    if len(compositions) > 1:
        raise AuthoringError(f"{design.__name__} declares more than one physical composer")
    composer = compositions[0] if compositions else None
    composition_inputs = (
        MappingProxyType(
            {
                name: cast("Ref[object]", _resolve_design_value(value, compiled, nodes, edges))
                for name, value in composer.inputs.items()
            }
        )
        if composer is not None
        else MappingProxyType({})
    )

    for supply in input_supplies:
        cast(Any, supply).apply(scope)
    declaration = DataflowDesignDeclaration(
        design.id,
        design.version,
        namespace,
        scope.spec(),
        scope.network_ref,
        scope.nodes,
        scope.placements,
        scope.input_mappings,
        scope.hardware_declarations,
        design,
        MappingProxyType(
            {
                **scope.handles,
                **{
                    name: value
                    for name, value in compiled.members.items()
                    if isinstance(value, Ref)
                },
            }
        ),
        scope.decision_handles,
        (
            *scope.constraint_handles,
            *(
                constraint
                for kernel in scope.hardware_declarations
                for constraint in kernel.constraint_handles
            ),
        ),
        None if composer is None else composer.composer,
        composition_inputs,
    )
    return declaration, scope


def singleton_network(node_id: str, region: DataflowRegion) -> DataflowNetwork:
    """Return the canonical one-node lift, exposing every Region interface."""

    boundaries = tuple(
        BoundaryContract(
            f"input.{interface.port.id}",
            RegionEndpoint(node_id, interface.port.id),
            interface.port.beat_sequence,
        )
        for interface in region.inputs
    ) + tuple(
        BoundaryContract(
            f"output.{interface.port.id}",
            RegionEndpoint(node_id, interface.port.id),
            interface.port.beat_sequence,
        )
        for interface in region.outputs
    )
    return DataflowNetwork((NetworkNode(node_id, region),), (), boundaries)


__all__ = [
    "DataflowDesign",
    "DataflowDesignScope",
    "DesignEdge",
    "DesignInput",
    "DesignNode",
    "KernelPlacement",
    "PlacementSelection",
    "singleton_network",
]
