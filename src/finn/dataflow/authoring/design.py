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

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

from finn.dataflow.authoring.scope import (
    AuthoringError,
    Dependencies,
    DomainFactory,
    Ref,
    Scope,
    T,
    finite,
    predicate,
    unresolved,
)
from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    Absent,
    Answer,
    Decided,
    DependencyKind,
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
from finn.dataflow.hardware.authoring import declare_hardware_kernel
from finn.dataflow.hardware.kernel import (
    BoundRegion,
    ComputationContract,
    HardwareKernel,
    HardwareKernelDeclaration,
    bind_hardware_kernel,
    check_declared_references,
)
from finn.dataflow.hardware.selection import (
    HARDWARE_KERNEL_ID_SEMANTICS,
    HardwareKernelSelection,
)
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import validate_network
from finn.dataflow.region import DataflowRegion, InputInterface
from finn.dataflow.spec_algebra import (
    SpecAuthoringError,
    SpecAuthoringIssue,
    assemble_specs,
    duplicate_values,
    gate_spec,
)

In = TypeVar("In")

DESIGN_REALIZATION_PATH = QualifiedPath("hardware.design_realization")

_COMPUTATION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(ComputationContract, name="ComputationContract")
)
_INPUT_INTERFACE_SEMANTICS = ValueSemantics.immutable_nominal(InputInterface, name="InputInterface")


def _finding(code: str, message: str, **values: object) -> Finding:
    return Finding(
        FindingKind.LIMITATION,
        code,
        DESIGN_REALIZATION_PATH,
        message,
        tuple(values.items()),
    )


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
    candidates: tuple[HardwareKernelDeclaration, ...]
    selected_kernel: Ref[PlacementSelection]

    def candidate(self, kernel_id: str) -> HardwareKernelDeclaration:
        for candidate in self.candidates:
            if candidate.id == kernel_id:
                return candidate
        raise KeyError(f"{kernel_id!r} is not a candidate for placement {self.name!r}")


@dataclass(frozen=True)
class DesignRealization:
    """One resolved Network and the bindings that cover it exactly."""

    design_id: str
    network: DataflowNetwork
    kernels: Mapping[str, HardwareKernel]
    unabsorbed_edges: tuple[str, ...]
    boundaries: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernels", MappingProxyType(dict(self.kernels)))

    def kernel(self, placement: str) -> HardwareKernel:
        try:
            return self.kernels[placement]
        except KeyError:
            raise KeyError(f"no configured Kernel for placement {placement!r}") from None


class DataflowDesign:
    """One contributor-authored logical Network and Kernel partition."""

    id: str = ""
    version: str = "1"

    @classmethod
    def define(cls, design: DataflowDesignScope[Any]) -> None:
        raise NotImplementedError(f"{cls.__name__} does not define a dataflow design")


class DataflowDesignScope(Scope, Generic[In]):
    """The scoped authoring surface handed to a ``DataflowDesign`` subclass."""

    def __init__(self, namespace: str, inputs: In) -> None:
        super().__init__(namespace)
        self.inputs = inputs
        self._network: Ref[DataflowNetwork] | None = None
        self._nodes: dict[str, DesignNode] = {}
        self._placements: list[KernelPlacement] = []
        self._inputs: dict[str, DesignInput] = {}
        self._physical_specs: list[DesignSpaceSpec] = []
        self._hardware: list[HardwareKernelDeclaration] = []

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
        candidates: Sequence[type[HardwareKernel]],
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
            declare_hardware_kernel(
                kernel,
                f"{self.namespace}.{name}.{kernel.id}",
                self.inputs if inputs is None else inputs,
                applies_if=applies_if if len(candidate_types) == 1 else None,
            )[0]
            for kernel in candidate_types
        )
        self._check_candidate_coverage(name, nodes, edges, declarations)

        selected_dependencies: Dependencies = {}
        physical_spec: DesignSpaceSpec | None = None
        if len(declarations) > 1:
            selection = HardwareKernelSelection(
                f"{self.namespace}.{name}", declarations, applies_if=applies_if
            )
            physical_spec = selection.build_spec()
            selected_dependencies = {
                "kernel_id": Ref(
                    selection.kernel_path,
                    DependencyKind.DECISION,
                    HARDWARE_KERNEL_ID_SEMANTICS,
                )
            }
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
        placement = KernelPlacement(name, nodes, edges, declarations, selected)
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
        candidates: tuple[HardwareKernelDeclaration, ...],
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
    def hardware_declarations(self) -> tuple[HardwareKernelDeclaration, ...]:
        return tuple(self._hardware)

    def spec(self) -> DesignSpaceSpec:
        return assemble_specs((super().spec(), *self._physical_specs))


@dataclass(frozen=True)
class SupplierAttachment:
    """One conditional supplier node and the output attached to a boundary."""

    node: DesignNode
    output_port_id: str
    edge_id: str

    def __post_init__(self) -> None:
        if not self.output_port_id or not self.edge_id:
            raise ValueError("a supplier attachment needs an output port and edge id")


@dataclass(frozen=True)
class InputSupplyContext:
    """Operation-neutral inputs handed to one declared supply alternative."""

    design: DataflowDesignScope[Any]
    mapping: DesignInput
    applies_if: EvaluatorSpec[Answer[bool]]
    namespace: str
    inputs: object

    def name(self, local: str) -> str:
        if not local:
            raise AuthoringError("a supply declaration name must not be empty")
        return f"{self.namespace}.{local}"


@dataclass(frozen=True)
class InputSupplyAlternative:
    """One explicitly admitted non-external supply realization."""

    id: str
    define: Callable[[InputSupplyContext], SupplierAttachment]

    def __post_init__(self) -> None:
        if not self.id or self.id == "external":
            raise ValueError("a supplied alternative needs a non-external id")


@dataclass(frozen=True)
class InputSupplyDeclaration:
    """One closed, versioned Operation-level input-supply inventory."""

    id: str
    version: str
    source_operand: str
    namespace: str
    choice: Ref[str]
    alternatives: tuple[InputSupplyAlternative, ...]
    spec: DesignSpaceSpec
    inputs: object
    external_id: str = "external"

    @property
    def modes(self) -> tuple[str, ...]:
        return (self.external_id, *(item.id for item in self.alternatives))

    def selects(self, mode: str) -> EvaluatorSpec[Answer[bool]]:
        if mode not in self.modes:
            raise KeyError(f"{mode!r} is not admitted by input-supply policy {self.id!r}")
        return predicate(
            self.choice.path,
            {"supply": self.choice},
            lambda supply: supply == mode,
        )

    def apply(self, design: DataflowDesignScope[Any]) -> None:
        """Compile this policy at a design mapping without copying its inventory."""

        mapping = design.input_mapping(self.source_operand)
        if mapping is None:
            return
        network = design.network_ref
        for alternative in self.alternatives:
            context = InputSupplyContext(
                design,
                mapping,
                self.selects(alternative.id),
                f"input.{self.source_operand}.{alternative.id}",
                self.inputs,
            )
            attachment = alternative.define(context)
            prior = network
            supplier = attachment.node.region.allow_absent()

            network = design.derived(
                f"input.{mapping.source_operand}.{alternative.id}.network",
                DATAFLOW_NETWORK_SEMANTICS,
                dependencies={
                    "network": prior,
                    "supply": self.choice,
                    "supplier": supplier,
                },
                evaluate=_attach_evaluator(alternative.id, mapping, attachment),
            )
        design._replace_network(network)


def declare_input_supply(
    id: str,
    version: str,
    *,
    namespace: str,
    source_operand: str,
    alternatives: Sequence[InputSupplyAlternative],
    inputs: object = None,
    external_id: str = "external",
) -> InputSupplyDeclaration:
    """Declare one closed supply decision owned by an enclosing Operation."""

    if not id or not version or not namespace or not source_operand or not external_id:
        raise AuthoringError(
            "input supply id, version, namespace, source, and external id are required"
        )
    declared = tuple(alternatives)
    modes = (external_id, *(item.id for item in declared))
    duplicates = duplicate_values(modes)
    if duplicates:
        raise AuthoringError(f"input supply modes are duplicated: {list(duplicates)}")
    scope = Scope(namespace)
    choice = scope.decision("supply", str, domain=finite(modes))
    return InputSupplyDeclaration(
        id,
        version,
        source_operand,
        namespace,
        choice,
        declared,
        scope.spec(),
        inputs,
        external_id,
    )


def _attach_evaluator(
    selected: str,
    mapping: DesignInput,
    attachment: SupplierAttachment,
) -> Callable[..., object]:
    def attach(network: DataflowNetwork, supply: str, supplier: object) -> object:
        if supply != selected:
            return network
        if supplier is ABSENT:
            return unresolved(
                "design-active-supplier-absent",
                f"input supply {selected!r} is selected but its Region is absent",
                trace=(attachment.node.region,),
            )
        return attach_supplier_network(
            network,
            mapping,
            attachment,
            cast(DataflowRegion, supplier),
        )

    return attach


def _design_selected(choice: Ref[str], design_id: str) -> EvaluatorSpec[Answer[bool]]:
    def selected(design: str) -> bool:
        return design == design_id

    return predicate(choice.path, {"design": choice}, selected)


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


def _position_map(source: DataflowRegion, output: str, consumer: InputInterface) -> PositionMap:
    produced = source.output_interface(output).port.beat_sequence
    demanded = consumer.port.beat_sequence
    source_positions = tuple(position for beat in produced.beats for position in beat)
    sink_positions = tuple(position for beat in demanded.beats for position in beat)
    return PositionMap(zip(source_positions, sink_positions))


def attach_supplier_network(
    network: DataflowNetwork,
    mapping: DesignInput,
    attachment: SupplierAttachment,
    supplier: DataflowRegion,
) -> DataflowNetwork:
    """Replace one external input boundary with a declared supplier node."""

    boundaries = tuple(item for item in network.boundaries if item.id == mapping.boundary_id)
    if len(boundaries) != 1:
        raise AuthoringError(
            f"Network must contain exactly one boundary {mapping.boundary_id!r} to supply"
        )
    if any(item.id == attachment.node.node_id for item in network.nodes):
        raise AuthoringError(f"supplier node id {attachment.node.node_id!r} is already present")
    boundary = boundaries[0]
    consumer_node = network.node(boundary.endpoint.node_id)
    consumer = consumer_node.region.input_interface(boundary.endpoint.port_id)
    supplier_boundaries = tuple(
        BoundaryContract(
            f"{attachment.node.node_id}.input.{interface.port.id}",
            RegionEndpoint(attachment.node.node_id, interface.port.id),
            interface.port.beat_sequence,
        )
        for interface in supplier.inputs
    ) + tuple(
        BoundaryContract(
            f"{attachment.node.node_id}.output.{interface.port.id}",
            RegionEndpoint(attachment.node.node_id, interface.port.id),
            interface.port.beat_sequence,
        )
        for interface in supplier.outputs
        if interface.port.id != attachment.output_port_id
    )
    edge = Edge(
        attachment.edge_id,
        RegionEndpoint(attachment.node.node_id, attachment.output_port_id),
        (
            SinkContract(
                boundary.endpoint,
                _position_map(supplier, attachment.output_port_id, consumer),
            ),
        ),
    )
    return DataflowNetwork(
        (*network.nodes, NetworkNode(attachment.node.node_id, supplier)),
        (*network.edges, edge),
        (
            *(item for item in network.boundaries if item.id != mapping.boundary_id),
            *supplier_boundaries,
        ),
    )


@dataclass(frozen=True)
class DataflowDesignDeclaration:
    """The immutable compiled declarations for one authored design."""

    id: str
    version: str
    namespace: str
    spec: DesignSpaceSpec
    network: Ref[DataflowNetwork]
    nodes: tuple[DesignNode, ...]
    placements: tuple[KernelPlacement, ...]
    input_mappings: tuple[DesignInput, ...]
    hardware: tuple[HardwareKernelDeclaration, ...]
    owner: type[DataflowDesign]

    def placement(self, name: str) -> KernelPlacement:
        for placement in self.placements:
            if placement.name == name:
                return placement
        raise KeyError(f"design {self.id!r} has no placement {name!r}")

    def realize(self, engine: Engine, point: DesignPoint) -> Answer[DesignRealization]:
        """Bind every active placement and validate exact resolved coverage."""

        network_answer = engine.query_property(point, self.network.path)
        if not isinstance(network_answer, Decided):
            return cast("Answer[DesignRealization]", network_answer)
        network = cast(DataflowNetwork, network_answer.value)
        network_issues = validate_network(network).issues
        if network_issues:
            return Unresolved(
                tuple(
                    _finding(
                        "design-network-invalid",
                        issue.message,
                        issue_code=issue.code,
                        issue_path=issue.path,
                    )
                    for issue in network_issues
                )
            )

        kernels: dict[str, HardwareKernel] = {}
        active: list[str] = []
        findings: list[Finding] = []
        for placement in self.placements:
            answer = engine.query_property(point, placement.selected_kernel.path)
            if isinstance(answer, Absent):
                continue
            active.append(placement.name)
            if not isinstance(answer, Decided):
                findings.extend(answer.findings)
                continue
            selected = cast(PlacementSelection, answer.value)
            if selected.kernel_id is None:
                findings.append(
                    _finding(
                        "design-placement-has-no-kernel",
                        f"active placement {placement.name!r} has no physical Kernel candidate",
                        placement=placement.name,
                    )
                )
                continue
            candidate = placement.candidate(selected.kernel_id)
            regions: dict[str, BoundRegion] = {}
            missing = False
            for node in placement.nodes:
                try:
                    resolved_node = network.node(node.node_id)
                except KeyError:
                    findings.append(
                        _finding(
                            "design-placement-node-absent",
                            f"active placement {placement.name!r} covers node "
                            f"{node.node_id!r}, which is absent from the resolved Network",
                            placement=placement.name,
                            node=node.node_id,
                        )
                    )
                    missing = True
                    continue
                regions[node.role] = BoundRegion(node.role, node.node_id, resolved_node.region)
            edge_ids = {item.role: item.edge_id for item in placement.edges}
            if missing:
                continue
            bound = bind_hardware_kernel(engine, candidate, point, regions, edge_ids)
            if isinstance(bound, Decided):
                kernels[placement.name] = bound.value
            else:
                findings.extend(bound.findings)
        if findings:
            return Unresolved(tuple(findings))
        return self.validate_realization(network, kernels, active_placements=tuple(active))

    def validate_realization(
        self,
        network: DataflowNetwork,
        kernels: Mapping[str, HardwareKernel],
        *,
        active_placements: Sequence[str] | None = None,
    ) -> Answer[DesignRealization]:
        """Validate one already-configured Kernel collection against this design."""

        active_names = (
            tuple(item.name for item in self.placements)
            if active_placements is None
            else tuple(active_placements)
        )
        by_name = {item.name: item for item in self.placements}
        findings: list[Finding] = []
        for name in sorted(set(active_names) - set(by_name)):
            findings.append(
                _finding(
                    "design-active-placement-unknown",
                    f"active placement {name!r} is not declared by design {self.id!r}",
                    placement=name,
                )
            )
        for name in sorted(set(active_names) - set(kernels)):
            findings.append(
                _finding(
                    "design-placement-kernel-missing",
                    f"active placement {name!r} has no configured Kernel",
                    placement=name,
                )
            )
        for name in sorted(set(kernels) - set(active_names)):
            findings.append(
                _finding(
                    "design-placement-kernel-foreign",
                    f"configured Kernel was supplied for inactive or unknown placement {name!r}",
                    placement=name,
                )
            )
        placed: dict[str, HardwareKernel] = {}
        for name in active_names:
            placement = by_name.get(name)
            kernel = kernels.get(name)
            if placement is None or kernel is None:
                continue
            candidate_ids = {item.id for item in placement.candidates}
            if kernel.id not in candidate_ids:
                findings.append(
                    _finding(
                        "design-placement-kernel-not-a-candidate",
                        f"Kernel {kernel.id!r} is not a candidate for placement {name!r}",
                        placement=name,
                        kernel=kernel.id,
                    )
                )
            expected_nodes = {item.node_id for item in placement.nodes}
            if set(kernel.node_ids) != expected_nodes:
                findings.append(
                    _finding(
                        "design-placement-node-coverage-mismatch",
                        f"Kernel {kernel.id!r} does not cover placement {name!r} exactly",
                        placement=name,
                        expected=tuple(sorted(expected_nodes)),
                        actual=kernel.node_ids,
                    )
                )
            expected_edges = {item.edge_id for item in placement.edges}
            if set(kernel.edge_ids) != expected_edges:
                findings.append(
                    _finding(
                        "design-placement-edge-coverage-mismatch",
                        f"Kernel {kernel.id!r} does not absorb placement {name!r} exactly",
                        placement=name,
                        expected=tuple(sorted(expected_edges)),
                        actual=kernel.edge_ids,
                    )
                )
            placed[name] = kernel
        coverage = _validate_realization(self.id, network, placed)
        if isinstance(coverage, Unresolved):
            findings.extend(coverage.findings)
        if findings:
            return Unresolved(tuple(findings))
        return coverage


def _validate_realization(
    design_id: str,
    network: DataflowNetwork,
    placed: Mapping[str, HardwareKernel],
) -> Answer[DesignRealization]:
    findings: list[Finding] = []
    node_counts = Counter(node for kernel in placed.values() for node in kernel.node_ids)
    edge_counts = Counter(edge for kernel in placed.values() for edge in kernel.edge_ids)
    network_nodes = {item.id for item in network.nodes}
    network_edges = {item.id: item for item in network.edges}

    for node in sorted(network_nodes):
        count = node_counts[node]
        if count != 1:
            findings.append(
                _finding(
                    "design-node-coverage-not-exact",
                    f"Network node {node!r} is covered {count} times instead of once",
                    node=node,
                    count=count,
                )
            )
    for node in sorted(set(node_counts) - network_nodes):
        findings.append(
            _finding(
                "design-foreign-node-coverage",
                f"configured Kernels cover foreign node {node!r}",
                node=node,
            )
        )
    for edge_id, count in sorted(edge_counts.items()):
        edge = network_edges.get(edge_id)
        if edge is None:
            findings.append(
                _finding(
                    "design-foreign-edge-absorption",
                    f"configured Kernels absorb foreign edge {edge_id!r}",
                    edge=edge_id,
                )
            )
            continue
        if count != 1:
            findings.append(
                _finding(
                    "design-edge-absorption-not-unique",
                    f"Network edge {edge_id!r} is absorbed {count} times",
                    edge=edge_id,
                    count=count,
                )
            )
        absorber = next(kernel for kernel in placed.values() if edge_id in kernel.edge_ids)
        required = {edge.source.node_id, *(sink.endpoint.node_id for sink in edge.sinks)}
        covered = set(absorber.node_ids)
        if not required <= covered:
            findings.append(
                _finding(
                    "design-absorbed-edge-incomplete-fanout",
                    f"Kernel {absorber.kernel_id!r} absorbs edge {edge_id!r} without "
                    "covering its source and every sink",
                    edge=edge_id,
                    missing=tuple(sorted(required - covered)),
                )
            )
    if findings:
        return Unresolved(tuple(findings))
    return Decided(
        DesignRealization(
            design_id,
            network,
            placed,
            tuple(sorted(set(network_edges) - set(edge_counts))),
            tuple(item.id for item in network.boundaries),
        )
    )


def declare_dataflow_design(
    design: type[DataflowDesign],
    namespace: str,
    inputs: object,
    *,
    input_supplies: Sequence[InputSupplyDeclaration] = (),
) -> tuple[DataflowDesignDeclaration, DataflowDesignScope[object]]:
    """Run one ``DataflowDesign`` subclass under a collision-free namespace."""

    if not design.id or not design.version:
        raise AuthoringError(f"{design.__name__} must set a design id and version")
    scope: DataflowDesignScope[object] = DataflowDesignScope(namespace, inputs)
    design.define(scope)
    for supply in input_supplies:
        supply.apply(scope)
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
    )
    return declaration, scope


@dataclass(frozen=True)
class DataflowDesignEntry:
    """One design class and the typed inputs its declaration receives."""

    design: type[DataflowDesign]
    inputs: object


@dataclass(frozen=True)
class DataflowDesignInventory:
    """One Operation's closed inventory of selectable designs."""

    namespace: str
    declarations: tuple[DataflowDesignDeclaration, ...]
    input_supplies: tuple[InputSupplyDeclaration, ...]
    design_path: QualifiedPath | None
    specification: DesignSpaceSpec

    @property
    def design_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.declarations)

    def declaration(self, design_id: str) -> DataflowDesignDeclaration:
        for declaration in self.declarations:
            if declaration.id == design_id:
                return declaration
        raise KeyError(f"{design_id!r} is not in the {self.namespace!r} design inventory")

    def selected(self, point: DesignPoint) -> Answer[DataflowDesignDeclaration]:
        if self.design_path is None:
            return Decided(self.declarations[0])
        if self.design_path not in point.assignments:
            return Unresolved(
                (
                    _finding(
                        "dataflow-design-unselected",
                        f"no DataflowDesign is committed for {self.namespace}",
                    ),
                )
            )
        return Decided(self.declaration(cast(str, point.assignments[self.design_path])))

    def realize(self, engine: Engine, point: DesignPoint) -> Answer[DesignRealization]:
        selected = self.selected(point)
        if not isinstance(selected, Decided):
            return cast("Answer[DesignRealization]", selected)
        return selected.value.realize(engine, point)


def declare_dataflow_design_inventory(
    namespace: str,
    entries: Sequence[DataflowDesignEntry],
    *,
    input_supplies: Sequence[InputSupplyDeclaration] = (),
    shared_specs: Sequence[DesignSpaceSpec] = (),
) -> DataflowDesignInventory:
    """Compile a closed Operation inventory into one ordinary flat spec."""

    declared_entries = tuple(entries)
    if not namespace or not declared_entries:
        raise AuthoringError("a design inventory needs a namespace and at least one design")
    ids = tuple(item.design.id for item in declared_entries)
    if any(not item for item in ids):
        raise AuthoringError("every DataflowDesign in an inventory must have an id")
    duplicates = duplicate_values(ids)
    if duplicates:
        raise AuthoringError(f"DataflowDesign ids are duplicated: {list(duplicates)}")
    supplies = tuple(input_supplies)
    source_duplicates = duplicate_values(tuple(item.source_operand for item in supplies))
    if source_duplicates:
        raise AuthoringError(
            f"input-supply policies overlap source operands: {list(source_duplicates)}"
        )

    declarations = tuple(
        declare_dataflow_design(
            entry.design,
            f"{namespace}.design.{entry.design.id}",
            entry.inputs,
            input_supplies=supplies,
        )[0]
        for entry in declared_entries
    )
    design_path: QualifiedPath | None = None
    own_spec = DesignSpaceSpec()
    design_specs: tuple[DesignSpaceSpec, ...]
    if len(declarations) == 1:
        design_specs = (declarations[0].spec,)
    else:
        selector = Scope(namespace)
        choice = selector.decision("design", str, domain=finite(ids))
        design_path = choice.path
        own_spec = selector.spec()
        design_specs = tuple(
            gate_spec(
                declaration.spec,
                _design_selected(choice, declaration.id),
            )
            for declaration in declarations
        )

    specification = assemble_specs(
        (
            *tuple(shared_specs),
            own_spec,
            *(item.spec for item in supplies),
            *design_specs,
        )
    )
    check_declared_references(
        specification,
        tuple(kernel for declaration in declarations for kernel in declaration.hardware),
    )
    return DataflowDesignInventory(
        namespace,
        declarations,
        supplies,
        design_path,
        specification,
    )


__all__ = [
    "DataflowDesign",
    "DataflowDesignEntry",
    "DataflowDesignInventory",
    "DataflowDesignScope",
    "DesignEdge",
    "DesignInput",
    "DesignNode",
    "DesignRealization",
    "InputSupplyAlternative",
    "InputSupplyContext",
    "InputSupplyDeclaration",
    "SupplierAttachment",
    "attach_supplier_network",
    "declare_dataflow_design",
    "declare_dataflow_design_inventory",
    "declare_input_supply",
    "singleton_network",
]
