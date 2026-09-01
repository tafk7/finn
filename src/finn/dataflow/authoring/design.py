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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from finn.dataflow.authoring.scope import (
    AuthoringError,
    Dependencies,
    DomainFactory,
    Ref,
    Scope,
    T,
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
