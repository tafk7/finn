# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-owned conditional input-supply authoring.

An input-supply declaration is shared by every design that maps the same
source operand. Applying it conditionally replaces that external Network
boundary with the supplier Region, edge, and physical placement declared by
the selected alternative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from finn.dataflow.authoring.design import DataflowDesignScope, DesignInput, DesignNode
from finn.dataflow.authoring.scope import AuthoringError, Ref, Scope, finite, predicate, unresolved
from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_NETWORK_SEMANTICS,
    Answer,
    DesignSpaceSpec,
    EvaluatorSpec,
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
from finn.dataflow.region import DataflowRegion, InputInterface
from finn.dataflow.spec_algebra import duplicate_values


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
    decision_handles: tuple[Ref[object], ...] = field(default=(), repr=False, compare=False)

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
                    "consumer": mapping.consumer,
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
    configure: Callable[[Scope, Ref[str]], object] | None = None,
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
    if configure is not None and inputs is not None:
        raise AuthoringError("input supply accepts either inputs or configure, not both")
    scope = Scope(namespace)
    choice = scope.decision("supply", str, domain=finite(modes))
    configured_inputs = configure(scope, choice) if configure is not None else inputs
    return InputSupplyDeclaration(
        id,
        version,
        source_operand,
        namespace,
        choice,
        declared,
        scope.spec(),
        configured_inputs,
        external_id,
        scope.decision_handles,
    )


def _attach_evaluator(
    selected: str,
    mapping: DesignInput,
    attachment: SupplierAttachment,
) -> Callable[..., object]:
    def attach(
        consumer: InputInterface,
        network: DataflowNetwork,
        supply: str,
        supplier: object,
    ) -> object:
        if supply != selected:
            return network
        if supplier is ABSENT:
            return unresolved(
                "design-active-supplier-absent",
                f"input supply {selected!r} is selected but its Region is absent",
                trace=(attachment.node.region,),
            )
        boundary = next(
            (item for item in network.boundaries if item.id == mapping.boundary_id), None
        )
        if boundary is None:
            return unresolved(
                "design-supplied-boundary-absent",
                f"input supply {selected!r} cannot find boundary {mapping.boundary_id!r}",
                trace=(mapping.consumer,),
            )
        declared_consumer = network.node(boundary.endpoint.node_id).region.input_interface(
            boundary.endpoint.port_id
        )
        if declared_consumer != consumer:
            return unresolved(
                "design-supply-consumer-mismatch",
                "the mapped consumer interface does not equal the selected Network boundary",
                trace=(mapping.consumer,),
            )
        return attach_supplier_network(
            network,
            mapping,
            attachment,
            cast(DataflowRegion, supplier),
        )

    return attach


def _position_map(source: DataflowRegion, output: str, consumer: InputInterface) -> PositionMap:
    produced = source.output_interface(output).port.beat_sequence
    demanded = consumer.port.beat_sequence
    source_positions = tuple(position for beat in produced.beats for position in beat)
    sink_positions = tuple(position for beat in demanded.beats for position in beat)
    if len(source_positions) != len(sink_positions):
        raise AuthoringError("supplier and consumer BeatSequences have different field counts")
    mapping: dict[tuple[int, ...], tuple[int, ...]] = {}
    for source_position, sink_position in zip(source_positions, sink_positions):
        previous = mapping.setdefault(source_position, sink_position)
        if previous != sink_position:
            raise AuthoringError("supplier-to-consumer position correspondence is not functional")
    return PositionMap(mapping)


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


__all__ = [
    "InputSupplyAlternative",
    "InputSupplyContext",
    "InputSupplyDeclaration",
    "SupplierAttachment",
    "attach_supplier_network",
    "declare_input_supply",
]
