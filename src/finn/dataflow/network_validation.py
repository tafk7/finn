# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Structural validation for normalized dataflow networks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    DirectConnection,
    Edge,
    FanoutMode,
    NetworkNode,
    OrderedChannel,
    PassCorrespondence,
    RegionEndpoint,
)
from finn.dataflow.region import InputInterface, OutputInterface, Port
from finn.dataflow.region_validation import validate_region


@dataclass(frozen=True)
class NetworkValidationIssue:
    """One deterministic model-local network issue."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class NetworkValidationReport:
    """Immutable collection of network structural issues."""

    issues: tuple[NetworkValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))

    def __len__(self) -> int:
        return len(self.issues)

    def __iter__(self) -> Iterator[NetworkValidationIssue]:
        return iter(self.issues)

    def __bool__(self) -> bool:
        return bool(self.issues)


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def _resolve_port(
    nodes: dict[str, NetworkNode], endpoint: RegionEndpoint, *, output: bool
) -> Port | None:
    node = nodes.get(endpoint.node_id)
    if node is None:
        return None
    region = node.region
    try:
        interface = (
            region.output_interface(endpoint.port_id)
            if output
            else region.input_interface(endpoint.port_id)
        )
    except KeyError:
        return None
    expected = OutputInterface if output else InputInterface
    return interface.port if isinstance(interface, expected) else None


def _validate_position_map(
    edge: Edge,
    sink_index: int,
    source: Port,
    sink: Port,
) -> list[NetworkValidationIssue]:
    path = f"edge[{edge.id!r}].sinks[{sink_index}]"
    issues = []
    entries = edge.sinks[sink_index].position_map.entries
    source_positions = tuple(position for position, _mapped in entries)
    sink_positions = tuple(mapped for _position, mapped in entries)
    if len(source_positions) != len(set(source_positions)):
        issues.append(
            NetworkValidationIssue(
                "position_map.source_not_function",
                f"{path}.position_map",
                "a source position is mapped more than once",
            )
        )
    if len(sink_positions) != len(set(sink_positions)):
        issues.append(
            NetworkValidationIssue(
                "position_map.not_injective",
                f"{path}.position_map",
                "more than one source position maps to the same sink position",
            )
        )
    if frozenset(source_positions) != source.beat_sequence.image:
        issues.append(
            NetworkValidationIssue(
                "position_map.source_domain_mismatch",
                f"{path}.position_map",
                "position-map domain does not equal the source beat image",
            )
        )
    if frozenset(sink_positions) != sink.beat_sequence.image:
        issues.append(
            NetworkValidationIssue(
                "position_map.sink_domain_mismatch",
                f"{path}.position_map",
                "position-map codomain does not equal the sink beat image",
            )
        )
    if source.operand.element_type != sink.operand.element_type:
        issues.append(
            NetworkValidationIssue(
                "edge.element_type_mismatch",
                path,
                "source and sink numeric element types differ",
            )
        )
    if source.beat_sequence.elements_per_beat != sink.beat_sequence.elements_per_beat:
        issues.append(
            NetworkValidationIssue(
                "edge.element_count_mismatch",
                path,
                "source and sink elements-per-beat differ",
            )
        )
    if source.beat_sequence.beat_count != sink.beat_sequence.beat_count:
        issues.append(
            NetworkValidationIssue(
                "edge.beat_count_mismatch",
                path,
                "source and sink beat counts differ",
            )
        )
    if not issues:
        mapping = dict(entries)
        for ordinal, source_beat in enumerate(source.beat_sequence.beats):
            sink_beat = sink.beat_sequence.beats[ordinal]
            for field, source_position in enumerate(source_beat):
                if mapping[source_position] != sink_beat[field]:
                    issues.append(
                        NetworkValidationIssue(
                            "edge.beat_sequence_mismatch",
                            f"{path}.beat[{ordinal}].field[{field}]",
                            "mapped source position does not equal the sink position",
                        )
                    )
    return issues


def _validate_edge(
    edge: Edge, nodes: dict[str, NetworkNode]
) -> tuple[list[NetworkValidationIssue], list[RegionEndpoint], list[RegionEndpoint]]:
    issues = []
    used_outputs = [edge.source]
    used_inputs = [sink.endpoint for sink in edge.sinks]
    source = _resolve_port(nodes, edge.source, output=True)
    if source is None:
        issues.append(
            NetworkValidationIssue(
                "edge.source_missing_or_not_output",
                f"edge[{edge.id!r}].source",
                "edge source must name an existing output interface",
            )
        )
    if not edge.sinks:
        issues.append(
            NetworkValidationIssue(
                "edge.sinks_empty",
                f"edge[{edge.id!r}].sinks",
                "an edge must contain at least one sink",
            )
        )
    if len(used_inputs) != len(set(used_inputs)):
        issues.append(
            NetworkValidationIssue(
                "edge.sink_duplicate",
                f"edge[{edge.id!r}].sinks",
                "an edge sink list must contain distinct endpoints",
            )
        )
    if edge.fanout is not FanoutMode.REPLICATE:
        issues.append(
            NetworkValidationIssue(
                "edge.fanout_unsupported",
                f"edge[{edge.id!r}].fanout",
                "only replicated fan-out is supported",
            )
        )
    if edge.pass_correspondence is not PassCorrespondence.ONE_TO_ONE:
        issues.append(
            NetworkValidationIssue(
                "edge.pass_correspondence_unsupported",
                f"edge[{edge.id!r}].pass_correspondence",
                "only one-to-one pass correspondence is supported",
            )
        )
    if not isinstance(edge.transport, (DirectConnection, OrderedChannel)):
        issues.append(
            NetworkValidationIssue(
                "edge.transport_unsupported",
                f"edge[{edge.id!r}].transport",
                "transport must be direct or an ordered channel",
            )
        )
    elif isinstance(edge.transport, OrderedChannel):
        channel = edge.transport.specification
        if not channel.preserve_order or (
            channel.pass_correspondence is not PassCorrespondence.ONE_TO_ONE
        ):
            issues.append(
                NetworkValidationIssue(
                    "channel.contract_unsupported",
                    f"edge[{edge.id!r}].transport",
                    "an ordered channel must preserve order and one-to-one passes",
                )
            )
    for sink_index, sink_contract in enumerate(edge.sinks):
        sink = _resolve_port(nodes, sink_contract.endpoint, output=False)
        if sink is None:
            issues.append(
                NetworkValidationIssue(
                    "edge.sink_missing_or_not_input",
                    f"edge[{edge.id!r}].sinks[{sink_index}].endpoint",
                    "edge sink must name an existing input interface",
                )
            )
        if source is not None and sink is not None:
            issues.extend(_validate_position_map(edge, sink_index, source, sink))
    return issues, used_inputs, used_outputs


def _validate_boundary(
    boundary: BoundaryContract, nodes: dict[str, NetworkNode]
) -> tuple[list[NetworkValidationIssue], bool | None]:
    path = f"boundary[{boundary.id!r}]"
    input_port = _resolve_port(nodes, boundary.endpoint, output=False)
    output_port = _resolve_port(nodes, boundary.endpoint, output=True)
    port = input_port if input_port is not None else output_port
    if port is None or (input_port is not None and output_port is not None):
        return (
            [
                NetworkValidationIssue(
                    "boundary.endpoint_missing",
                    f"{path}.endpoint",
                    "boundary must name exactly one existing region interface",
                )
            ],
            None,
        )
    issues = []
    if boundary.pass_correspondence is not PassCorrespondence.ONE_TO_ONE:
        issues.append(
            NetworkValidationIssue(
                "boundary.pass_correspondence_unsupported",
                f"{path}.pass_correspondence",
                "only one-to-one boundary pass correspondence is supported",
            )
        )
    if boundary.external_beat_sequence != port.beat_sequence:
        issues.append(
            NetworkValidationIssue(
                "boundary.beat_sequence_mismatch",
                f"{path}.external_beat_sequence",
                "external beat sequence does not equal the endpoint sequence",
            )
        )
    return issues, input_port is not None


def _has_cycle(network: DataflowNetwork) -> bool:
    adjacency: dict[str, set[str]] = {node.id: set() for node in network.nodes}
    indegree = {node.id: 0 for node in network.nodes}
    for edge in network.edges:
        for sink in edge.sinks:
            if edge.source.node_id in adjacency and sink.endpoint.node_id in adjacency:
                if sink.endpoint.node_id not in adjacency[edge.source.node_id]:
                    adjacency[edge.source.node_id].add(sink.endpoint.node_id)
                    indegree[sink.endpoint.node_id] += 1
    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for successor in sorted(adjacency[node_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    return visited != len(adjacency)


def validate_network(network: DataflowNetwork) -> NetworkValidationReport:
    """Return all independently detectable ``NETWORK.md`` §11.2.1 issues."""
    if not isinstance(network, DataflowNetwork):
        raise TypeError("network must be a DataflowNetwork")
    issues = []
    for identity, label in (
        (tuple(node.id for node in network.nodes), "node"),
        (tuple(edge.id for edge in network.edges), "edge"),
        (tuple(boundary.id for boundary in network.boundaries), "boundary"),
    ):
        for duplicate in _duplicates(identity):
            issues.append(
                NetworkValidationIssue(
                    f"{label}.id_duplicate",
                    f"network.{label}s",
                    f"{label} identity {duplicate!r} is not unique",
                )
            )
    nodes = {node.id: node for node in network.nodes}
    for node in network.nodes:
        for issue in validate_region(node.region).issues:
            issues.append(
                NetworkValidationIssue(
                    f"node.region.{issue.code}",
                    f"node[{node.id!r}].{issue.path}",
                    issue.message,
                )
            )

    input_uses: list[RegionEndpoint] = []
    output_uses: list[RegionEndpoint] = []
    for edge in network.edges:
        edge_issues, inputs, outputs = _validate_edge(edge, nodes)
        issues.extend(edge_issues)
        input_uses.extend(inputs)
        output_uses.extend(outputs)
    for boundary in network.boundaries:
        boundary_issues, is_input = _validate_boundary(boundary, nodes)
        issues.extend(boundary_issues)
        if is_input is True:
            input_uses.append(boundary.endpoint)
        elif is_input is False:
            output_uses.append(boundary.endpoint)

    expected_inputs = [
        RegionEndpoint(node.id, interface.port.id)
        for node in network.nodes
        for interface in node.region.inputs
    ]
    expected_outputs = [
        RegionEndpoint(node.id, interface.port.id)
        for node in network.nodes
        for interface in node.region.outputs
    ]
    for endpoint in expected_inputs:
        count = input_uses.count(endpoint)
        if count != 1:
            issues.append(
                NetworkValidationIssue(
                    "endpoint.input_ownership",
                    f"endpoint[{endpoint.node_id!r},{endpoint.port_id!r}]",
                    f"input endpoint must be consumed or exposed exactly once, found {count}",
                )
            )
    for endpoint in expected_outputs:
        count = output_uses.count(endpoint)
        if count != 1:
            issues.append(
                NetworkValidationIssue(
                    "endpoint.output_ownership",
                    f"endpoint[{endpoint.node_id!r},{endpoint.port_id!r}]",
                    f"output endpoint must be consumed or exposed exactly once, found {count}",
                )
            )
    if _has_cycle(network):
        issues.append(
            NetworkValidationIssue(
                "network.cycle",
                "network.edges",
                "the directed region graph must be acyclic",
            )
        )
    return NetworkValidationReport(
        tuple(sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message)))
    )


def is_network_structurally_well_formed(network: DataflowNetwork) -> bool:
    """Return whether ``network`` has no structural validation issues."""
    return not validate_network(network)


__all__ = [
    "NetworkValidationIssue",
    "NetworkValidationReport",
    "is_network_structurally_well_formed",
    "validate_network",
]
