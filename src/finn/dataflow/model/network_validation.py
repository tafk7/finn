# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Structural validation for normalized dataflow networks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from finn.dataflow.model.maps import (
    AffineRankMap,
    IdentityCoordinateMap,
    InvalidMapError,
    MapCapabilityError,
    ValidationCapabilityError,
    rank_transform_affine,
    require_int,
)
from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    DirectConnection,
    Edge,
    FanoutMode,
    NetworkNode,
    OrderedChannel,
    PassCorrespondence,
    PositionMap,
    RegionEndpoint,
)
from finn.dataflow.model.region import InputInterface, OutputInterface, Port
from finn.dataflow.model.region_validation import validate_region


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


@dataclass(frozen=True, slots=True)
class NetworkValidationBudget:
    """Explicit limit for generated beat fields in an exact fallback."""

    max_generated_fields: int

    def __post_init__(self) -> None:
        value = require_int(self.max_generated_fields, "max_generated_fields")
        if value < 0:
            raise ValueError("max_generated_fields must be non-negative")


@dataclass(frozen=True, slots=True)
class _OrderedAgreementMismatch:
    ordinal: int | None = None
    field: int | None = None


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
    expansion_budget: NetworkValidationBudget | None,
) -> list[NetworkValidationIssue]:
    path = f"edge[{edge.id!r}].sinks[{sink_index}]"
    issues = []
    declared_map = edge.sinks[sink_index].position_map
    try:
        position_map = declared_map.bind_domains(
            source.operand.position_domain, sink.operand.position_domain
        )
        domains_usable = True
    except MapCapabilityError as exc:
        raise ValidationCapabilityError(
            "position-map domains cannot be compared by the compact validator"
        ) from exc
    except ValueError:
        position_map = declared_map
        domains_usable = False
    position_map_usable = domains_usable
    entries = position_map.entries if position_map.is_explicit else None
    explicit_lookup: Mapping[tuple[int, ...], tuple[int, ...]] | None = None
    if entries is not None:
        source_positions = tuple(position for position, _mapped in entries)
        sink_positions = tuple(mapped for _position, mapped in entries)
        if len(source_positions) != len(set(source_positions)):
            position_map_usable = False
            issues.append(
                NetworkValidationIssue(
                    "position_map.source_not_function",
                    f"{path}.position_map",
                    "a source position is mapped more than once",
                )
            )
        else:
            explicit_lookup = dict(entries)
        if len(sink_positions) != len(set(sink_positions)):
            position_map_usable = False
            issues.append(
                NetworkValidationIssue(
                    "position_map.not_injective",
                    f"{path}.position_map",
                    "more than one source position maps to the same sink position",
                )
            )
        if not all(source.operand.contains_position(position) for position in source_positions):
            position_map_usable = False
        if not all(sink.operand.contains_position(position) for position in sink_positions):
            position_map_usable = False
    else:
        coordinate_map = position_map.coordinate_map
        if isinstance(coordinate_map, AffineRankMap) and not coordinate_map.is_injective:
            position_map_usable = False
            issues.append(
                NetworkValidationIssue(
                    "position_map.not_injective",
                    f"{path}.position_map",
                    "compact position map is not injective on its typed source domain",
                )
            )

    if domains_usable:
        try:
            source_set = position_map.source_set
            sink_set = position_map.sink_set
            source_image = source.beat_sequence.image_set
            sink_image = sink.beat_sequence.image_set
            source_matches = source_set == source_image
            sink_matches = sink_set == sink_image
        except (InvalidMapError, MapCapabilityError, ValueError):
            position_map_usable = False
            source_matches = False
            sink_matches = False
    else:
        source_matches = False
        sink_matches = False

    if not source_matches:
        position_map_usable = False
        issues.append(
            NetworkValidationIssue(
                "position_map.source_domain_mismatch",
                f"{path}.position_map",
                "position-map domain does not equal the source beat image",
            )
        )
    if not sink_matches:
        position_map_usable = False
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
    equal_field_count = (
        source.beat_sequence.elements_per_beat == sink.beat_sequence.elements_per_beat
    )
    if not equal_field_count:
        issues.append(
            NetworkValidationIssue(
                "edge.element_count_mismatch",
                path,
                "source and sink elements-per-beat differ",
            )
        )
    equal_beat_count = source.beat_sequence.beat_count == sink.beat_sequence.beat_count
    if not equal_beat_count:
        issues.append(
            NetworkValidationIssue(
                "edge.beat_count_mismatch",
                path,
                "source and sink beat counts differ",
            )
        )
    if position_map_usable and equal_field_count and equal_beat_count:
        mismatch = _ordered_agreement_mismatch(
            position_map,
            source,
            sink,
            expansion_budget,
            explicit_lookup=explicit_lookup,
        )
        if mismatch is not None:
            issue_path = (
                f"{path}.beat_sequence"
                if mismatch.ordinal is None
                else f"{path}.beat[{mismatch.ordinal}].field[{mismatch.field}]"
            )
            issues.append(
                NetworkValidationIssue(
                    "edge.beat_sequence_mismatch",
                    issue_path,
                    "mapped source position does not equal the sink position",
                )
            )
    return issues


def _recognized_rank_form(position_map: PositionMap, source: Port, sink: Port) -> str | None:
    coordinate_map = position_map.coordinate_map
    if isinstance(coordinate_map, IdentityCoordinateMap):
        return "identity"
    if isinstance(coordinate_map, AffineRankMap):
        if coordinate_map.is_rank_identity:
            return "identity"
        if coordinate_map.is_rank_reversal:
            return "reversal"
        return None
    entries = position_map.entries
    cardinality = source.operand.position_domain.cardinality
    if len(entries) != cardinality:
        return None
    source_domain = source.operand.position_domain
    sink_domain = sink.operand.position_domain
    if all(
        sink_domain.rank_of(target) == source_domain.rank_of(origin) for origin, target in entries
    ):
        return "identity"
    if all(
        sink_domain.rank_of(target) == cardinality - 1 - source_domain.rank_of(origin)
        for origin, target in entries
    ):
        return "reversal"
    return None


def _ordered_agreement_mismatch(
    position_map: PositionMap,
    source: Port,
    sink: Port,
    expansion_budget: NetworkValidationBudget | None,
    *,
    explicit_lookup: Mapping[tuple[int, ...], tuple[int, ...]] | None,
) -> _OrderedAgreementMismatch | None:
    source_beats = source.beat_sequence
    sink_beats = sink.beat_sequence

    def mapped(position: tuple[int, ...]) -> tuple[int, ...]:
        return (
            explicit_lookup[position]
            if explicit_lookup is not None
            else position_map.mapped(position)
        )

    if source_beats.is_explicit:
        for ordinal, beat in enumerate(source_beats.beats):
            for field, source_position in enumerate(beat):
                if mapped(source_position) != sink_beats.position_at(ordinal, field):
                    return _OrderedAgreementMismatch(ordinal, field)
        return None
    if sink_beats.is_explicit:
        for ordinal, beat in enumerate(sink_beats.beats):
            for field, sink_position in enumerate(beat):
                source_position = source_beats.position_at(ordinal, field)
                if mapped(source_position) != sink_position:
                    return _OrderedAgreementMismatch(ordinal, field)
        return None

    rank_form = _recognized_rank_form(position_map, source, sink)
    source_affine = source_beats.affine_map
    sink_affine = sink_beats.affine_map
    assert source_affine is not None and sink_affine is not None
    if rank_form is not None:
        composed = rank_transform_affine(
            source_affine,
            target=sink.operand.position_domain,
            reverse=rank_form == "reversal",
        )
        return None if composed == sink_affine else _OrderedAgreementMismatch()

    field_count = source_beats.delivered_field_count
    if expansion_budget is None:
        raise ValidationCapabilityError(
            "compact edge agreement is unsupported without NetworkValidationBudget"
        )
    if field_count > expansion_budget.max_generated_fields:
        raise ValidationCapabilityError(
            "compact edge agreement requires "
            f"{field_count} generated fields, exceeding budget "
            f"{expansion_budget.max_generated_fields}"
        )
    for ordinal in range(source_beats.beat_count):
        for field in range(source_beats.elements_per_beat):
            if mapped(source_beats.position_at(ordinal, field)) != (
                sink_beats.position_at(ordinal, field)
            ):
                return _OrderedAgreementMismatch(ordinal, field)
    return None


def _validate_edge(
    edge: Edge,
    nodes: dict[str, NetworkNode],
    expansion_budget: NetworkValidationBudget | None,
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
            issues.extend(_validate_position_map(edge, sink_index, source, sink, expansion_budget))
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


def validate_network(
    network: DataflowNetwork,
    *,
    expansion_budget: NetworkValidationBudget | None = None,
) -> NetworkValidationReport:
    """Return all independently detectable ``NETWORK.md`` §11.2.1 issues."""
    if not isinstance(network, DataflowNetwork):
        raise TypeError("network must be a DataflowNetwork")
    if expansion_budget is not None and not isinstance(expansion_budget, NetworkValidationBudget):
        raise TypeError("expansion_budget must be a NetworkValidationBudget")
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
        edge_issues, inputs, outputs = _validate_edge(edge, nodes, expansion_budget)
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

    # Ported inputs only.  An internal input is not a network endpoint: there is
    # no port for an edge to sink into or a boundary to expose, so requiring it
    # to be consumed or exposed exactly once would refuse every region that
    # declares one.
    expected_inputs = [
        RegionEndpoint(node.id, interface.port.id)
        for node in network.nodes
        for interface in node.region.input_interfaces
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
    "NetworkValidationBudget",
    "NetworkValidationIssue",
    "NetworkValidationReport",
    "is_network_structurally_well_formed",
    "validate_network",
]
