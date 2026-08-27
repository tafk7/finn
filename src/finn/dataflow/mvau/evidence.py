# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Named realizability evidence for the emitted RTL soft-vector MVAU slice."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from finn.dataflow.mvau.artifacts import MVAUBuiltRTLArtifact
from finn.dataflow.mvau.definition import MVAUComputeKernelPaths
from finn.dataflow.mvau.elaboration import MVAUPhysicalElaboration, MVAUSemanticPortRef
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.ops.mvau import NetworkRef, RegionRef
from finn.dataflow.region import BeatSequence, Coordinate, DataflowRegion, InputInterface


@dataclass(frozen=True, order=True)
class MVAURequirementServiceAssignment:
    """One declared requirement occurrence and the beat slot that serves it."""

    iteration: Coordinate
    operand_position: Coordinate
    multiplicity_index: int
    beat_ordinal: int
    field_ordinal: int


@dataclass(frozen=True)
class MVAUInputServiceEvidence:
    """Exact service mapping for one input of the covered MVAU realization."""

    region_id: str
    port_id: str
    source_kind: str
    source_sequence: BeatSequence
    assignments: tuple[MVAURequirementServiceAssignment, ...]
    replay_component_id: str | None
    complete: bool


@dataclass(frozen=True)
class MVAUOutputEvidence:
    """Exact final-availability and boundary-order evidence."""

    region_id: str
    availability: tuple[tuple[Coordinate, Coordinate], ...]
    output_sequence: BeatSequence
    artifact_sequence: BeatSequence
    every_final_output_available: bool
    field_and_beat_order_preserved: bool


@dataclass(frozen=True)
class MVAUAssociationEvidence:
    """Coverage of semantic ports and edges by physical associations."""

    required_ports: tuple[MVAUSemanticPortRef, ...]
    missing_ports: tuple[MVAUSemanticPortRef, ...]
    required_edge_ids: tuple[str, ...]
    missing_edge_ids: tuple[str, ...]
    required_source_owner_ids: tuple[str, ...]
    missing_source_owner_ids: tuple[str, ...]
    required_region_ids: tuple[str, ...]
    missing_region_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            not self.missing_ports
            and not self.missing_edge_ids
            and not self.missing_source_owner_ids
            and not self.missing_region_ids
        )


@dataclass(frozen=True)
class MVAURTLCycleEvidence:
    """Configuration-qualified empirical cycle observation."""

    oracle: str
    configuration: tuple[tuple[str, bool | float | int | str], ...]
    analytical_work_cycles: int
    measured_cycles: int
    absolute_difference: int
    within_legacy_tolerance: bool


@dataclass(frozen=True)
class MVAURTLSoftvecEvidence:
    """Evidence bundle scoped only to the emitted FINN RTL soft-vector path."""

    activation_service: MVAUInputServiceEvidence
    weight_service: MVAUInputServiceEvidence
    output: MVAUOutputEvidence
    associations: MVAUAssociationEvidence
    cycles: MVAURTLCycleEvidence | None = None

    @property
    def semantic_contract_covered(self) -> bool:
        return (
            self.activation_service.complete
            and self.weight_service.complete
            and self.output.every_final_output_available
            and self.output.field_and_beat_order_preserved
            and self.associations.complete
        )


def _slots(sequence: BeatSequence) -> dict[Coordinate, tuple[tuple[int, int], ...]]:
    slots: dict[Coordinate, list[tuple[int, int]]] = defaultdict(list)
    for ordinal, beat in enumerate(sequence.beats):
        for field, position in enumerate(beat):
            slots[position].append((ordinal, field))
    return {position: tuple(values) for position, values in slots.items()}


def _activation_service(
    source_scope_id: str,
    region_id: str,
    interface: InputInterface,
    elaboration: MVAUPhysicalElaboration,
) -> MVAUInputServiceEvidence:
    slots = _slots(interface.port.beat_sequence)
    assignments = []
    complete = True
    for iteration, position, multiplicity in interface.requirements.occurrences:
        candidates = slots.get(position, ())
        if not candidates:
            complete = False
            continue
        ordinal, field = candidates[0]
        assignments.append(
            MVAURequirementServiceAssignment(iteration, position, multiplicity, ordinal, field)
        )
    replay_component_id = f"{source_scope_id}.compute.activation_replay"
    replay_present = any(
        component.id == replay_component_id for component in elaboration.components
    )
    return MVAUInputServiceEvidence(
        region_id,
        interface.port.id,
        "activation_boundary_with_local_replay",
        interface.port.beat_sequence,
        tuple(assignments),
        replay_component_id,
        replay_present and complete and len(assignments) == interface.requirements.occurrence_count,
    )


def _weight_service(
    region_id: str,
    interface: InputInterface,
    source_sequence: BeatSequence,
    source_kind: str,
) -> MVAUInputServiceEvidence:
    slots = _slots(source_sequence)
    next_slot: dict[Coordinate, int] = defaultdict(int)
    assignments = []
    complete = True
    for iteration, position, multiplicity in interface.requirements.occurrences:
        candidates = slots.get(position, ())
        index = next_slot[position]
        if index >= len(candidates):
            complete = False
            continue
        ordinal, field = candidates[index]
        next_slot[position] += 1
        assignments.append(
            MVAURequirementServiceAssignment(iteration, position, multiplicity, ordinal, field)
        )
    return MVAUInputServiceEvidence(
        region_id,
        interface.port.id,
        source_kind,
        source_sequence,
        tuple(assignments),
        None,
        complete and len(assignments) == interface.requirements.occurrence_count,
    )


def _compute_region(resolved: MVAUResolvedDesign) -> tuple[str, DataflowRegion]:
    if isinstance(resolved.result, RegionRef):
        return resolved.result.region_id, resolved.result.region
    return "compute", resolved.result.network.node("compute").region


def _output_evidence(
    region_id: str,
    region: DataflowRegion,
    artifact: MVAUBuiltRTLArtifact,
) -> MVAUOutputEvidence:
    output = region.output_interface("output")
    artifact_output = next(
        item for item in artifact.requirements.interfaces if item.interface_name == "out0_V"
    )
    return MVAUOutputEvidence(
        region_id,
        output.availability.entries,
        output.port.beat_sequence,
        artifact_output.beat_sequence,
        output.availability.domain == output.port.beat_sequence.image,
        artifact_output.beat_sequence == output.port.beat_sequence,
    )


def _association_evidence(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    compute_region_id: str,
    compute_region: DataflowRegion,
) -> MVAUAssociationEvidence:
    required_ports = tuple(
        sorted(
            MVAUSemanticPortRef(compute_region_id, interface.port.id)
            for interface in compute_region.interfaces
        )
    )
    required_edges: tuple[str, ...] = ()
    if isinstance(resolved.result, NetworkRef):
        delivery = resolved.result.network.node("delivery").region
        required_ports = tuple(
            sorted(
                (
                    *required_ports,
                    MVAUSemanticPortRef("delivery", delivery.output_interface("weight").port.id),
                )
            )
        )
        required_edges = tuple(edge.id for edge in resolved.result.network.edges)
    associated_ports = {
        port for association in elaboration.associations for port in association.semantic_ports
    }
    associated_edges = {
        edge for association in elaboration.associations for edge in association.semantic_edge_ids
    }
    required_source_owners = (
        resolved.result.source_association.source_node_id,
        *resolved.result.source_association.fused_source_node_ids,
    )
    required_regions: tuple[str, ...] = (compute_region_id,)
    if isinstance(resolved.result, NetworkRef):
        required_regions = tuple(node.id for node in resolved.result.network.nodes)
    associated_owners = {
        owner for association in elaboration.associations for owner in association.source_owner_ids
    }
    associated_regions = {
        region
        for association in elaboration.associations
        for region in association.semantic_region_ids
    }
    return MVAUAssociationEvidence(
        required_ports,
        tuple(port for port in required_ports if port not in associated_ports),
        required_edges,
        tuple(edge for edge in required_edges if edge not in associated_edges),
        required_source_owners,
        tuple(owner for owner in required_source_owners if owner not in associated_owners),
        required_regions,
        tuple(region for region in required_regions if region not in associated_regions),
    )


def collect_mvau_rtl_softvec_evidence(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    artifact: MVAUBuiltRTLArtifact,
    *,
    measured_cycles: int | None = None,
) -> MVAURTLSoftvecEvidence:
    """Collect evidence only for the emitted standard RTL soft-vector implementation."""
    if (
        elaboration.semantic_result != resolved.result
        or artifact.requirements.elaboration != elaboration
    ):
        raise ValueError("resolved point, elaboration, and artifact requirements must correspond")
    compute_region_id, region = _compute_region(resolved)
    activation = region.input_interface("activation")
    weight = region.input_interface("weight")
    activation_evidence = _activation_service(
        resolved.result.source_association.source_node_id,
        compute_region_id,
        activation,
        elaboration,
    )
    if isinstance(resolved.result, NetworkRef):
        delivery = resolved.result.network.node("delivery").region
        weight_sequence = delivery.output_interface("weight").port.beat_sequence
        weight_source = "cyclic_delivery_region"
    else:
        weight_sequence = weight.port.beat_sequence
        weight_source = "direct_weight_boundary"
    weight_evidence = _weight_service(
        compute_region_id,
        weight,
        weight_sequence,
        weight_source,
    )
    cycle_evidence = None
    if measured_cycles is not None:
        repetitions = cast(int, resolved.point.problem[MVAUComputeKernelPaths.REPETITIONS])
        matrix_width = cast(int, resolved.point.problem[MVAUComputeKernelPaths.MATRIX_WIDTH])
        matrix_height = cast(int, resolved.point.problem[MVAUComputeKernelPaths.MATRIX_HEIGHT])
        pe = cast(int, resolved.point.assignments[MVAUComputeKernelPaths.PE])
        simd = cast(int, resolved.point.assignments[MVAUComputeKernelPaths.SIMD])
        analytical = repetitions * (matrix_height // pe) * (matrix_width // simd)
        unsorted_configuration: tuple[tuple[str, bool | float | int | str], ...] = (
            ("clock_period_ns", elaboration.target_clock_period_ns),
            ("fpga_part", elaboration.target_fpga_part),
            ("matrix_height", matrix_height),
            ("matrix_width", matrix_width),
            ("mem_mode", artifact.requirements.mem_mode),
            ("pe", pe),
            ("repetitions", repetitions),
            ("simd", simd),
        )
        configuration = tuple(sorted(unsorted_configuration))
        cycle_evidence = MVAURTLCycleEvidence(
            "finn.xsi:MVAU_rtl.cycles_rtlsim",
            configuration,
            analytical,
            measured_cycles,
            abs(measured_cycles - analytical),
            abs(measured_cycles - analytical) <= 15,
        )
    return MVAURTLSoftvecEvidence(
        activation_evidence,
        weight_evidence,
        _output_evidence(compute_region_id, region, artifact),
        _association_evidence(resolved, elaboration, compute_region_id, region),
        cycle_evidence,
    )


__all__ = [
    "MVAUAssociationEvidence",
    "MVAUInputServiceEvidence",
    "MVAUOutputEvidence",
    "MVAURequirementServiceAssignment",
    "MVAURTLCycleEvidence",
    "MVAURTLSoftvecEvidence",
    "collect_mvau_rtl_softvec_evidence",
]
