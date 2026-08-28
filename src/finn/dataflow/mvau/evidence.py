# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Emitted-artifact evidence for the FINN RTL soft-vector MVAU slice."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import cast

import numpy as np  # type: ignore[import-not-found]

from finn.dataflow.mvau.artifacts import (
    MVAUBuiltRTLArtifact,
    MVAURTLSimulationObservation,
    MVAUStitchedSimulationObservation,
    mvau_built_artifact_identity,
)
from finn.dataflow.mvau.compute_kernels import SOFT_VECTOR_PATHS, MVAUComputeProblemPaths
from finn.dataflow.mvau.elaboration import MVAUPhysicalElaboration, MVAUSemanticPortRef
from finn.dataflow.mvau.source import MVAUResolvedDesign
from finn.dataflow.ops.mvau import NetworkRef, RegionRef
from finn.dataflow.region import BeatSequence, Coordinate, DataflowRegion, InputInterface


@dataclass(frozen=True, order=True)
class MVAURequirementServiceAssignment:
    """One declared requirement occurrence and its semantic source beat slot."""

    iteration: Coordinate
    operand_position: Coordinate
    multiplicity_index: int
    beat_ordinal: int
    field_ordinal: int


@dataclass(frozen=True)
class MVAUInputCorrespondenceEvidence:
    """Static correspondence between requirements and a selected boundary sequence."""

    region_id: str
    port_id: str
    source_kind: str
    source_sequence: BeatSequence
    assignments: tuple[MVAURequirementServiceAssignment, ...]
    implementation_parameters: tuple[tuple[str, int], ...]
    emitted_configuration_match: bool
    complete: bool


@dataclass(frozen=True)
class MVAUOutputCorrespondenceEvidence:
    """Static availability and generated-interface sequence correspondence."""

    region_id: str
    availability: tuple[tuple[Coordinate, Coordinate], ...]
    output_sequence: BeatSequence
    artifact_sequence: BeatSequence
    every_final_output_available: bool
    field_and_beat_order_preserved: bool


@dataclass(frozen=True)
class MVAUAssociationEvidence:
    """Coverage of semantic identities by the physical representation."""

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
class MVAUGeneratedArtifactEvidence:
    """Checks against generated wrappers, parameters, wiring, and weight data."""

    compute_parameters: tuple[tuple[str, int], ...]
    compute_parameters_match: bool
    compute_wiring_match: bool
    replay_instantiation_present: bool
    memstream_parameters: tuple[tuple[str, int | str], ...] = ()
    memstream_parameters_match: bool = True
    initialized_weight_sequence_match: bool | None = None

    @property
    def complete(self) -> bool:
        return (
            self.compute_parameters_match
            and self.compute_wiring_match
            and self.replay_instantiation_present
            and self.memstream_parameters_match
            and self.initialized_weight_sequence_match is not False
        )


@dataclass(frozen=True)
class MVAURTLCycleEvidence:
    """A configuration-qualified observation, not a universal latency model."""

    oracle: str
    configuration: tuple[tuple[str, bool | float | int | str], ...]
    analytical_work_cycles: int
    measured_cycles: int
    measured_overhead_cycles: int


@dataclass(frozen=True)
class MVAURTLSoftvecEvidence:
    """Evidence scoped only to the emitted FINN RTL soft-vector implementation."""

    activation_service: MVAUInputCorrespondenceEvidence
    weight_service: MVAUInputCorrespondenceEvidence
    output: MVAUOutputCorrespondenceEvidence
    associations: MVAUAssociationEvidence
    generated_artifact: MVAUGeneratedArtifactEvidence
    requires_stitched_observation: bool
    simulation: MVAURTLSimulationObservation | None = None
    stitched_simulation: MVAUStitchedSimulationObservation | None = None
    cycles: MVAURTLCycleEvidence | None = None

    @property
    def static_correspondence_complete(self) -> bool:
        return (
            self.activation_service.complete
            and self.activation_service.emitted_configuration_match
            and self.weight_service.complete
            and self.output.every_final_output_available
            and self.output.field_and_beat_order_preserved
            and self.associations.complete
            and self.generated_artifact.complete
        )

    @property
    def emitted_realization_observed(self) -> bool:
        return (
            self.static_correspondence_complete
            and self.simulation is not None
            and self.simulation.numerical_match
            and self.simulation.output_order_match
            and (
                not self.requires_stitched_observation
                or (
                    self.stitched_simulation is not None
                    and self.stitched_simulation.numerical_match
                )
            )
            and self.cycles is not None
        )


def _slots(sequence: BeatSequence) -> dict[Coordinate, tuple[tuple[int, int], ...]]:
    slots: dict[Coordinate, list[tuple[int, int]]] = defaultdict(list)
    for ordinal, beat in enumerate(sequence.beats):
        for field, position in enumerate(beat):
            slots[position].append((ordinal, field))
    return {position: tuple(values) for position, values in slots.items()}


def _integer_parameter(parameters: tuple[tuple[str, object], ...], name: str) -> int:
    value = dict(parameters)[name]
    if type(value) is not int:
        raise TypeError(f"physical parameter {name} must be an integer")
    return value


def _activation_service(
    source_scope_id: str,
    region_id: str,
    interface: InputInterface,
    elaboration: MVAUPhysicalElaboration,
) -> MVAUInputCorrespondenceEvidence:
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
    shell = elaboration.component(f"{source_scope_id}.compute.stream_shell")
    expected = (
        (
            "LEN",
            interface.port.operand.shape[-1] // interface.port.beat_sequence.elements_per_beat,
        ),
        ("REP", max(1, len(assignments) // interface.port.operand.position_count)),
        ("W", interface.port.logical_beat_bits),
    )
    actual = (
        ("LEN", _integer_parameter(shell.parameters, "ACTIVATION_REPLAY_LEN")),
        ("REP", _integer_parameter(shell.parameters, "ACTIVATION_REPLAY_REP")),
        ("W", _integer_parameter(shell.parameters, "ACTIVATION_REPLAY_WIDTH")),
    )
    return MVAUInputCorrespondenceEvidence(
        region_id,
        interface.port.id,
        "activation_boundary_with_declared_replay",
        interface.port.beat_sequence,
        tuple(assignments),
        actual,
        actual == expected,
        complete and len(assignments) == interface.requirements.occurrence_count,
    )


def _weight_service(
    region_id: str,
    interface: InputInterface,
    source_sequence: BeatSequence,
    source_kind: str,
) -> MVAUInputCorrespondenceEvidence:
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
    return MVAUInputCorrespondenceEvidence(
        region_id,
        interface.port.id,
        source_kind,
        source_sequence,
        tuple(assignments),
        (),
        True,
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
) -> MVAUOutputCorrespondenceEvidence:
    output = region.output_interface("output")
    artifact_output = next(
        item for item in artifact.requirements.interfaces if item.interface_name == "out0_V"
    )
    return MVAUOutputCorrespondenceEvidence(
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


def _generated_file(artifact: MVAUBuiltRTLArtifact, output_id: str) -> Path:
    requirement = next(
        item for item in artifact.requirements.generated_outputs if item.id == output_id
    )
    return Path(artifact.output_directory) / requirement.relative_path


def _source_dependency(artifact: MVAUBuiltRTLArtifact, source_id: str) -> Path:
    requirement = next(
        item for item in artifact.requirements.source_dependencies if item.id == source_id
    )
    return Path(artifact.output_directory) / "declared_sources" / requirement.relative_path


def _verilog_parameter(text: str, name: str) -> str | None:
    match = re.search(rf"parameter\s+{re.escape(name)}\s*=\s*([^,\n]+)", text)
    return None if match is None else match.group(1).strip().strip('"')


def _generated_artifact_evidence(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    artifact: MVAUBuiltRTLArtifact,
) -> MVAUGeneratedArtifactEvidence:
    wrapper_text = _generated_file(artifact, "compute.wrapper").read_text()
    shell_text = _source_dependency(artifact, "compute.library.1").read_text()
    wrapper = elaboration.component(
        f"{resolved.result.source_association.source_node_id}.compute.wrapper"
    )
    parameter_names = {
        "ACCU_WIDTH",
        "ACTIVATION_WIDTH",
        "IS_MVU",
        "MH",
        "MW",
        "NARROW_WEIGHTS",
        "PE",
        "PUMPED_COMPUTE",
        "SEGMENTLEN",
        "SIGNED_ACTIVATIONS",
        "SIMD",
        "VERSION",
        "WEIGHT_WIDTH",
    }
    expected_compute = tuple(
        (name, int(value)) for name, value in wrapper.parameters if name in parameter_names
    )
    observed_compute = tuple(
        (name, int(float(cast(str, _verilog_parameter(wrapper_text, name)))))
        for name, _value in expected_compute
        if _verilog_parameter(wrapper_text, name) is not None
    )
    compute_wiring = all(
        fragment in wrapper_text
        for fragment in (
            ".ap_clk(ap_clk)",
            ".ap_clk2x(ap_clk2x)",
            ".ap_rst_n(ap_rst_n)",
            ".s_axis_weights_tdata(in1_V_TDATA)",
            ".s_axis_weights_tvalid(in1_V_TVALID)",
            ".s_axis_weights_tready(in1_V_TREADY)",
            ".s_axis_input_tdata(in0_V_TDATA)",
            ".s_axis_input_tvalid(in0_V_TVALID)",
            ".s_axis_input_tready(in0_V_TREADY)",
            ".m_axis_output_tdata(out0_V_TDATA)",
            ".m_axis_output_tvalid(out0_V_TVALID)",
            ".m_axis_output_tready(out0_V_TREADY)",
        )
    )
    replay_instantiation = all(
        fragment in shell_text
        for fragment in (
            "replay_buffer #(.LEN(SF)",
            ".REP(IS_MVU ? NF : 1)",
            ".W($bits(mvu_flatin_t))",
        )
    )
    memstream_parameters: tuple[tuple[str, int | str], ...] = ()
    memstream_parameters_match = True
    initialized_weight_sequence_match = None
    if isinstance(resolved.result, NetworkRef):
        delivery = elaboration.component(
            f"{resolved.result.source_association.source_node_id}.delivery.wrapper"
        )
        memstream_names = {"DEPTH", "INIT_FILE", "PUMPED_MEMORY", "RAM_STYLE", "SETS", "WIDTH"}
        expected_memstream = tuple(
            (name, int(value) if type(value) is bool else cast(int | str, value))
            for name, value in delivery.parameters
            if name in memstream_names
        )
        memstream_text = _generated_file(artifact, "delivery.wrapper").read_text()
        observed_values: list[tuple[str, int | str]] = []
        for name, expected in expected_memstream:
            raw = _verilog_parameter(memstream_text, name)
            if raw is None:
                continue
            observed = int(raw) if type(expected) is int else raw
            if name == "INIT_FILE":
                observed = Path(cast(str, observed)).name
            observed_values.append((name, observed))
        memstream_parameters = tuple(observed_values)
        memstream_parameters_match = memstream_parameters == expected_memstream
        weight_data = artifact.requirements.weight_initializer
        if weight_data is not None:
            initializer_path = _generated_file(artifact, "delivery.simulation_weights")
            sequence = (
                resolved.result.network.node("delivery")
                .region.output_interface("weight")
                .port.beat_sequence
            )
            stored = np.load(initializer_path).reshape(-1, sequence.elements_per_beat)
            repeats = sequence.beat_count // len(stored)
            observed_sequence = np.tile(stored, (repeats, 1))
            source_weights = weight_data.as_array()
            expected_sequence = np.asarray(
                [[source_weights[mw, mh] for mh, mw in beat] for beat in sequence.beats],
                dtype=np.float32,
            )
            initialized_weight_sequence_match = np.array_equal(observed_sequence, expected_sequence)
    return MVAUGeneratedArtifactEvidence(
        observed_compute,
        observed_compute == expected_compute,
        compute_wiring,
        replay_instantiation,
        memstream_parameters,
        memstream_parameters_match,
        initialized_weight_sequence_match,
    )


def collect_mvau_rtl_softvec_evidence(
    resolved: MVAUResolvedDesign,
    elaboration: MVAUPhysicalElaboration,
    artifact: MVAUBuiltRTLArtifact,
    *,
    simulation: MVAURTLSimulationObservation | None = None,
    stitched_simulation: MVAUStitchedSimulationObservation | None = None,
) -> MVAURTLSoftvecEvidence:
    """Collect evidence only for the emitted standard RTL soft-vector implementation."""
    if (
        elaboration.semantic_result != resolved.result
        or artifact.requirements.elaboration != elaboration
    ):
        raise ValueError("resolved point, elaboration, and artifact requirements must correspond")
    artifact_identity = mvau_built_artifact_identity(artifact)
    if simulation is not None and simulation.artifact_identity != artifact_identity:
        raise ValueError("node-level simulation observation belongs to another artifact")
    if (
        stitched_simulation is not None
        and stitched_simulation.artifact_identity != artifact_identity
    ):
        raise ValueError("stitched simulation observation belongs to another artifact")
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
    if simulation is not None:
        repetitions = cast(int, resolved.point.problem[MVAUComputeProblemPaths.REPETITIONS])
        matrix_width = cast(int, resolved.point.problem[MVAUComputeProblemPaths.MATRIX_WIDTH])
        matrix_height = cast(int, resolved.point.problem[MVAUComputeProblemPaths.MATRIX_HEIGHT])
        pe = cast(int, resolved.point.assignments[SOFT_VECTOR_PATHS.pe])
        simd = cast(int, resolved.point.assignments[SOFT_VECTOR_PATHS.simd])
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
        cycle_evidence = MVAURTLCycleEvidence(
            f"{simulation.oracle}:cycles_rtlsim",
            tuple(sorted(unsorted_configuration)),
            analytical,
            simulation.measured_cycles,
            simulation.measured_cycles - analytical,
        )
    return MVAURTLSoftvecEvidence(
        activation_evidence,
        weight_evidence,
        _output_evidence(compute_region_id, region, artifact),
        _association_evidence(resolved, elaboration, compute_region_id, region),
        _generated_artifact_evidence(resolved, elaboration, artifact),
        isinstance(resolved.result, NetworkRef),
        simulation,
        stitched_simulation,
        cycle_evidence,
    )


__all__ = [
    "MVAUAssociationEvidence",
    "MVAUGeneratedArtifactEvidence",
    "MVAUInputCorrespondenceEvidence",
    "MVAUOutputCorrespondenceEvidence",
    "MVAURequirementServiceAssignment",
    "MVAURTLCycleEvidence",
    "MVAURTLSoftvecEvidence",
    "collect_mvau_rtl_softvec_evidence",
]
