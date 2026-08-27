# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat MVAU ``DataflowOp`` assembly over compute and cyclic-delivery Kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd, prod
from typing import cast

from finn.dataflow.design import (
    ABSENT,
    DATAFLOW_NETWORK_SEMANTICS,
    DATAFLOW_REGION_SEMANTICS,
    NETWORK_VALIDATION_REPORT_SEMANTICS,
    AbsenceMode,
    Absent,
    Answer,
    Constraint,
    ConstraintSet,
    Decided,
    Decision,
    DecisionDomain,
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    FindingKind,
    ProblemField,
    ProblemSchema,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernel import (
    KernelDefinition,
    assemble_kernel_specs,
    gate_design_space_spec,
)
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.definition import MVAU_COMPUTE_KERNEL, MVAUComputeKernelPaths
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    construct_batch_interleaved_mvau_weight_port,
    construct_standard_mvau_weight_port,
)
from finn.dataflow.mvau.weight_adapter import (
    construct_weight_sequence_adapter_region,
    weight_sequence_adapter_applicable,
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
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL,
    CyclicParameterKernelPaths,
    build_cyclic_parameter_kernel_spec,
)
from finn.dataflow.region import BeatSequence, DataflowRegion, NumericElementType, Port


class MVAUParameterTopology(str, Enum):
    """Initial op-level parameter-delivery topology alternatives."""

    EMBEDDED = "embedded"
    DIRECT = "direct"
    CYCLIC = "cyclic"


class MVAUWeightDeliveryDeclaration(str, Enum):
    """Independently selected cyclic-delivery weight boundaries."""

    STANDARD_FULL_TILE = "standard.full_tile"
    BATCH_INTERLEAVED_CHUNKED = "batch_interleaved.chunked"


class MVAUConnectionTopology(str, Enum):
    """Semantic connection alternatives between delivery and compute."""

    DIRECT = "direct"
    ADAPTER = "adapter"


class CoordinateMappingKind(str, Enum):
    """Explicit source-to-region coordinate transformations used by MVAU."""

    FLATTEN_LEADING = "flatten_leading"
    TRANSPOSE_2D = "transpose_2d"
    BINDING_LOCAL_STATE = "binding_local_state"


@dataclass(frozen=True)
class SemanticOperandDestination:
    """Qualified operand destination in a selected region or network node."""

    owner_id: str
    operand_id: str


@dataclass(frozen=True)
class BindingLocalStateDestination:
    """Qualified binding-local state destination."""

    owner_id: str
    state_id: str


SourceOperandDestination = SemanticOperandDestination | BindingLocalStateDestination


@dataclass(frozen=True)
class MVAUSourceDescription:
    """Compiler-owned source identities and shapes for one MVAU scope."""

    source_node_id: str
    activation_operand_id: str
    weight_operand_id: str
    output_operand_id: str
    leading_shape: tuple[int, ...]
    threshold_operand_id: str | None = None
    fused_source_node_ids: tuple[str, ...] = ()
    threshold_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "leading_shape", tuple(self.leading_shape))
        object.__setattr__(self, "fused_source_node_ids", tuple(self.fused_source_node_ids))
        if self.threshold_shape is not None:
            object.__setattr__(self, "threshold_shape", tuple(self.threshold_shape))


@dataclass(frozen=True)
class SourceOperandAssociation:
    """Explicit association from one source operand to a semantic or binding target."""

    role: str
    source_operand_id: str
    destination: SourceOperandDestination
    mapping: CoordinateMappingKind
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]

    def map_position(self, position: tuple[int, ...]) -> tuple[int, ...]:
        if len(position) != len(self.source_shape) or any(
            index < 0 or index >= extent for index, extent in zip(position, self.source_shape)
        ):
            raise ValueError("source position is outside source shape")
        if self.mapping is CoordinateMappingKind.FLATTEN_LEADING:
            leading = position[:-1]
            flattened = 0
            for index, extent in zip(leading, self.source_shape[:-1]):
                flattened = flattened * extent + index
            return (flattened, position[-1])
        if self.mapping is CoordinateMappingKind.TRANSPOSE_2D:
            if len(position) != 2:
                raise ValueError("transpose mapping requires a rank-two position")
            return (position[1], position[0])
        return position


@dataclass(frozen=True)
class MVAUSourceAssociation:
    """Source provenance and operand mappings for one selected MVAU result."""

    source_node_id: str
    fused_source_node_ids: tuple[str, ...]
    region_declaration_id: str
    parameter_topology: MVAUParameterTopology
    operands: tuple[SourceOperandAssociation, ...]


@dataclass(frozen=True)
class RegionRef:
    """Tagged selected-region result associated with one source operation."""

    region_id: str
    region: DataflowRegion
    source_association: MVAUSourceAssociation


@dataclass(frozen=True)
class NetworkRef:
    """Tagged selected-network result associated with one source operation."""

    network_id: str
    network: DataflowNetwork
    source_association: MVAUSourceAssociation


DataflowOpResult = RegionRef | NetworkRef


class MVAUDataflowOpPaths:
    """Stable paths owned by the MVAU source-operation assembly."""

    SOURCE_DESCRIPTION = QualifiedPath("problem.mvau.source_description")
    EXTERNAL_WEIGHT_SEQUENCE = QualifiedPath("problem.mvau.external_weight_sequence")
    TARGET_FPGA_PART = QualifiedPath("problem.target.fpga_part")
    TARGET_CLOCK_PERIOD_NS = QualifiedPath("problem.target.clock_period_ns")
    PARAMETER_TOPOLOGY = QualifiedPath("mvau.op.parameter_topology")
    DELIVERY_PE = QualifiedPath("mvau.op.delivery.pe")
    DELIVERY_SIMD = QualifiedPath("mvau.op.delivery.simd")
    DELIVERY_DECLARATION = QualifiedPath("mvau.op.delivery.weight_sequence")
    DELIVERY_INTERLEAVE = QualifiedPath("mvau.op.delivery.interleave")
    CONNECTION_TOPOLOGY = QualifiedPath("mvau.op.weight_connection")

    COMPUTE_WEIGHT_PORT = QualifiedPath("semantic.mvau.op.compute_weight_port")
    DELIVERY_WEIGHT_PORT = QualifiedPath("semantic.mvau.op.delivery_weight_port")
    WEIGHT_ADAPTER_REGION = QualifiedPath("semantic.mvau.op.weight_adapter_region")
    SOURCE_ASSOCIATION = QualifiedPath("semantic.mvau.op.source_association")
    NETWORK = QualifiedPath("semantic.mvau.op.network")
    NETWORK_VALIDATION = QualifiedPath("semantic.mvau.op.network_validation")
    RESULT = QualifiedPath("semantic.mvau.op.result")

    TOPOLOGY_MATCHES_REGION = QualifiedPath("constraint.mvau.op.topology_matches_region")
    CYCLIC_INTERLEAVED_PUMPING_SUPPORTED = QualifiedPath(
        "constraint.mvau.op.cyclic_interleaved_pumping_supported"
    )
    DIRECT_INTERLEAVED_SOURCE_AVAILABLE = QualifiedPath(
        "constraint.mvau.op.direct_interleaved_source_available"
    )
    WEIGHT_CONNECTION_SUPPORTED = QualifiedPath("constraint.mvau.op.weight_connection_supported")
    SOURCE_ASSOCIATION_VALID = QualifiedPath("constraint.mvau.op.source_association_valid")
    NETWORK_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.op.network_structurally_well_formed"
    )


_INTEGER_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_FLOAT_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(float, name="float"))
_STRING_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(str, name="string"))
_BOOL_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))
_ELEMENT_TYPE_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(NumericElementType, name="NumericElementType")
)
_COMPUTATION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUComputationProfile, name="MVAUComputationProfile")
)
_SOURCE_DESCRIPTION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUSourceDescription, name="MVAUSourceDescription")
)
_SOURCE_ASSOCIATION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUSourceAssociation, name="MVAUSourceAssociation")
)
_PORT_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
_BEAT_SEQUENCE_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(BeatSequence, name="BeatSequence")
)
_REGION_SEMANTICS = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_NETWORK_SEMANTICS = as_object_semantics(DATAFLOW_NETWORK_SEMANTICS)
_NETWORK_REPORT_SEMANTICS = as_object_semantics(NETWORK_VALIDATION_REPORT_SEMANTICS)
_RESULT_SEMANTICS: ValueSemantics[object] = ValueSemantics(
    DataflowOpResult,
    "DataflowOpResult",
    lambda value: type(value) in {RegionRef, NetworkRef},
    lambda left, right: left == right,
    lambda value: value,
)
_TOPOLOGY_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUParameterTopology, name="MVAUParameterTopology")
)
_DELIVERY_DECLARATION_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(
        MVAUWeightDeliveryDeclaration, name="MVAUWeightDeliveryDeclaration"
    )
)
_CONNECTION_TOPOLOGY_SEMANTICS = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUConnectionTopology, name="MVAUConnectionTopology")
)


def _source_description_valid(value: object) -> bool:
    if type(value) is not MVAUSourceDescription:
        return False
    description = value
    names: tuple[object, ...] = (
        description.source_node_id,
        description.activation_operand_id,
        description.weight_operand_id,
        description.output_operand_id,
        *description.fused_source_node_ids,
    )
    threshold_valid = description.threshold_operand_id is None or (
        isinstance(description.threshold_operand_id, str) and bool(description.threshold_operand_id)
    )
    threshold_shape_valid = description.threshold_shape is None or all(
        type(extent) is int and extent > 0 for extent in description.threshold_shape
    )
    threshold_pair_valid = (description.threshold_operand_id is None) == (
        description.threshold_shape is None
    )
    return (
        all(isinstance(name, str) and bool(name) for name in names)
        and threshold_valid
        and threshold_shape_valid
        and threshold_pair_valid
        and all(type(extent) is int and extent > 0 for extent in description.leading_shape)
    )


def _positive_float(value: object) -> bool:
    return type(value) is float and value > 0


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


def _divisor_domain(dimension: QualifiedPath) -> DecisionDomain:
    dependency = DependencyRef.problem("dimension", dimension, _INTEGER_SEMANTICS)

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        extent = cast(int, dependencies["dimension"])
        return Decided(type(value) is int and value > 0 and extent % value == 0)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        extent = cast(int, dependencies["dimension"])
        return Decided(tuple(value for value in range(1, extent + 1) if extent % value == 0))

    return DecisionDomain((dependency,), accepts, EvaluatorSpec((dependency,), candidates))


_TOPOLOGY_REF = DependencyRef.decision(
    "op_topology", MVAUDataflowOpPaths.PARAMETER_TOPOLOGY, _TOPOLOGY_SEMANTICS
)
_DELIVERY_PE_REF = DependencyRef.decision(
    "delivery_pe", MVAUDataflowOpPaths.DELIVERY_PE, _INTEGER_SEMANTICS
)
_DELIVERY_SIMD_REF = DependencyRef.decision(
    "delivery_simd", MVAUDataflowOpPaths.DELIVERY_SIMD, _INTEGER_SEMANTICS
)
_DELIVERY_DECLARATION_REF = DependencyRef.decision(
    "delivery_declaration",
    MVAUDataflowOpPaths.DELIVERY_DECLARATION,
    _DELIVERY_DECLARATION_SEMANTICS,
)
_DELIVERY_INTERLEAVE_REF = DependencyRef.decision(
    "delivery_interleave",
    MVAUDataflowOpPaths.DELIVERY_INTERLEAVE,
    _INTEGER_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_CONNECTION_TOPOLOGY_REF = DependencyRef.decision(
    "connection_topology",
    MVAUDataflowOpPaths.CONNECTION_TOPOLOGY,
    _CONNECTION_TOPOLOGY_SEMANTICS,
)
_REGION_DECLARATION_REF = DependencyRef.decision(
    "region_declaration",
    MVAUComputeKernelPaths.REGION_DECLARATION,
    MVAU_COMPUTE_KERNEL.spec.decisions[2].value_semantics,
)
_COMPUTE_PE_REF = DependencyRef.decision(
    "compute_pe", MVAUComputeKernelPaths.PE, _INTEGER_SEMANTICS
)
_COMPUTE_SIMD_REF = DependencyRef.decision(
    "compute_simd", MVAUComputeKernelPaths.SIMD, _INTEGER_SEMANTICS
)
_COMPUTE_INTERLEAVE_REF = DependencyRef.decision(
    "compute_interleave",
    MVAUComputeKernelPaths.INTERLEAVE,
    _INTEGER_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_COMPUTE_REGION_DECLARATION_FOR_SCOPE_REF = DependencyRef.decision(
    "compute_region_declaration",
    MVAUComputeKernelPaths.REGION_DECLARATION,
    MVAU_COMPUTE_KERNEL.spec.decisions[2].value_semantics,
)
_COMPUTE_REGION_REF = DependencyRef.property(
    "compute_region", MVAUComputeKernelPaths.REGION, _REGION_SEMANTICS
)
_COMPUTE_WEIGHT_PORT_REF = DependencyRef.property(
    "compute_weight_port", MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT, _PORT_SEMANTICS
)
_DELIVERY_WEIGHT_PORT_REF = DependencyRef.property(
    "delivery_weight_port", MVAUDataflowOpPaths.DELIVERY_WEIGHT_PORT, _PORT_SEMANTICS
)
_COMPUTATION_REF = DependencyRef.problem(
    "computation_profile",
    MVAUComputeKernelPaths.COMPUTATION_PROFILE,
    _COMPUTATION_SEMANTICS,
)
_DELIVERY_REGION_REF = DependencyRef.property(
    "delivery_region",
    CyclicParameterKernelPaths.REGION,
    _REGION_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_NETWORK_REF = DependencyRef.property(
    "network",
    MVAUDataflowOpPaths.NETWORK,
    _NETWORK_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_ADAPTER_REGION_REF = DependencyRef.property(
    "adapter_region",
    MVAUDataflowOpPaths.WEIGHT_ADAPTER_REGION,
    _REGION_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)


def _topology_applies(topology: MVAUParameterTopology) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(dependencies["op_topology"] is topology)

    return EvaluatorSpec((_TOPOLOGY_REF,), evaluate)


def _cyclic_streamed_region_applies(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        dependencies["op_topology"] is MVAUParameterTopology.CYCLIC
        and dependencies["compute_region_declaration"]
        is not MVAURegionDeclaration.STANDARD_EMBEDDED
    )


_CYCLIC_STREAMED_REGION_APPLICABILITY = EvaluatorSpec(
    (_TOPOLOGY_REF, _COMPUTE_REGION_DECLARATION_FOR_SCOPE_REF),
    _cyclic_streamed_region_applies,
)


def _delivery_chunked_applies(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        dependencies["op_topology"] is MVAUParameterTopology.CYCLIC
        and dependencies["compute_region_declaration"]
        is not MVAURegionDeclaration.STANDARD_EMBEDDED
        and dependencies["delivery_declaration"]
        is MVAUWeightDeliveryDeclaration.BATCH_INTERLEAVED_CHUNKED
    )


_DELIVERY_CHUNKED_APPLICABILITY = EvaluatorSpec(
    (
        _TOPOLOGY_REF,
        _COMPUTE_REGION_DECLARATION_FOR_SCOPE_REF,
        _DELIVERY_DECLARATION_REF,
    ),
    _delivery_chunked_applies,
)


def _connection_applies(topology: MVAUConnectionTopology) -> EvaluatorSpec[Answer[bool]]:
    def evaluate(dependencies: DependencyView) -> Answer[bool]:
        return Decided(
            dependencies["op_topology"] is MVAUParameterTopology.CYCLIC
            and dependencies["compute_region_declaration"]
            is not MVAURegionDeclaration.STANDARD_EMBEDDED
            and dependencies["connection_topology"] is topology
        )

    return EvaluatorSpec(
        (
            _TOPOLOGY_REF,
            _COMPUTE_REGION_DECLARATION_FOR_SCOPE_REF,
            _CONNECTION_TOPOLOGY_REF,
        ),
        evaluate,
    )


_ADAPTER_CONNECTION_APPLICABILITY = _connection_applies(MVAUConnectionTopology.ADAPTER)


def _delivery_interleave_domain() -> DecisionDomain:
    repetitions = DependencyRef.problem(
        "repetitions", MVAUComputeKernelPaths.REPETITIONS, _INTEGER_SEMANTICS
    )

    def accepts(value: object, dependencies: DependencyView) -> Answer[bool]:
        if type(value) is not int or value <= 1:
            return Decided(False)
        repeated = cast(int, dependencies["repetitions"])
        tile = cast(int, dependencies["delivery_pe"]) * cast(int, dependencies["delivery_simd"])
        return Decided(repeated % value == 0 and tile % value == 0)

    def candidates(dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        repeated = cast(int, dependencies["repetitions"])
        tile = cast(int, dependencies["delivery_pe"]) * cast(int, dependencies["delivery_simd"])
        limit = gcd(repeated, tile)
        return Decided(tuple(value for value in range(2, limit + 1) if limit % value == 0))

    dependencies = (repetitions, _DELIVERY_PE_REF, _DELIVERY_SIMD_REF)
    return DecisionDomain(dependencies, accepts, EvaluatorSpec(dependencies, candidates))


def _direct_interleaved_applies(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        dependencies["op_topology"] is MVAUParameterTopology.DIRECT
        and dependencies["region_declaration"] is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
    )


def _derive_compute_weight_port(dependencies: DependencyView) -> Answer[object]:
    region = cast(DataflowRegion, dependencies["compute_region"])
    return Decided(region.input_interface("weight").port)


def _derive_delivery_weight_port(dependencies: DependencyView) -> Answer[object]:
    declaration = cast(MVAUWeightDeliveryDeclaration, dependencies["delivery_declaration"])
    repetitions = cast(int, dependencies["repetitions"])
    matrix_width = cast(int, dependencies["matrix_width"])
    matrix_height = cast(int, dependencies["matrix_height"])
    weight_type = cast(NumericElementType, dependencies["weight_element_type"])
    pe = cast(int, dependencies["delivery_pe"])
    simd = cast(int, dependencies["delivery_simd"])
    if declaration is MVAUWeightDeliveryDeclaration.STANDARD_FULL_TILE:
        return Decided(
            construct_standard_mvau_weight_port(
                repetitions, matrix_width, matrix_height, weight_type, pe, simd
            )
        )
    interleave = dependencies["delivery_interleave"]
    if interleave is ABSENT:
        raise AssertionError("chunked delivery requires an interleave decision")
    return Decided(
        construct_batch_interleaved_mvau_weight_port(
            repetitions,
            matrix_width,
            matrix_height,
            weight_type,
            pe,
            simd,
            cast(int, interleave),
        )
    )


def _adapter_family_supported(dependencies: DependencyView) -> bool:
    delivery_declaration = cast(MVAUWeightDeliveryDeclaration, dependencies["delivery_declaration"])
    compute_declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    same_base_tile = (
        dependencies["delivery_pe"] == dependencies["compute_pe"]
        and dependencies["delivery_simd"] == dependencies["compute_simd"]
    )
    full_to_chunked = (
        delivery_declaration is MVAUWeightDeliveryDeclaration.STANDARD_FULL_TILE
        and compute_declaration is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
    )
    chunked_to_full = (
        delivery_declaration is MVAUWeightDeliveryDeclaration.BATCH_INTERLEAVED_CHUNKED
        and compute_declaration is MVAURegionDeclaration.STANDARD_STREAMED
    )
    return same_base_tile and (full_to_chunked or chunked_to_full)


def _derive_weight_adapter_region(dependencies: DependencyView) -> Answer[object]:
    source = cast(Port, dependencies["delivery_weight_port"])
    sink = cast(Port, dependencies["compute_weight_port"])
    if not _adapter_family_supported(dependencies) or not weight_sequence_adapter_applicable(
        source, sink
    ):
        return Absent(
            (
                Finding(
                    FindingKind.REJECTION,
                    "mvau-weight-adapter-not-applicable",
                    MVAUDataflowOpPaths.WEIGHT_ADAPTER_REGION,
                    "the first adapter covers only equal-base-tile full/chunked conversion",
                ),
            )
        )
    return Decided(construct_weight_sequence_adapter_region(source, sink))


def _weight_connection_supported(dependencies: DependencyView) -> Answer[bool]:
    source = cast(Port, dependencies["delivery_weight_port"])
    sink = cast(Port, dependencies["compute_weight_port"])
    topology = cast(MVAUConnectionTopology, dependencies["connection_topology"])
    if topology is MVAUConnectionTopology.DIRECT:
        return Decided(
            source.operand.element_type == sink.operand.element_type
            and source.beat_sequence.image == sink.beat_sequence.image
            and source.beat_sequence == sink.beat_sequence
        )
    return Decided(
        _adapter_family_supported(dependencies) and weight_sequence_adapter_applicable(source, sink)
    )


def _derive_source_association(dependencies: DependencyView) -> Answer[object]:
    description = cast(MVAUSourceDescription, dependencies["source_description"])
    topology = cast(MVAUParameterTopology, dependencies["op_topology"])
    declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    repetitions = cast(int, dependencies["repetitions"])
    matrix_width = cast(int, dependencies["matrix_width"])
    matrix_height = cast(int, dependencies["matrix_height"])
    compute_owner = "compute" if topology is MVAUParameterTopology.CYCLIC else "mvau.compute"
    operands = [
        SourceOperandAssociation(
            "activation",
            description.activation_operand_id,
            SemanticOperandDestination(compute_owner, "X"),
            CoordinateMappingKind.FLATTEN_LEADING,
            (*description.leading_shape, matrix_width),
            (repetitions, matrix_width),
        ),
        SourceOperandAssociation(
            "output",
            description.output_operand_id,
            SemanticOperandDestination(compute_owner, "Y"),
            CoordinateMappingKind.FLATTEN_LEADING,
            (*description.leading_shape, matrix_height),
            (repetitions, matrix_height),
        ),
    ]
    if topology is MVAUParameterTopology.EMBEDDED:
        weight_destination: SourceOperandDestination = BindingLocalStateDestination(
            "mvau.compute", "weights"
        )
    elif topology is MVAUParameterTopology.DIRECT:
        weight_destination = SemanticOperandDestination("mvau.compute", "W")
    else:
        weight_destination = BindingLocalStateDestination("delivery", "weights")
    operands.insert(
        1,
        SourceOperandAssociation(
            "weight",
            description.weight_operand_id,
            weight_destination,
            CoordinateMappingKind.TRANSPOSE_2D,
            (matrix_width, matrix_height),
            (matrix_height, matrix_width),
        ),
    )
    if (
        profile is MVAUComputationProfile.FUSED_THRESHOLD
        and description.threshold_operand_id is not None
        and description.threshold_shape is not None
    ):
        operands.append(
            SourceOperandAssociation(
                "threshold",
                description.threshold_operand_id,
                BindingLocalStateDestination(compute_owner, "thresholds"),
                CoordinateMappingKind.BINDING_LOCAL_STATE,
                description.threshold_shape,
                description.threshold_shape,
            )
        )
    return Decided(
        MVAUSourceAssociation(
            description.source_node_id,
            description.fused_source_node_ids,
            declaration.value,
            topology,
            tuple(operands),
        )
    )


def _source_association_valid(dependencies: DependencyView) -> Answer[bool]:
    description = cast(MVAUSourceDescription, dependencies["source_description"])
    association = cast(MVAUSourceAssociation, dependencies["source_association"])
    topology = cast(MVAUParameterTopology, dependencies["op_topology"])
    compute = cast(DataflowRegion, dependencies["compute_region"])
    repetitions = cast(int, dependencies["repetitions"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    threshold_valid = profile is not MVAUComputationProfile.FUSED_THRESHOLD or (
        description.threshold_operand_id is not None and description.threshold_shape is not None
    )
    compute_operands = {interface.port.operand.id for interface in compute.interfaces}
    semantic_destinations_valid = True
    for operand in association.operands:
        destination = operand.destination
        if not isinstance(destination, SemanticOperandDestination):
            continue
        expected_owner = "compute" if topology is MVAUParameterTopology.CYCLIC else "mvau.compute"
        if destination.owner_id != expected_owner or destination.operand_id not in compute_operands:
            semantic_destinations_valid = False
    local_destinations = {
        operand.role: operand.destination
        for operand in association.operands
        if isinstance(operand.destination, BindingLocalStateDestination)
    }
    weight_destination_valid = (
        (
            topology is MVAUParameterTopology.EMBEDDED
            and local_destinations.get("weight")
            == BindingLocalStateDestination("mvau.compute", "weights")
        )
        or (topology is MVAUParameterTopology.DIRECT and "weight" not in local_destinations)
        or (
            topology is MVAUParameterTopology.CYCLIC
            and local_destinations.get("weight")
            == BindingLocalStateDestination("delivery", "weights")
        )
    )
    return Decided(
        prod(description.leading_shape) == repetitions
        and threshold_valid
        and semantic_destinations_valid
        and weight_destination_valid
    )


def _topology_matches_region(dependencies: DependencyView) -> Answer[bool]:
    topology = cast(MVAUParameterTopology, dependencies["op_topology"])
    declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    if topology is MVAUParameterTopology.EMBEDDED:
        return Decided(declaration is MVAURegionDeclaration.STANDARD_EMBEDDED)
    return Decided(declaration is not MVAURegionDeclaration.STANDARD_EMBEDDED)


def _cyclic_interleaved_pumping_supported(dependencies: DependencyView) -> Answer[bool]:
    pumped = dependencies["pumped_memory"]
    return Decided(pumped is ABSENT or not cast(bool, pumped))


def _direct_interleaved_source_available(dependencies: DependencyView) -> Answer[bool]:
    external = dependencies["external_weight_sequence"]
    if external is ABSENT:
        return Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-direct-interleaved-source-missing",
                    MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE,
                    "direct interleaved MVAU requires an external weight sequence",
                    trace=(MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE,),
                ),
            )
        )
    port = cast(Port, dependencies["compute_weight_port"])
    return Decided(cast(BeatSequence, external) == port.beat_sequence)


def _construct_network(
    delivery: DataflowRegion,
    compute: DataflowRegion,
    topology: MVAUConnectionTopology,
    adapter: DataflowRegion | None,
) -> DataflowNetwork:
    delivery_port = delivery.output_interface("weight").port
    compute_port = compute.input_interface("weight").port
    nodes: tuple[NetworkNode, ...]
    edges: tuple[Edge, ...]
    if topology is MVAUConnectionTopology.DIRECT:
        nodes = (NetworkNode("delivery", delivery), NetworkNode("compute", compute))
        edges = (
            Edge(
                "weight",
                RegionEndpoint("delivery", "weight"),
                (
                    SinkContract(
                        RegionEndpoint("compute", "weight"),
                        PositionMap.identity(delivery_port.beat_sequence.image),
                    ),
                ),
            ),
        )
    else:
        if adapter is None:
            raise ValueError("adapter connection requires an adapter region")
        nodes = (
            NetworkNode("delivery", delivery),
            NetworkNode("weight_adapter", adapter),
            NetworkNode("compute", compute),
        )
        edges = (
            Edge(
                "delivery_to_adapter",
                RegionEndpoint("delivery", "weight"),
                (
                    SinkContract(
                        RegionEndpoint("weight_adapter", "weight_in"),
                        PositionMap.identity(delivery_port.beat_sequence.image),
                    ),
                ),
            ),
            Edge(
                "adapter_to_compute",
                RegionEndpoint("weight_adapter", "weight_out"),
                (
                    SinkContract(
                        RegionEndpoint("compute", "weight"),
                        PositionMap.identity(compute_port.beat_sequence.image),
                    ),
                ),
            ),
        )
    return DataflowNetwork(
        nodes,
        edges,
        (
            BoundaryContract(
                "activation",
                RegionEndpoint("compute", "activation"),
                compute.input_interface("activation").port.beat_sequence,
            ),
            BoundaryContract(
                "output",
                RegionEndpoint("compute", "output"),
                compute.output_interface("output").port.beat_sequence,
            ),
        ),
    )


def _derive_network(dependencies: DependencyView) -> Answer[object]:
    topology = cast(MVAUConnectionTopology, dependencies["connection_topology"])
    adapter_value = dependencies["adapter_region"]
    if topology is MVAUConnectionTopology.ADAPTER and adapter_value is ABSENT:
        return Absent(
            (
                Finding(
                    FindingKind.REJECTION,
                    "mvau-weight-adapter-unavailable",
                    MVAUDataflowOpPaths.NETWORK,
                    "adapter topology selected without an applicable adapter region",
                ),
            )
        )
    return Decided(
        _construct_network(
            cast(DataflowRegion, dependencies["delivery_region"]),
            cast(DataflowRegion, dependencies["compute_region"]),
            topology,
            None if adapter_value is ABSENT else cast(DataflowRegion, adapter_value),
        )
    )


def _derive_network_validation(dependencies: DependencyView) -> Answer[object]:
    return Decided(validate_network(cast(DataflowNetwork, dependencies["network"])))


def _network_is_structurally_well_formed(dependencies: DependencyView) -> Answer[bool]:
    return Decided(not cast(NetworkValidationReport, dependencies["report"]))


def _derive_op_result(dependencies: DependencyView) -> Answer[object]:
    topology = cast(MVAUParameterTopology, dependencies["op_topology"])
    association = cast(MVAUSourceAssociation, dependencies["source_association"])
    if topology is MVAUParameterTopology.CYCLIC:
        network = dependencies["network"]
        if network is ABSENT:
            return Absent(
                (
                    Finding(
                        FindingKind.REJECTION,
                        "mvau-cyclic-topology-has-no-network",
                        MVAUDataflowOpPaths.RESULT,
                        "cyclic topology requires a streamed compute region",
                    ),
                )
            )
        return Decided(NetworkRef("mvau", cast(DataflowNetwork, network), association))
    return Decided(
        RegionRef(
            "mvau.compute",
            cast(DataflowRegion, dependencies["compute_region"]),
            association,
        )
    )


def build_mvau_dataflow_op_spec() -> DesignSpaceSpec:
    """Build one flat MVAU operation-level design-space specification."""
    compute_weight_port = DerivedProperty(
        MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT,
        _PORT_SEMANTICS,
        EvaluatorSpec((_COMPUTE_REGION_REF,), _derive_compute_weight_port),
        applies_if=EvaluatorSpec(
            (_REGION_DECLARATION_REF,),
            lambda dependencies: Decided(
                dependencies["region_declaration"] is not MVAURegionDeclaration.STANDARD_EMBEDDED
            ),
        ),
    )
    delivery_weight_port = DerivedProperty(
        MVAUDataflowOpPaths.DELIVERY_WEIGHT_PORT,
        _PORT_SEMANTICS,
        EvaluatorSpec(
            (
                _DELIVERY_DECLARATION_REF,
                _DELIVERY_PE_REF,
                _DELIVERY_SIMD_REF,
                _DELIVERY_INTERLEAVE_REF,
                DependencyRef.problem(
                    "repetitions", MVAUComputeKernelPaths.REPETITIONS, _INTEGER_SEMANTICS
                ),
                DependencyRef.problem(
                    "matrix_width", MVAUComputeKernelPaths.MATRIX_WIDTH, _INTEGER_SEMANTICS
                ),
                DependencyRef.problem(
                    "matrix_height", MVAUComputeKernelPaths.MATRIX_HEIGHT, _INTEGER_SEMANTICS
                ),
                DependencyRef.problem(
                    "weight_element_type",
                    MVAUComputeKernelPaths.WEIGHT_ELEMENT_TYPE,
                    _ELEMENT_TYPE_SEMANTICS,
                ),
            ),
            _derive_delivery_weight_port,
        ),
        applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
    )
    cyclic_spec = build_cyclic_parameter_kernel_spec(
        DependencyRef.property(
            "output_port", MVAUDataflowOpPaths.DELIVERY_WEIGHT_PORT, _PORT_SEMANTICS
        ),
        problem_fields_required=False,
    )
    cyclic_spec = gate_design_space_spec(cyclic_spec, _CYCLIC_STREAMED_REGION_APPLICABILITY)
    cyclic_definition = KernelDefinition(
        "mvau.parameter.cyclic",
        cyclic_spec,
        CYCLIC_PARAMETER_KERNEL.region_declarations,
        CYCLIC_PARAMETER_KERNEL.binding_definitions,
        CyclicParameterKernelPaths.REGION,
        CyclicParameterKernelPaths.BINDING,
        CyclicParameterKernelPaths.BINDING_SELECTION,
        "cyclic_model_structural",
        "cyclic_binding_feasibility",
    )

    source_description_ref = DependencyRef.problem(
        "source_description",
        MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
        _SOURCE_DESCRIPTION_SEMANTICS,
    )
    source_association_ref = DependencyRef.property(
        "source_association",
        MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
        _SOURCE_ASSOCIATION_SEMANTICS,
    )
    network_ref = DependencyRef.property("network", MVAUDataflowOpPaths.NETWORK, _NETWORK_SEMANTICS)
    network_report_ref = DependencyRef.property(
        "report", MVAUDataflowOpPaths.NETWORK_VALIDATION, _NETWORK_REPORT_SEMANTICS
    )
    compute_binding_constraints = next(
        item.constraints
        for item in MVAU_COMPUTE_KERNEL.spec.constraint_sets
        if item.name == "binding_feasibility"
    )
    cyclic_binding_constraints = next(
        item.constraints
        for item in cyclic_definition.spec.constraint_sets
        if item.name == "cyclic_binding_feasibility"
    )
    op_structural_constraints = (
        MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
        MVAUDataflowOpPaths.TOPOLOGY_MATCHES_REGION,
        MVAUDataflowOpPaths.CYCLIC_INTERLEAVED_PUMPING_SUPPORTED,
        MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE,
        MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED,
        MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
        MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
    )
    artifact_constraints = tuple(
        dict.fromkeys(
            (*compute_binding_constraints, *cyclic_binding_constraints, *op_structural_constraints)
        )
    )
    additions = DesignSpaceSpec(
        ProblemSchema(
            (
                ProblemField(
                    MVAUDataflowOpPaths.SOURCE_DESCRIPTION,
                    _SOURCE_DESCRIPTION_SEMANTICS,
                    constraint=_source_description_valid,
                    constraint_description="must contain complete source identities and extents",
                ),
                ProblemField(
                    MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE,
                    _BEAT_SEQUENCE_SEMANTICS,
                    required=False,
                ),
                ProblemField(
                    MVAUDataflowOpPaths.TARGET_FPGA_PART,
                    _STRING_SEMANTICS,
                    required=False,
                    constraint=lambda value: bool(value),
                    constraint_description="must be a non-empty FPGA part identifier",
                ),
                ProblemField(
                    MVAUDataflowOpPaths.TARGET_CLOCK_PERIOD_NS,
                    _FLOAT_SEMANTICS,
                    required=False,
                    constraint=_positive_float,
                    constraint_description="must be a positive clock period",
                ),
            )
        ),
        decisions=(
            Decision(
                MVAUDataflowOpPaths.PARAMETER_TOPOLOGY,
                _TOPOLOGY_SEMANTICS,
                _finite_domain(tuple(MVAUParameterTopology)),
            ),
            Decision(
                MVAUDataflowOpPaths.DELIVERY_PE,
                _INTEGER_SEMANTICS,
                _divisor_domain(MVAUComputeKernelPaths.MATRIX_HEIGHT),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
            Decision(
                MVAUDataflowOpPaths.DELIVERY_SIMD,
                _INTEGER_SEMANTICS,
                _divisor_domain(MVAUComputeKernelPaths.MATRIX_WIDTH),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
            Decision(
                MVAUDataflowOpPaths.DELIVERY_DECLARATION,
                _DELIVERY_DECLARATION_SEMANTICS,
                _finite_domain(tuple(MVAUWeightDeliveryDeclaration)),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
            Decision(
                MVAUDataflowOpPaths.DELIVERY_INTERLEAVE,
                _INTEGER_SEMANTICS,
                _delivery_interleave_domain(),
                applies_if=_DELIVERY_CHUNKED_APPLICABILITY,
            ),
            Decision(
                MVAUDataflowOpPaths.CONNECTION_TOPOLOGY,
                _CONNECTION_TOPOLOGY_SEMANTICS,
                _finite_domain(tuple(MVAUConnectionTopology)),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
        ),
        properties=(
            compute_weight_port,
            delivery_weight_port,
            DerivedProperty(
                MVAUDataflowOpPaths.WEIGHT_ADAPTER_REGION,
                _REGION_SEMANTICS,
                EvaluatorSpec(
                    (
                        _DELIVERY_WEIGHT_PORT_REF,
                        _COMPUTE_WEIGHT_PORT_REF,
                        _DELIVERY_DECLARATION_REF,
                        _REGION_DECLARATION_REF,
                        _DELIVERY_PE_REF,
                        _DELIVERY_SIMD_REF,
                        _COMPUTE_PE_REF,
                        _COMPUTE_SIMD_REF,
                    ),
                    _derive_weight_adapter_region,
                ),
                applies_if=_ADAPTER_CONNECTION_APPLICABILITY,
            ),
            DerivedProperty(
                MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
                _SOURCE_ASSOCIATION_SEMANTICS,
                EvaluatorSpec(
                    (
                        source_description_ref,
                        _TOPOLOGY_REF,
                        _REGION_DECLARATION_REF,
                        _COMPUTATION_REF,
                        DependencyRef.problem(
                            "repetitions",
                            MVAUComputeKernelPaths.REPETITIONS,
                            _INTEGER_SEMANTICS,
                        ),
                        DependencyRef.problem(
                            "matrix_width",
                            MVAUComputeKernelPaths.MATRIX_WIDTH,
                            _INTEGER_SEMANTICS,
                        ),
                        DependencyRef.problem(
                            "matrix_height",
                            MVAUComputeKernelPaths.MATRIX_HEIGHT,
                            _INTEGER_SEMANTICS,
                        ),
                    ),
                    _derive_source_association,
                ),
            ),
            DerivedProperty(
                MVAUDataflowOpPaths.NETWORK,
                _NETWORK_SEMANTICS,
                EvaluatorSpec(
                    (
                        _DELIVERY_REGION_REF,
                        _COMPUTE_REGION_REF,
                        _CONNECTION_TOPOLOGY_REF,
                        _ADAPTER_REGION_REF,
                    ),
                    _derive_network,
                ),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
            DerivedProperty(
                MVAUDataflowOpPaths.NETWORK_VALIDATION,
                _NETWORK_REPORT_SEMANTICS,
                EvaluatorSpec((network_ref,), _derive_network_validation),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
            DerivedProperty(
                MVAUDataflowOpPaths.RESULT,
                _RESULT_SEMANTICS,
                EvaluatorSpec(
                    (
                        _TOPOLOGY_REF,
                        _COMPUTE_REGION_REF,
                        _NETWORK_REF,
                        source_association_ref,
                    ),
                    _derive_op_result,
                ),
            ),
        ),
        constraints=(
            Constraint(
                MVAUDataflowOpPaths.TOPOLOGY_MATCHES_REGION,
                EvaluatorSpec((_TOPOLOGY_REF, _REGION_DECLARATION_REF), _topology_matches_region),
            ),
            Constraint(
                MVAUDataflowOpPaths.CYCLIC_INTERLEAVED_PUMPING_SUPPORTED,
                EvaluatorSpec(
                    (
                        DependencyRef.decision(
                            "pumped_memory",
                            CyclicParameterKernelPaths.PUMPED_MEMORY,
                            _BOOL_SEMANTICS,
                            absence=AbsenceMode.ALLOWS_ABSENT,
                        ),
                    ),
                    _cyclic_interleaved_pumping_supported,
                ),
                applies_if=EvaluatorSpec(
                    (_TOPOLOGY_REF, _REGION_DECLARATION_REF),
                    lambda dependencies: Decided(
                        dependencies["op_topology"] is MVAUParameterTopology.CYCLIC
                        and dependencies["region_declaration"]
                        is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
                    ),
                ),
            ),
            Constraint(
                MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE,
                EvaluatorSpec(
                    (
                        DependencyRef.problem(
                            "external_weight_sequence",
                            MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE,
                            _BEAT_SEQUENCE_SEMANTICS,
                            absence=AbsenceMode.ALLOWS_ABSENT,
                        ),
                        DependencyRef.property(
                            "compute_weight_port",
                            MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT,
                            _PORT_SEMANTICS,
                        ),
                    ),
                    _direct_interleaved_source_available,
                ),
                applies_if=EvaluatorSpec(
                    (_TOPOLOGY_REF, _REGION_DECLARATION_REF),
                    _direct_interleaved_applies,
                ),
            ),
            Constraint(
                MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED,
                EvaluatorSpec(
                    (
                        _DELIVERY_WEIGHT_PORT_REF,
                        _COMPUTE_WEIGHT_PORT_REF,
                        _CONNECTION_TOPOLOGY_REF,
                        _DELIVERY_DECLARATION_REF,
                        _REGION_DECLARATION_REF,
                        _DELIVERY_PE_REF,
                        _DELIVERY_SIMD_REF,
                        _COMPUTE_PE_REF,
                        _COMPUTE_SIMD_REF,
                    ),
                    _weight_connection_supported,
                ),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
            Constraint(
                MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
                EvaluatorSpec(
                    (
                        source_description_ref,
                        source_association_ref,
                        _TOPOLOGY_REF,
                        _COMPUTE_REGION_REF,
                        DependencyRef.problem(
                            "repetitions",
                            MVAUComputeKernelPaths.REPETITIONS,
                            _INTEGER_SEMANTICS,
                        ),
                        DependencyRef.problem(
                            "computation_profile",
                            MVAUComputeKernelPaths.COMPUTATION_PROFILE,
                            _COMPUTATION_SEMANTICS,
                        ),
                    ),
                    _source_association_valid,
                ),
            ),
            Constraint(
                MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
                EvaluatorSpec((network_report_ref,), _network_is_structurally_well_formed),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
            ),
        ),
        constraint_sets=(
            ConstraintSet(
                "mvau_op_structural",
                op_structural_constraints,
            ),
        ),
        readiness_profiles=(
            ReadinessProfile(
                "mvau_op_structural",
                decisions=(
                    MVAUComputeKernelPaths.PE,
                    MVAUComputeKernelPaths.SIMD,
                    MVAUComputeKernelPaths.REGION_DECLARATION,
                    MVAUComputeKernelPaths.INTERLEAVE,
                    MVAUDataflowOpPaths.PARAMETER_TOPOLOGY,
                    MVAUDataflowOpPaths.DELIVERY_PE,
                    MVAUDataflowOpPaths.DELIVERY_SIMD,
                    MVAUDataflowOpPaths.DELIVERY_DECLARATION,
                    MVAUDataflowOpPaths.DELIVERY_INTERLEAVE,
                    MVAUDataflowOpPaths.CONNECTION_TOPOLOGY,
                ),
                properties=(
                    MVAUComputeKernelPaths.REGION,
                    MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
                    MVAUDataflowOpPaths.DELIVERY_WEIGHT_PORT,
                    CyclicParameterKernelPaths.REGION,
                    MVAUDataflowOpPaths.WEIGHT_ADAPTER_REGION,
                    MVAUDataflowOpPaths.NETWORK,
                    MVAUDataflowOpPaths.NETWORK_VALIDATION,
                    MVAUDataflowOpPaths.RESULT,
                ),
                constraints=(
                    MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    MVAUDataflowOpPaths.TOPOLOGY_MATCHES_REGION,
                    MVAUDataflowOpPaths.CYCLIC_INTERLEAVED_PUMPING_SUPPORTED,
                    MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE,
                    MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED,
                    MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
                    MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
                ),
            ),
            ReadinessProfile(
                "artifact_inputs",
                decisions=(
                    *(item.path for item in MVAU_COMPUTE_KERNEL.spec.decisions),
                    *(item.path for item in cyclic_definition.spec.decisions),
                    MVAUDataflowOpPaths.PARAMETER_TOPOLOGY,
                    MVAUDataflowOpPaths.DELIVERY_PE,
                    MVAUDataflowOpPaths.DELIVERY_SIMD,
                    MVAUDataflowOpPaths.DELIVERY_DECLARATION,
                    MVAUDataflowOpPaths.DELIVERY_INTERLEAVE,
                    MVAUDataflowOpPaths.CONNECTION_TOPOLOGY,
                ),
                properties=(
                    MVAUComputeKernelPaths.REGION,
                    MVAUComputeKernelPaths.BINDING_SELECTION,
                    MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
                    MVAUDataflowOpPaths.DELIVERY_WEIGHT_PORT,
                    CyclicParameterKernelPaths.REGION,
                    CyclicParameterKernelPaths.BINDING_SELECTION,
                    MVAUDataflowOpPaths.WEIGHT_ADAPTER_REGION,
                    MVAUDataflowOpPaths.NETWORK,
                    MVAUDataflowOpPaths.RESULT,
                ),
                constraints=artifact_constraints,
            ),
        ),
    )
    return assemble_kernel_specs((MVAU_COMPUTE_KERNEL, cyclic_definition), additions=additions)


MVAU_DATAFLOW_OP_SPEC = build_mvau_dataflow_op_spec()

__all__ = [
    "BindingLocalStateDestination",
    "CoordinateMappingKind",
    "DataflowOpResult",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAUConnectionTopology",
    "MVAUDataflowOpPaths",
    "MVAUParameterTopology",
    "MVAUSourceAssociation",
    "MVAUSourceDescription",
    "MVAUWeightDeliveryDeclaration",
    "NetworkRef",
    "RegionRef",
    "SemanticOperandDestination",
    "SourceOperandAssociation",
    "SourceOperandDestination",
    "build_mvau_dataflow_op_spec",
]
