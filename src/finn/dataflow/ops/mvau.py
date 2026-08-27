# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Flat MVAU ``DataflowOp`` assembly over compute and cyclic-delivery Kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
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
from finn.dataflow.mvau.regions import MVAURegionDeclaration
from finn.dataflow.network import (
    BoundaryContract,
    ChannelSpec,
    DataflowNetwork,
    Edge,
    NetworkNode,
    OrderedChannel,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import NetworkValidationReport, validate_network
from finn.dataflow.parameters.cyclic.definition import (
    CYCLIC_PARAMETER_KERNEL,
    CyclicParameterKernelPaths,
    CyclicParameterRegionDeclaration,
    build_cyclic_parameter_kernel_spec,
)
from finn.dataflow.region import BeatSequence, DataflowRegion, Port


class MVAUParameterTopology(str, Enum):
    """Initial op-level parameter-delivery topology alternatives."""

    EMBEDDED = "embedded"
    DIRECT = "direct"
    CYCLIC = "cyclic"


class CoordinateMappingKind(str, Enum):
    """Explicit source-to-region coordinate transformations used by MVAU."""

    FLATTEN_LEADING = "flatten_leading"
    TRANSPOSE_2D = "transpose_2d"
    BINDING_LOCAL_STATE = "binding_local_state"


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "leading_shape", tuple(self.leading_shape))
        object.__setattr__(self, "fused_source_node_ids", tuple(self.fused_source_node_ids))


@dataclass(frozen=True)
class SourceOperandAssociation:
    """Explicit association from one source operand to a semantic or binding target."""

    role: str
    source_operand_id: str
    destination_id: str
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
    PARAMETER_TOPOLOGY = QualifiedPath("mvau.op.parameter_topology")

    COMPUTE_WEIGHT_PORT = QualifiedPath("semantic.mvau.op.compute_weight_port")
    SOURCE_ASSOCIATION = QualifiedPath("semantic.mvau.op.source_association")
    NETWORK = QualifiedPath("semantic.mvau.op.network")
    NETWORK_VALIDATION = QualifiedPath("semantic.mvau.op.network_validation")
    RESULT = QualifiedPath("semantic.mvau.op.result")

    TOPOLOGY_MATCHES_REGION = QualifiedPath("constraint.mvau.op.topology_matches_region")
    DELIVERY_MATCHES_REGION = QualifiedPath("constraint.mvau.op.delivery_matches_region")
    DIRECT_INTERLEAVED_SOURCE_AVAILABLE = QualifiedPath(
        "constraint.mvau.op.direct_interleaved_source_available"
    )
    SOURCE_ASSOCIATION_VALID = QualifiedPath("constraint.mvau.op.source_association_valid")
    NETWORK_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.op.network_structurally_well_formed"
    )


_INTEGER_SEMANTICS = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
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
    return (
        all(isinstance(name, str) and bool(name) for name in names)
        and threshold_valid
        and all(type(extent) is int and extent > 0 for extent in description.leading_shape)
    )


def _finite_domain(values: tuple[object, ...]) -> DecisionDomain:
    allowed = frozenset(values)

    def accepts(value: object, _dependencies: DependencyView) -> Answer[bool]:
        return Decided(value in allowed)

    def candidates(_dependencies: DependencyView) -> Answer[tuple[object, ...]]:
        return Decided(values)

    return DecisionDomain((), accepts, EvaluatorSpec((), candidates))


_TOPOLOGY_REF = DependencyRef.decision(
    "op_topology", MVAUDataflowOpPaths.PARAMETER_TOPOLOGY, _TOPOLOGY_SEMANTICS
)
_REGION_DECLARATION_REF = DependencyRef.decision(
    "region_declaration",
    MVAUComputeKernelPaths.REGION_DECLARATION,
    MVAU_COMPUTE_KERNEL.spec.decisions[2].value_semantics,
)
_COMPUTE_REGION_DECLARATION_FOR_SCOPE_REF = DependencyRef.decision(
    "compute_region_declaration",
    MVAUComputeKernelPaths.REGION_DECLARATION,
    MVAU_COMPUTE_KERNEL.spec.decisions[2].value_semantics,
)
_COMPUTE_REGION_REF = DependencyRef.property(
    "compute_region", MVAUComputeKernelPaths.REGION, _REGION_SEMANTICS
)
_DELIVERY_DECLARATION_REF = DependencyRef.decision(
    "delivery_declaration",
    CyclicParameterKernelPaths.REGION_DECLARATION,
    build_cyclic_parameter_kernel_spec().decisions[0].value_semantics,
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


def _direct_interleaved_applies(dependencies: DependencyView) -> Answer[bool]:
    return Decided(
        dependencies["op_topology"] is MVAUParameterTopology.DIRECT
        and dependencies["region_declaration"] is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
    )


def _derive_compute_weight_port(dependencies: DependencyView) -> Answer[object]:
    region = cast(DataflowRegion, dependencies["compute_region"])
    return Decided(region.input_interface("weight").port)


def _derive_source_association(dependencies: DependencyView) -> Answer[object]:
    description = cast(MVAUSourceDescription, dependencies["source_description"])
    repetitions = cast(int, dependencies["repetitions"])
    matrix_width = cast(int, dependencies["matrix_width"])
    matrix_height = cast(int, dependencies["matrix_height"])
    operands = [
        SourceOperandAssociation(
            "activation",
            description.activation_operand_id,
            "X",
            CoordinateMappingKind.FLATTEN_LEADING,
            (*description.leading_shape, matrix_width),
            (repetitions, matrix_width),
        ),
        SourceOperandAssociation(
            "weight",
            description.weight_operand_id,
            "W",
            CoordinateMappingKind.TRANSPOSE_2D,
            (matrix_width, matrix_height),
            (matrix_height, matrix_width),
        ),
        SourceOperandAssociation(
            "output",
            description.output_operand_id,
            "Y",
            CoordinateMappingKind.FLATTEN_LEADING,
            (*description.leading_shape, matrix_height),
            (repetitions, matrix_height),
        ),
    ]
    if description.threshold_operand_id is not None:
        operands.append(
            SourceOperandAssociation(
                "threshold",
                description.threshold_operand_id,
                "thresholds",
                CoordinateMappingKind.BINDING_LOCAL_STATE,
                (),
                (),
            )
        )
    return Decided(
        MVAUSourceAssociation(
            description.source_node_id,
            description.fused_source_node_ids,
            tuple(operands),
        )
    )


def _source_association_valid(dependencies: DependencyView) -> Answer[bool]:
    description = cast(MVAUSourceDescription, dependencies["source_description"])
    repetitions = cast(int, dependencies["repetitions"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    threshold_valid = (
        profile is not MVAUComputationProfile.FUSED_THRESHOLD
        or description.threshold_operand_id is not None
    )
    return Decided(prod(description.leading_shape) == repetitions and threshold_valid)


def _topology_matches_region(dependencies: DependencyView) -> Answer[bool]:
    topology = cast(MVAUParameterTopology, dependencies["op_topology"])
    declaration = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    if topology is MVAUParameterTopology.EMBEDDED:
        return Decided(declaration is MVAURegionDeclaration.STANDARD_EMBEDDED)
    return Decided(declaration is not MVAURegionDeclaration.STANDARD_EMBEDDED)


def _delivery_matches_region(dependencies: DependencyView) -> Answer[bool]:
    compute = cast(MVAURegionDeclaration, dependencies["region_declaration"])
    delivery = cast(CyclicParameterRegionDeclaration, dependencies["delivery_declaration"])
    expected = (
        CyclicParameterRegionDeclaration.CHUNKED
        if compute is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
        else CyclicParameterRegionDeclaration.FULL_TILE
    )
    return Decided(delivery is expected)


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


def _construct_network(delivery: DataflowRegion, compute: DataflowRegion) -> DataflowNetwork:
    delivery_port = delivery.output_interface("weight").port
    return DataflowNetwork(
        (NetworkNode("delivery", delivery), NetworkNode("compute", compute)),
        (
            Edge(
                "weight",
                RegionEndpoint("delivery", "weight"),
                (
                    SinkContract(
                        RegionEndpoint("compute", "weight"),
                        PositionMap.identity(delivery_port.beat_sequence.image),
                    ),
                ),
                transport=OrderedChannel(ChannelSpec("weight")),
            ),
        ),
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
    return Decided(
        _construct_network(
            cast(DataflowRegion, dependencies["delivery_region"]),
            cast(DataflowRegion, dependencies["compute_region"]),
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
    cyclic_spec = build_cyclic_parameter_kernel_spec(
        DependencyRef.property(
            "output_port", MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT, _PORT_SEMANTICS
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
        CyclicParameterKernelPaths.BINDING_WITNESS,
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
            )
        ),
        decisions=(
            Decision(
                MVAUDataflowOpPaths.PARAMETER_TOPOLOGY,
                _TOPOLOGY_SEMANTICS,
                _finite_domain(tuple(MVAUParameterTopology)),
            ),
        ),
        properties=(
            compute_weight_port,
            DerivedProperty(
                MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
                _SOURCE_ASSOCIATION_SEMANTICS,
                EvaluatorSpec(
                    (
                        source_description_ref,
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
                EvaluatorSpec((_DELIVERY_REGION_REF, _COMPUTE_REGION_REF), _derive_network),
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
                MVAUDataflowOpPaths.DELIVERY_MATCHES_REGION,
                EvaluatorSpec(
                    (_REGION_DECLARATION_REF, _DELIVERY_DECLARATION_REF),
                    _delivery_matches_region,
                ),
                applies_if=_CYCLIC_STREAMED_REGION_APPLICABILITY,
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
                MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
                EvaluatorSpec(
                    (
                        source_description_ref,
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
                (
                    MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    MVAUDataflowOpPaths.TOPOLOGY_MATCHES_REGION,
                    MVAUDataflowOpPaths.DELIVERY_MATCHES_REGION,
                    MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE,
                    MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
                    MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
                ),
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
                    CyclicParameterKernelPaths.REGION_DECLARATION,
                ),
                properties=(
                    MVAUComputeKernelPaths.REGION,
                    MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
                    CyclicParameterKernelPaths.REGION,
                    MVAUDataflowOpPaths.NETWORK,
                    MVAUDataflowOpPaths.NETWORK_VALIDATION,
                    MVAUDataflowOpPaths.RESULT,
                ),
                constraints=(
                    MVAUComputeKernelPaths.REGION_STRUCTURALLY_WELL_FORMED,
                    MVAUDataflowOpPaths.TOPOLOGY_MATCHES_REGION,
                    MVAUDataflowOpPaths.DELIVERY_MATCHES_REGION,
                    MVAUDataflowOpPaths.DIRECT_INTERLEAVED_SOURCE_AVAILABLE,
                    MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
                    MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
                ),
            ),
        ),
    )
    return assemble_kernel_specs((MVAU_COMPUTE_KERNEL, cyclic_definition), additions=additions)


MVAU_DATAFLOW_OP_SPEC = build_mvau_dataflow_op_spec()

__all__ = [
    "CoordinateMappingKind",
    "DataflowOpResult",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAUDataflowOpPaths",
    "MVAUParameterTopology",
    "MVAUSourceAssociation",
    "MVAUSourceDescription",
    "NetworkRef",
    "RegionRef",
    "SourceOperandAssociation",
    "build_mvau_dataflow_op_spec",
]
