# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Provider-era MVAU assembly retained behind the compatibility boundary.

The operation class owns the source semantics, the problem facts, and three
static Kernel pools.  A design point selects one compute Kernel, optionally one
supplier Kernel, and optionally one adapter Kernel; the selected Kernels derive
their own Regions and demands, and assembly composes them.  Nothing here
re-decides a Region form, an implementation binding, or a delivery tile.
"""

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
    DependencyRef,
    DependencyView,
    DerivedProperty,
    DesignSpaceSpec,
    EvaluatorSpec,
    Finding,
    FindingKind,
    QualifiedPath,
    ReadinessProfile,
    Unresolved,
    ValueSemantics,
    as_object_semantics,
)
from finn.dataflow.kernels import (
    KERNEL_ID_SEMANTICS,
    NO_KERNEL,
    SELECTED_KERNEL_SEMANTICS,
    KernelSelection,
    SelectedKernel,
)
from finn.dataflow.hardware import check_declared_references
from finn.dataflow.mvau.associations import (
    BindingLocalStateDestination,
    CoordinateMappingKind,
    MVAUNetworkRef as NetworkRef,
    MVAUParameterTopology,
    MVAUSourceAssociation,
    SemanticOperandDestination,
    SourceOperandAssociation,
    SourceOperandDestination,
)
from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.compute_kernels import (
    FULL_TILE_WEIGHT_EXPORT,
    MVAU_COMPUTE_SELECTION,
    REGION_FORM_EXPORT,
    WEIGHT_INTERFACE,
)
from finn.dataflow.mvau.compute_kernels import DECOMPOSED_MVAU_KERNELS, MVAU_REPLAY_SELECTION
from finn.dataflow.mvau.decomposed import ACTIVATION_EDGE, DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.mvau.regions import MVAURegionDeclaration
from finn.dataflow.mvau.weight_adapter_kernel import build_mvau_weight_adapter_selection
from finn.dataflow.mvau_problem import (
    MVAU_PROBLEM,
    MVAU_PROBLEM_SPEC,
    MVAUDspBlock,
    MVAUProblemPaths,
    MVAUSourceDescription,
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
from finn.dataflow.parameters.cyclic.definition import CyclicTargetMemoryCapabilities
from finn.dataflow.parameters.supply_kernels import (
    FINN_RTL_MEMSTREAM_PATHS,
    OUTPUT_PORT_EXPORT,
    build_mvau_weight_supply_selection,
)
from finn.dataflow.design.region import QONNX_DATATYPE_SEMANTICS
from finn.dataflow.region import BeatSequence, DataflowRegion, Port
from finn.dataflow.resolution import (
    DATAFLOW_OP_RESULT_SEMANTICS,
    RegionRef as GenericRegionRef,
)
from finn.dataflow.spec_algebra import assemble_specs


@dataclass(frozen=True)
class RegionRef(GenericRegionRef):
    """Provider-era selected Region with a typed source association."""

    source_association: MVAUSourceAssociation


DataflowOpResult = RegionRef | NetworkRef


class MVAUDataflowOpPaths:
    """Stable paths owned by the MVAU source-operation assembly."""

    # Problem paths are declared in ``finn.dataflow.mvau_problem`` and named
    # here for the assembly's convenience; they are not a second definition.
    SOURCE_DESCRIPTION = MVAUProblemPaths.SOURCE_DESCRIPTION
    ACCUMULATOR_TYPE_ANALYSIS_OWNER = MVAUProblemPaths.ACCUMULATOR_TYPE_ANALYSIS_OWNER
    WEIGHT_INITIALIZER_FINGERPRINT = MVAUProblemPaths.WEIGHT_INITIALIZER_FINGERPRINT
    THRESHOLD_INITIALIZER_FINGERPRINT = MVAUProblemPaths.THRESHOLD_INITIALIZER_FINGERPRINT
    EXTERNAL_WEIGHT_SEQUENCE = MVAUProblemPaths.EXTERNAL_WEIGHT_SEQUENCE
    TARGET_FPGA_PART = MVAUProblemPaths.TARGET_FPGA_PART
    TARGET_CLOCK_PERIOD_NS = MVAUProblemPaths.TARGET_CLOCK_PERIOD_NS
    EFFECTIVE_NARROW_WEIGHTS = MVAUProblemPaths.EFFECTIVE_NARROW_WEIGHTS

    DESIGN = QualifiedPath("mvau.design")

    COMPUTE_KERNEL = MVAU_COMPUTE_SELECTION.paths.kernel
    COMPUTE_REGION = MVAU_COMPUTE_SELECTION.paths.region
    COMPUTE_SELECTED_KERNEL = MVAU_COMPUTE_SELECTION.paths.selected_kernel
    COMPUTE_WEIGHT_PORT = MVAU_COMPUTE_SELECTION.paths.demand(WEIGHT_INTERFACE)
    COMPUTE_REGION_FORM = MVAU_COMPUTE_SELECTION.paths.export(REGION_FORM_EXPORT)
    COMPUTE_FULL_TILE_WEIGHT_PORT = MVAU_COMPUTE_SELECTION.paths.export(FULL_TILE_WEIGHT_EXPORT)

    PARAMETER_TOPOLOGY = QualifiedPath("semantic.mvau.op.parameter_topology")
    SOURCE_ASSOCIATION = QualifiedPath("semantic.mvau.op.source_association")
    NETWORK = QualifiedPath("semantic.mvau.op.network")
    NETWORK_VALIDATION = QualifiedPath("semantic.mvau.op.network_validation")
    RESULT = QualifiedPath("semantic.mvau.op.result")

    WEIGHT_CONNECTION_SUPPORTED = QualifiedPath("constraint.mvau.op.weight_connection_supported")
    DECOMPOSED_SUPPLY_SUPPORTED = QualifiedPath("constraint.mvau.op.decomposed_supply_supported")
    EXPOSED_WEIGHT_SOURCE_AVAILABLE = QualifiedPath(
        "constraint.mvau.op.exposed_weight_source_available"
    )
    INTERLEAVED_PUMPING_SUPPORTED = QualifiedPath(
        "constraint.mvau.op.interleaved_pumping_supported"
    )
    SOURCE_ASSOCIATION_VALID = QualifiedPath("constraint.mvau.op.source_association_valid")
    NETWORK_STRUCTURALLY_WELL_FORMED = QualifiedPath(
        "constraint.mvau.op.network_structurally_well_formed"
    )


_INTEGER = as_object_semantics(ValueSemantics.immutable_nominal(int, name="integer"))
_FLOAT = as_object_semantics(ValueSemantics.immutable_nominal(float, name="float"))
_STRING = as_object_semantics(ValueSemantics.immutable_nominal(str, name="string"))
_BOOL = as_object_semantics(ValueSemantics.immutable_nominal(bool, name="boolean"))
_ELEMENT_TYPE = QONNX_DATATYPE_SEMANTICS
_PORT = as_object_semantics(ValueSemantics.immutable_nominal(Port, name="Port"))
_BEAT_SEQUENCE = as_object_semantics(
    ValueSemantics.immutable_nominal(BeatSequence, name="BeatSequence")
)
_REGION = as_object_semantics(DATAFLOW_REGION_SEMANTICS)
_NETWORK = as_object_semantics(DATAFLOW_NETWORK_SEMANTICS)
_NETWORK_REPORT = as_object_semantics(NETWORK_VALIDATION_REPORT_SEMANTICS)
_TARGET_MEMORY = as_object_semantics(
    ValueSemantics.immutable_nominal(
        CyclicTargetMemoryCapabilities, name="CyclicTargetMemoryCapabilities"
    )
)
_SOURCE_DESCRIPTION = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUSourceDescription, name="MVAUSourceDescription")
)
_SOURCE_ASSOCIATION = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUSourceAssociation, name="MVAUSourceAssociation")
)
_TOPOLOGY = as_object_semantics(
    ValueSemantics.immutable_nominal(MVAUParameterTopology, name="MVAUParameterTopology")
)


def _enum_semantics(enum_type: type[Enum]) -> ValueSemantics[object]:
    semantics: ValueSemantics[Enum] = ValueSemantics(
        enum_type,
        enum_type.__name__,
        lambda value: type(value) is enum_type,
        lambda left, right: left is right,
        lambda value: value,
    )
    return as_object_semantics(semantics)


_COMPUTATION = _enum_semantics(MVAUComputationProfile)
_DSP_BLOCK = _enum_semantics(MVAUDspBlock)


# -- selection wiring --------------------------------------------------------

_COMPUTE_REGION_REF = DependencyRef.property(
    "compute_region", MVAUDataflowOpPaths.COMPUTE_REGION, _REGION
)
_COMPUTE_WEIGHT_PORT_REF = DependencyRef.property(
    "compute_weight_port",
    MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT,
    _PORT,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_COMPUTE_REGION_FORM_REF = DependencyRef.property(
    "region_form", MVAUDataflowOpPaths.COMPUTE_REGION_FORM, _enum_semantics(MVAURegionDeclaration)
)
_COMPUTE_SELECTED_REF = DependencyRef.property(
    "compute_kernel", MVAUDataflowOpPaths.COMPUTE_SELECTED_KERNEL, SELECTED_KERNEL_SEMANTICS
)


def _weight_is_streamed(dependencies: DependencyView) -> Answer[bool]:
    return Decided(dependencies["region_form"] is not MVAURegionDeclaration.STANDARD_EMBEDDED)


#: The supply pool is applicable exactly when the selected compute Kernel has
#: published a weight demand.  Initializer presence never activates it.
_SUPPLY_APPLIES = EvaluatorSpec((_COMPUTE_REGION_FORM_REF,), _weight_is_streamed)

MVAU_WEIGHT_SUPPLY_SELECTION = build_mvau_weight_supply_selection(
    DependencyRef.property("demand", MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT, _PORT),
    DependencyRef.property("full_tile", MVAUDataflowOpPaths.COMPUTE_FULL_TILE_WEIGHT_PORT, _PORT),
    applies_if=_SUPPLY_APPLIES,
)

_SUPPLY_PATHS = MVAU_WEIGHT_SUPPLY_SELECTION.paths
_SUPPLY_KERNEL_REF = DependencyRef.decision(
    "supply_kernel",
    _SUPPLY_PATHS.kernel,
    KERNEL_ID_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_SUPPLY_REGION_REF = DependencyRef.property(
    "supply_region", _SUPPLY_PATHS.region, _REGION, absence=AbsenceMode.ALLOWS_ABSENT
)
_REPLAY_REGION_REF = DependencyRef.property(
    "replay_region",
    MVAU_REPLAY_SELECTION.paths.region,
    _REGION,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_REPLAY_KERNEL_REF = DependencyRef.decision(
    "replay_kernel",
    MVAU_REPLAY_SELECTION.paths.kernel,
    KERNEL_ID_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_SUPPLY_OUTPUT_PORT_REF = DependencyRef.property(
    "supply_output_port",
    _SUPPLY_PATHS.export(OUTPUT_PORT_EXPORT),
    _PORT,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_SUPPLY_SELECTED_REF = DependencyRef.property(
    "supply_selected",
    _SUPPLY_PATHS.selected_kernel,
    SELECTED_KERNEL_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)


def _supplier_selected(dependencies: DependencyView) -> Answer[bool]:
    value = dependencies["supply_kernel"]
    return Decided(value is not ABSENT and value != NO_KERNEL)


#: The adapter pool is applicable exactly when a supplier is producing a
#: sequence that has to reach the compute demand.
_ADAPTER_APPLIES = EvaluatorSpec((_SUPPLY_KERNEL_REF,), _supplier_selected)

MVAU_WEIGHT_ADAPTER_SELECTION = build_mvau_weight_adapter_selection(
    DependencyRef.property("source_port", _SUPPLY_PATHS.export(OUTPUT_PORT_EXPORT), _PORT),
    DependencyRef.property("sink_port", MVAUDataflowOpPaths.COMPUTE_WEIGHT_PORT, _PORT),
    applies_if=_ADAPTER_APPLIES,
)

_ADAPTER_PATHS = MVAU_WEIGHT_ADAPTER_SELECTION.paths
_ADAPTER_KERNEL_REF = DependencyRef.decision(
    "adapter_kernel",
    _ADAPTER_PATHS.kernel,
    KERNEL_ID_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_ADAPTER_REGION_REF = DependencyRef.property(
    "adapter_region", _ADAPTER_PATHS.region, _REGION, absence=AbsenceMode.ALLOWS_ABSENT
)
_ADAPTER_SELECTED_REF = DependencyRef.property(
    "adapter_selected",
    _ADAPTER_PATHS.selected_kernel,
    SELECTED_KERNEL_SEMANTICS,
    absence=AbsenceMode.ALLOWS_ABSENT,
)

_SOURCE_DESCRIPTION_REF = DependencyRef.problem(
    "source_description", MVAUDataflowOpPaths.SOURCE_DESCRIPTION, _SOURCE_DESCRIPTION
)
_SOURCE_ASSOCIATION_REF = DependencyRef.property(
    "source_association", MVAUDataflowOpPaths.SOURCE_ASSOCIATION, _SOURCE_ASSOCIATION
)
_TOPOLOGY_REF = DependencyRef.property(
    "parameter_topology", MVAUDataflowOpPaths.PARAMETER_TOPOLOGY, _TOPOLOGY
)
_NETWORK_REF = DependencyRef.property(
    "network", MVAUDataflowOpPaths.NETWORK, _NETWORK, absence=AbsenceMode.ALLOWS_ABSENT
)
_REPETITIONS_REF = MVAU_PROBLEM.repetitions.dependency("repetitions")
_MATRIX_WIDTH_REF = MVAU_PROBLEM.matrix_width.dependency("matrix_width")
_MATRIX_HEIGHT_REF = MVAU_PROBLEM.matrix_height.dependency("matrix_height")
_COMPUTATION_REF = MVAU_PROBLEM.computation_profile.dependency("computation_profile")


# -- derived assembly --------------------------------------------------------


def _derive_parameter_topology(dependencies: DependencyView) -> Answer[object]:
    if dependencies["region_form"] is MVAURegionDeclaration.STANDARD_EMBEDDED:
        return Decided(MVAUParameterTopology.EMBEDDED)
    supply = dependencies["supply_kernel"]
    if supply is not ABSENT and supply != NO_KERNEL:
        return Decided(MVAUParameterTopology.CYCLIC)
    return Decided(MVAUParameterTopology.DIRECT)


def _weight_connection_supported(dependencies: DependencyView) -> Answer[bool]:
    """Test exact endpoint compatibility first, then declared adapters.

    Exact compatibility is the direct connection, never a width-only rule.  An
    adapter that is not needed is not merely wasteful, it is a second answer to
    a question already settled, so it is rejected.
    """

    raw_source = dependencies["supply_output_port"]
    raw_sink = dependencies["compute_weight_port"]
    if raw_source is ABSENT or raw_sink is ABSENT:
        # A connection whose endpoints are not both derived is not unsupported;
        # it is unanswered, and saying so beats crashing on an absent value.
        return Unresolved(
            (
                Finding(
                    FindingKind.LIMITATION,
                    "mvau-weight-endpoints-unavailable",
                    MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED,
                    "both weight endpoints must derive before compatibility is decidable",
                ),
            )
        )
    source = cast(Port, raw_source)
    sink = cast(Port, raw_sink)
    directly_compatible = (
        source.operand == sink.operand and source.beat_sequence == sink.beat_sequence
    )
    adapter = dependencies["adapter_kernel"]
    if adapter is ABSENT or adapter == NO_KERNEL:
        return Decided(directly_compatible)
    return Decided(not directly_compatible and dependencies["adapter_region"] is not ABSENT)


def _exposed_weight_source_available(dependencies: DependencyView) -> Answer[bool]:
    """An unsupplied streamed weight boundary must be servable by the graph.

    A batch-interleaved compute Kernel demands a chunked sequence no ordinary
    graph edge produces, so leaving it exposed requires an explicit external
    sequence that matches it exactly.
    """

    external = dependencies["external_weight_sequence"]
    port = cast(Port, dependencies["compute_weight_port"])
    if external is ABSENT:
        if dependencies["region_form"] is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED:
            return Unresolved(
                (
                    Finding(
                        FindingKind.LIMITATION,
                        "mvau-exposed-interleaved-source-missing",
                        MVAUDataflowOpPaths.EXPOSED_WEIGHT_SOURCE_AVAILABLE,
                        "an exposed batch-interleaved weight boundary requires an "
                        "external weight sequence",
                        trace=(MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE,),
                    ),
                )
            )
        return Decided(True)
    return Decided(cast(BeatSequence, external) == port.beat_sequence)


def _interleaved_pumping_supported(dependencies: DependencyView) -> Answer[bool]:
    pumped = dependencies["pumped_memory"]
    return Decided(pumped is ABSENT or not cast(bool, pumped))


def _semantic_owners(dependencies: DependencyView) -> tuple[str, str]:
    """The node ids the source tensors land on, as ``(compute, activation)``.

    Assembly is what moves them.  A Network names its compute node
    ``DOT_PRODUCT_NODE``, and a replay node in front of it takes the activation
    tensor while everything else keeps the owner it had.  Derivation and
    validation must agree on this exactly, so both ask here rather than each
    reconstructing it from the topology.
    """

    topology = cast(MVAUParameterTopology, dependencies["parameter_topology"])
    assembled = topology is MVAUParameterTopology.CYCLIC or _has_replay(dependencies)
    compute_owner = DOT_PRODUCT_NODE if assembled else "mvau.compute"
    activation_owner = REPLAY_NODE if _has_replay(dependencies) else compute_owner
    return compute_owner, activation_owner


def _derive_source_association(dependencies: DependencyView) -> Answer[object]:
    description = cast(MVAUSourceDescription, dependencies["source_description"])
    topology = cast(MVAUParameterTopology, dependencies["parameter_topology"])
    region_form = cast(MVAURegionDeclaration, dependencies["region_form"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    repetitions = cast(int, dependencies["repetitions"])
    matrix_width = cast(int, dependencies["matrix_width"])
    matrix_height = cast(int, dependencies["matrix_height"])
    compute_owner, activation_owner = _semantic_owners(dependencies)
    operands = [
        SourceOperandAssociation(
            "activation",
            description.activation_operand_id,
            SemanticOperandDestination(activation_owner, "X"),
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
            compute_owner, "weights"
        )
    elif topology is MVAUParameterTopology.DIRECT:
        weight_destination = SemanticOperandDestination(compute_owner, "W")
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
    supply = dependencies["supply_selected"]
    adapter = dependencies["adapter_selected"]
    return Decided(
        MVAUSourceAssociation(
            description.source_node_id,
            description.fused_source_node_ids,
            region_form.value,
            topology,
            tuple(operands),
            cast(SelectedKernel, dependencies["compute_kernel"]).kernel_id,
            None if supply is ABSENT else cast(SelectedKernel, supply).kernel_id,
            None if adapter is ABSENT else cast(SelectedKernel, adapter).kernel_id,
        )
    )


def _source_association_valid(dependencies: DependencyView) -> Answer[bool]:
    description = cast(MVAUSourceDescription, dependencies["source_description"])
    association = cast(MVAUSourceAssociation, dependencies["source_association"])
    topology = cast(MVAUParameterTopology, dependencies["parameter_topology"])
    compute = cast(DataflowRegion, dependencies["compute_region"])
    repetitions = cast(int, dependencies["repetitions"])
    profile = cast(MVAUComputationProfile, dependencies["computation_profile"])
    threshold_valid = profile is not MVAUComputationProfile.FUSED_THRESHOLD or (
        description.threshold_operand_id is not None and description.threshold_shape is not None
    )
    compute_owner, activation_owner = _semantic_owners(dependencies)
    # A semantic destination is valid only if the node it names is really in
    # the assembly *and* carries that operand.  Checking membership per owner
    # is what makes a misrouted activation -- the replay tensor addressed to
    # the dot product, or the other way round -- fail here.
    replay = dependencies["replay_region"]
    operands_by_owner = {
        compute_owner: {interface.port.operand.id for interface in compute.interfaces}
    }
    if activation_owner != compute_owner:
        operands_by_owner[activation_owner] = (
            set()
            if replay is ABSENT
            else {
                interface.port.operand.id for interface in cast(DataflowRegion, replay).interfaces
            }
        )
    semantic_destinations_valid = True
    for operand in association.operands:
        destination = operand.destination
        if not isinstance(destination, SemanticOperandDestination):
            continue
        available = operands_by_owner.get(destination.owner_id)
        if available is None or destination.operand_id not in available:
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
            == BindingLocalStateDestination(compute_owner, "weights")
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


def _construct_network(
    delivery: DataflowRegion | None,
    compute: DataflowRegion,
    adapter: DataflowRegion | None,
    replay: DataflowRegion | None = None,
) -> DataflowNetwork:
    """Assemble whichever nodes the selected Kernels put in play.

    Two independent reasons produce more than one node: a selected supplier
    delivers the weights, and the decomposed compute member needs an activation
    replay in front of it.  They compose -- either, both, or neither.
    """

    compute_port = compute.input_interface("weight").port
    nodes: list[NetworkNode] = [NetworkNode(DOT_PRODUCT_NODE, compute)]
    edges: list[Edge] = []

    if delivery is not None:
        delivery_port = delivery.output_interface("weight").port
        nodes.append(NetworkNode("delivery", delivery))
        if adapter is None:
            edges.append(
                Edge(
                    "weight",
                    RegionEndpoint("delivery", "weight"),
                    (
                        SinkContract(
                            RegionEndpoint("compute", "weight"),
                            PositionMap.identity(delivery_port.beat_sequence.image),
                        ),
                    ),
                )
            )
        else:
            nodes.append(NetworkNode("weight_adapter", adapter))
            edges.extend(
                (
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
            )

    activation_endpoint = RegionEndpoint("compute", "activation")
    activation_sequence = compute.input_interface("activation").port.beat_sequence
    if replay is not None:
        produced = replay.output_interface("activation_out").port
        nodes.append(NetworkNode(REPLAY_NODE, replay))
        edges.append(
            Edge(
                ACTIVATION_EDGE,
                RegionEndpoint(REPLAY_NODE, "activation_out"),
                (
                    SinkContract(
                        RegionEndpoint("compute", "activation"),
                        PositionMap.identity(produced.beat_sequence.image),
                    ),
                ),
            )
        )
        # The activation now enters at the replay node, and the sequence the
        # outside sees is the compact one it consumes, not the expanded one the
        # dot product does.
        activation_endpoint = RegionEndpoint(REPLAY_NODE, "activation_in")
        activation_sequence = replay.input_interface("activation_in").port.beat_sequence

    boundaries = [
        BoundaryContract("activation", activation_endpoint, activation_sequence),
        BoundaryContract(
            "output",
            RegionEndpoint("compute", "output"),
            compute.output_interface("output").port.beat_sequence,
        ),
    ]
    if delivery is None:
        boundaries.append(
            BoundaryContract(
                "weight", RegionEndpoint("compute", "weight"), compute_port.beat_sequence
            )
        )
    return DataflowNetwork(tuple(nodes), tuple(edges), tuple(boundaries))


def _optional_region(value: object) -> DataflowRegion | None:
    return None if value is ABSENT else cast(DataflowRegion, value)


def _derive_network(dependencies: DependencyView) -> Answer[object]:
    return Decided(
        _construct_network(
            _optional_region(dependencies["supply_region"]),
            cast(DataflowRegion, dependencies["compute_region"]),
            _optional_region(dependencies["adapter_region"]),
            _optional_region(dependencies["replay_region"]),
        )
    )


def _derive_network_validation(dependencies: DependencyView) -> Answer[object]:
    return Decided(validate_network(cast(DataflowNetwork, dependencies["network"])))


def _network_is_structurally_well_formed(dependencies: DependencyView) -> Answer[bool]:
    return Decided(not cast(NetworkValidationReport, dependencies["report"]))


def _derive_op_result(dependencies: DependencyView) -> Answer[object]:
    """One Region when the compute stands alone, a Network when it does not."""

    topology = cast(MVAUParameterTopology, dependencies["parameter_topology"])
    association = cast(MVAUSourceAssociation, dependencies["source_association"])
    assembled = topology is MVAUParameterTopology.CYCLIC or _has_replay(dependencies)
    if assembled:
        network = dependencies["network"]
        if network is ABSENT:
            return Absent(
                (
                    Finding(
                        FindingKind.REJECTION,
                        "mvau-assembled-topology-has-no-network",
                        MVAUDataflowOpPaths.RESULT,
                        "a selected supplier or replay Kernel requires an assembled network",
                    ),
                )
            )
        return Decided(NetworkRef("mvau", cast(DataflowNetwork, network), association))
    return Decided(
        RegionRef("mvau.compute", cast(DataflowRegion, dependencies["compute_region"]), association)
    )


_EXTERNAL_SEQUENCE_REF = DependencyRef.problem(
    "external_weight_sequence",
    MVAUDataflowOpPaths.EXTERNAL_WEIGHT_SEQUENCE,
    _BEAT_SEQUENCE,
    absence=AbsenceMode.ALLOWS_ABSENT,
)
_PUMPED_MEMORY_REF = DependencyRef.decision(
    "pumped_memory",
    FINN_RTL_MEMSTREAM_PATHS.pumped_memory,
    _BOOL,
    absence=AbsenceMode.ALLOWS_ABSENT,
)


def _weight_is_supplied(dependencies: DependencyView) -> Answer[bool]:
    value = dependencies["supply_kernel"]
    return Decided(value is not ABSENT and value != NO_KERNEL)


def _weight_is_exposed(dependencies: DependencyView) -> Answer[bool]:
    if dependencies["region_form"] is MVAURegionDeclaration.STANDARD_EMBEDDED:
        return Decided(False)
    value = dependencies["supply_kernel"]
    return Decided(value is ABSENT or value == NO_KERNEL)


def _has_replay(dependencies: DependencyView) -> bool:
    value = dependencies["replay_kernel"]
    return value is not ABSENT and value != NO_KERNEL


def _needs_a_network(dependencies: DependencyView) -> Answer[bool]:
    """More than one node, for either of the two independent reasons.

    A selected supplier adds a delivery node; the decomposed compute member
    adds a replay node.  Either alone is enough to make the result a Network
    rather than a Region.
    """

    supplied = cast(Decided[bool], _weight_is_supplied(dependencies)).value
    return Decided(supplied or _has_replay(dependencies))


_SUPPLIED = EvaluatorSpec((_SUPPLY_KERNEL_REF,), _weight_is_supplied)
_EXPOSED = EvaluatorSpec((_COMPUTE_REGION_FORM_REF, _SUPPLY_KERNEL_REF), _weight_is_exposed)
_ASSEMBLED = EvaluatorSpec((_SUPPLY_KERNEL_REF, _REPLAY_KERNEL_REF), _needs_a_network)


def _interleaved_supply(dependencies: DependencyView) -> Answer[bool]:
    value = dependencies["supply_kernel"]
    return Decided(
        value is not ABSENT
        and value != NO_KERNEL
        and dependencies["region_form"] is MVAURegionDeclaration.BATCH_INTERLEAVED_STREAMED
    )


def _op_properties() -> tuple[DerivedProperty, ...]:
    return (
        DerivedProperty(
            MVAUDataflowOpPaths.PARAMETER_TOPOLOGY,
            _TOPOLOGY,
            EvaluatorSpec(
                (_COMPUTE_REGION_FORM_REF, _SUPPLY_KERNEL_REF), _derive_parameter_topology
            ),
        ),
        DerivedProperty(
            MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
            _SOURCE_ASSOCIATION,
            EvaluatorSpec(
                (
                    _SOURCE_DESCRIPTION_REF,
                    _TOPOLOGY_REF,
                    _COMPUTE_REGION_FORM_REF,
                    _COMPUTATION_REF,
                    _REPETITIONS_REF,
                    _MATRIX_WIDTH_REF,
                    _MATRIX_HEIGHT_REF,
                    _COMPUTE_SELECTED_REF,
                    _SUPPLY_SELECTED_REF,
                    _ADAPTER_SELECTED_REF,
                    _REPLAY_KERNEL_REF,
                ),
                _derive_source_association,
            ),
        ),
        DerivedProperty(
            MVAUDataflowOpPaths.NETWORK,
            _NETWORK,
            EvaluatorSpec(
                (
                    _SUPPLY_REGION_REF,
                    _COMPUTE_REGION_REF,
                    _ADAPTER_REGION_REF,
                    _REPLAY_REGION_REF,
                ),
                _derive_network,
            ),
            applies_if=_ASSEMBLED,
        ),
        DerivedProperty(
            MVAUDataflowOpPaths.NETWORK_VALIDATION,
            _NETWORK_REPORT,
            EvaluatorSpec(
                (DependencyRef.property("network", MVAUDataflowOpPaths.NETWORK, _NETWORK),),
                _derive_network_validation,
            ),
            applies_if=_ASSEMBLED,
        ),
        DerivedProperty(
            MVAUDataflowOpPaths.RESULT,
            DATAFLOW_OP_RESULT_SEMANTICS,
            EvaluatorSpec(
                (
                    _TOPOLOGY_REF,
                    _COMPUTE_REGION_REF,
                    _NETWORK_REF,
                    _SOURCE_ASSOCIATION_REF,
                    _REPLAY_KERNEL_REF,
                ),
                _derive_op_result,
            ),
        ),
    )


def _decomposed_supply_supported(dependencies: DependencyView) -> Answer[bool]:
    """The decomposed compute takes its weights at the boundary, for now.

    Re-attaching the memstream supplier is its own increment: the supplier
    connects to the dot product's weight demand, which is unchanged, so nothing
    about this is hard -- it is simply not done, and claiming otherwise would
    let a point resolve that no provider can build.
    """

    if not _has_replay(dependencies):
        return Decided(True)
    supply = dependencies["supply_kernel"]
    return Decided(supply is ABSENT or supply == NO_KERNEL)


def _op_constraints() -> tuple[Constraint, ...]:
    return (
        Constraint(
            MVAUDataflowOpPaths.DECOMPOSED_SUPPLY_SUPPORTED,
            EvaluatorSpec((_REPLAY_KERNEL_REF, _SUPPLY_KERNEL_REF), _decomposed_supply_supported),
        ),
        Constraint(
            MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED,
            EvaluatorSpec(
                (
                    _SUPPLY_OUTPUT_PORT_REF,
                    _COMPUTE_WEIGHT_PORT_REF,
                    _ADAPTER_KERNEL_REF,
                    _ADAPTER_REGION_REF,
                ),
                _weight_connection_supported,
            ),
            applies_if=_SUPPLIED,
        ),
        Constraint(
            MVAUDataflowOpPaths.EXPOSED_WEIGHT_SOURCE_AVAILABLE,
            EvaluatorSpec(
                (_EXTERNAL_SEQUENCE_REF, _COMPUTE_WEIGHT_PORT_REF, _COMPUTE_REGION_FORM_REF),
                _exposed_weight_source_available,
            ),
            applies_if=_EXPOSED,
        ),
        Constraint(
            MVAUDataflowOpPaths.INTERLEAVED_PUMPING_SUPPORTED,
            EvaluatorSpec((_PUMPED_MEMORY_REF,), _interleaved_pumping_supported),
            applies_if=EvaluatorSpec(
                (_SUPPLY_KERNEL_REF, _COMPUTE_REGION_FORM_REF), _interleaved_supply
            ),
        ),
        Constraint(
            MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
            EvaluatorSpec(
                (
                    _SOURCE_DESCRIPTION_REF,
                    _SOURCE_ASSOCIATION_REF,
                    _TOPOLOGY_REF,
                    _COMPUTE_REGION_REF,
                    _REPETITIONS_REF,
                    _COMPUTATION_REF,
                    _REPLAY_KERNEL_REF,
                    _REPLAY_REGION_REF,
                ),
                _source_association_valid,
            ),
        ),
        Constraint(
            MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
            EvaluatorSpec(
                (
                    DependencyRef.property(
                        "report", MVAUDataflowOpPaths.NETWORK_VALIDATION, _NETWORK_REPORT
                    ),
                ),
                _network_is_structurally_well_formed,
            ),
            # Whenever there is a Network at all, not only when a supplier made
            # it: a replay node produces one too, and its structure is exactly
            # as much in need of checking.
            applies_if=_ASSEMBLED,
        ),
    )


_OP_STRUCTURAL_CONSTRAINTS = (
    MVAU_COMPUTE_SELECTION.paths.region_structurally_well_formed,
    MVAU_WEIGHT_SUPPLY_SELECTION.paths.region_structurally_well_formed,
    MVAU_WEIGHT_ADAPTER_SELECTION.paths.region_structurally_well_formed,
    MVAU_REPLAY_SELECTION.paths.region_structurally_well_formed,
    MVAUDataflowOpPaths.WEIGHT_CONNECTION_SUPPORTED,
    MVAUDataflowOpPaths.DECOMPOSED_SUPPLY_SUPPORTED,
    MVAUDataflowOpPaths.EXPOSED_WEIGHT_SOURCE_AVAILABLE,
    MVAUDataflowOpPaths.INTERLEAVED_PUMPING_SUPPORTED,
    MVAUDataflowOpPaths.SOURCE_ASSOCIATION_VALID,
    MVAUDataflowOpPaths.NETWORK_STRUCTURALLY_WELL_FORMED,
)

_OP_FEASIBILITY_CONSTRAINTS = tuple(
    dict.fromkeys(
        (
            *MVAU_COMPUTE_SELECTION.feasibility_constraints(),
            *MVAU_WEIGHT_SUPPLY_SELECTION.feasibility_constraints(),
            *MVAU_WEIGHT_ADAPTER_SELECTION.feasibility_constraints(),
            *MVAU_REPLAY_SELECTION.feasibility_constraints(),
            # Physical coverage is a feasibility question, not only a binding
            # one.  A point no hardware can build should be refused while it is
            # still a design point, rather than resolved, elaborated, and then
            # turned away by a Kernel that was never consulted.
            *DECOMPOSED_MVAU_KERNELS.coverage_constraints,
            *_OP_STRUCTURAL_CONSTRAINTS,
        )
    )
)

#: Every choice that changes a Region or the Network.
#:
#: Structural readiness asks for exactly these.  A physical choice belongs in
#: the artifact set below and nowhere near this one: ``compute_pumping`` changes
#: how fast the datapath runs and moves no beat, so a point whose Regions and
#: Network are fully derived is structurally ready whether or not anyone has
#: decided to pump it.  Mixing the two made structural readiness answer ``None``
#: for a design that had resolved perfectly, which defeats the separation this
#: whole layer exists to draw.
_SEMANTIC_DECISIONS = (
    MVAU_COMPUTE_SELECTION.paths.kernel,
    *(item.path for kernel in MVAU_COMPUTE_SELECTION.kernels for item in kernel.spec.decisions),
    MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel,
    *(
        item.path
        for kernel in MVAU_WEIGHT_SUPPLY_SELECTION.kernels
        for item in kernel.spec.decisions
    ),
    MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel,
    MVAU_REPLAY_SELECTION.paths.kernel,
    *(item.path for kernel in MVAU_REPLAY_SELECTION.kernels for item in kernel.spec.decisions),
)

#: The physical Kernels' own choices.  There is no Kernel *selection* here --
#: one covers each Region -- but the choices those Kernels make are ordinary
#: decisions, and nothing can be built until they are committed.
_PHYSICAL_DECISIONS = tuple(
    item.path for kernel in DECOMPOSED_MVAU_KERNELS.hardware for item in kernel.spec.decisions
)

#: Every Region, Network, and association value the operation derives.
_SEMANTIC_PROPERTIES = (
    MVAU_COMPUTE_SELECTION.paths.region,
    MVAU_COMPUTE_SELECTION.paths.selected_kernel,
    MVAUDataflowOpPaths.COMPUTE_REGION_FORM,
    MVAU_WEIGHT_SUPPLY_SELECTION.paths.region,
    MVAU_WEIGHT_ADAPTER_SELECTION.paths.region,
    MVAU_REPLAY_SELECTION.paths.region,
    MVAU_REPLAY_SELECTION.paths.selected_kernel,
    MVAUDataflowOpPaths.PARAMETER_TOPOLOGY,
    MVAUDataflowOpPaths.SOURCE_ASSOCIATION,
    MVAUDataflowOpPaths.NETWORK,
    MVAUDataflowOpPaths.NETWORK_VALIDATION,
    MVAUDataflowOpPaths.RESULT,
)

#: Every physical value the decomposed hardware derives.  Artifact inputs, so a
#: point is not ready to *build* until they resolve -- but it is ready to be a
#: design long before that.
_PHYSICAL_PROPERTIES = tuple(
    item.path for kernel in DECOMPOSED_MVAU_KERNELS.hardware for item in kernel.spec.properties
)

_OP_DECISIONS = (*_SEMANTIC_DECISIONS, *_PHYSICAL_DECISIONS)
_OP_PROPERTIES = (*_SEMANTIC_PROPERTIES, *_PHYSICAL_PROPERTIES)


def _op_constraint_sets() -> tuple[ConstraintSet, ...]:
    return (
        ConstraintSet("mvau_op_structural", _OP_STRUCTURAL_CONSTRAINTS),
        ConstraintSet("mvau_op_feasibility", _OP_FEASIBILITY_CONSTRAINTS),
    )


def _op_readiness_profiles() -> tuple[ReadinessProfile, ...]:
    return (
        ReadinessProfile(
            "mvau_op_structural",
            decisions=_SEMANTIC_DECISIONS,
            properties=_SEMANTIC_PROPERTIES,
            constraints=_OP_STRUCTURAL_CONSTRAINTS,
        ),
        ReadinessProfile(
            "artifact_inputs",
            decisions=_OP_DECISIONS,
            properties=_OP_PROPERTIES,
            constraints=_OP_FEASIBILITY_CONSTRAINTS,
        ),
    )


def build_mvau_dataflow_op_spec() -> DesignSpaceSpec:
    """Build one flat MVAU operation-level design-space specification."""

    additions = DesignSpaceSpec(
        properties=_op_properties(),
        constraints=_op_constraints(),
        constraint_sets=_op_constraint_sets(),
        readiness_profiles=_op_readiness_profiles(),
    )
    assembled = assemble_specs(
        (
            MVAU_COMPUTE_SELECTION.build_spec(),
            # Present only when the decomposed compute member is selected.
            MVAU_REPLAY_SELECTION.build_spec(),
            # The hardware that covers the decomposed Regions, gated the same
            # way.  Not a pool: one Kernel covers each Region, so there is no
            # choice and the design space claims none.
            *(item.spec for item in DECOMPOSED_MVAU_KERNELS.hardware),
            MVAU_WEIGHT_SUPPLY_SELECTION.build_spec(),
            MVAU_WEIGHT_ADAPTER_SELECTION.build_spec(),
            MVAU_PROBLEM_SPEC,
            additions,
        )
    )
    # Every handle the physical Kernels read has to name something this space
    # really declares, with the right kind and the right type.  ``Engine
    # .validate`` cannot ask this -- a path a Kernel merely reads is not part of
    # its own specification -- so a forgotten scope would validate cleanly here
    # and fail at binding time with an engine request error instead.
    check_declared_references(assembled, DECOMPOSED_MVAU_KERNELS.hardware)
    return assembled


MVAU_DATAFLOW_OP_SPEC = build_mvau_dataflow_op_spec()

MVAU_SELECTIONS: tuple[KernelSelection, ...] = (
    MVAU_COMPUTE_SELECTION,
    MVAU_REPLAY_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
    MVAU_WEIGHT_ADAPTER_SELECTION,
)

MVAU_LEGACY_DATAFLOW_OP_SPEC = MVAU_DATAFLOW_OP_SPEC
MVAU_LEGACY_SELECTIONS = MVAU_SELECTIONS


__all__ = [
    "BindingLocalStateDestination",
    "CoordinateMappingKind",
    "DataflowOpResult",
    "MVAU_COMPUTE_SELECTION",
    "MVAU_DATAFLOW_OP_SPEC",
    "MVAU_LEGACY_DATAFLOW_OP_SPEC",
    "MVAU_LEGACY_SELECTIONS",
    "MVAU_REPLAY_SELECTION",
    "MVAU_SELECTIONS",
    "MVAU_WEIGHT_ADAPTER_SELECTION",
    "MVAU_WEIGHT_SUPPLY_SELECTION",
    "MVAUDataflowOpPaths",
    "MVAUParameterTopology",
    "MVAUSourceAssociation",
    "MVAUSourceDescription",
    "NetworkRef",
    "RegionRef",
    "SemanticOperandDestination",
    "SourceOperandAssociation",
    "SourceOperandDestination",
    "build_mvau_dataflow_op_spec",
]
