# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Real operation/Design/Kernel fixtures for selected-transform lifecycle tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Finding, FindingKind, QualifiedPath
from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.designs.design import (
    DataflowDesign,
    EdgeSink,
    KernelChoice,
    NetworkBoundary,
    NetworkEdge,
    SelectedGraph,
)
from finn.dataflow.kernels.kernel import Kernel, RegionDeclaration
from finn.dataflow.model import (
    BeatSequence,
    BoundaryContract,
    DataflowNetwork,
    DataflowRegion,
    Edge,
    InputInterface,
    InternalInput,
    LogicalSchedule,
    NetworkNode,
    Operand,
    OutputInterface,
    Port,
    PositionMap,
    RectangularDomain,
    RegionEndpoint,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    SinkContract,
)
from finn.dataflow.model.refs import DataflowOperandRef, RegionInputRef, RegionOutputRef
from finn.dataflow.ops.base import DATAFLOW_DOMAIN, DataflowOp, DataflowOpError
from finn.dataflow.ops.mapping import CoordinateMapping
from finn.dataflow.ops.native import operation_choice_schema
from finn.dataflow.ops.persistence import assign_dataflow_scope_ids
from finn.dataflow.ops.schema import Attribute, OpInput, OpOutput
from finn.dataflow.ops.selected import (
    SELECTED_DECLARATION_ID,
    SELECTED_DECLARATION_VERSION,
    ComputationOwner,
    ConstructionIdentity,
    ConstructionInputs,
    ConstructionRegistry,
    EncodedSourceSemantics,
    GraphNodeBinding,
    GraphSlotKind,
    GraphSlotRef,
    InterfaceBinding,
    InterfaceDirection,
    OwnerKind,
    PositionRelation,
    QualifiedInterfaceRef,
    RecordedChoice,
    RequiredSupply,
    SelectedConstruction,
    SelectedGraphDeclaration,
    SelectedGraphSnapshot,
    SelectedInitializerInput,
    SelectionFacts,
    SourceDirection,
    SourceOperandKey,
    SourceProvenance,
    SourceValueBinding,
    SourceValueRef,
    build_selected_snapshot,
    set_frozen_initializer,
    validate_construction_inputs,
)
from finn.dataflow.ops.selected_registry import DEFAULT_SELECTED_CONSTRUCTIONS
from finn.dataflow.ops.selected_transform_registry import DEFAULT_SELECTED_TRANSFORMS
from finn.dataflow.ops.selected_transforms import (
    BOUNDED_CLEANUP_TRANSFORM,
    ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM,
    READABLE_NAMES_TRANSFORM,
    TRANSFORM_VERSION,
    SelectedTransformAuthorization,
    SelectedTransformRegistry,
)
from finn.dataflow.ops.selected_verification import (
    frozen_initializer_for_source,
    verify_normalized_selected_snapshot,
)
from finn.dataflow.ops.tensor_summary import FrozenInitializer
from finn.dataflow.space.dataflow_value_semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.space.declarations import Decision, Input, Subspace, allow_absent, derived
from finn.dataflow.space.occurrence import ProjectionAssessment

IDENTITY_CHAIN_CONSTRUCTION_FAMILY = "test.selected.identity_chain"
IDENTITY_CHAIN_CONSTRUCTION_VERSION = "1"
IDENTITY_CHAIN_SOURCE_FAMILY = "test.dataflow.identity_chain"
IDENTITY_CHAIN_SOURCE_VERSION = "1"
IDENTITY_CHAIN_SOURCE_SEMANTICS = "test.source.identity_chain"
IDENTITY_CHAIN_SOURCE_SEMANTICS_VERSION = 1
IDENTITY_CHAIN_GRAPH_NAME = "selected_identity_chain"
IDENTITY_ELIDED_FORM = "identity_elided"
REFERENCE_REGIONS = ("none", "left", "right")

ACTIVATION_KEY = SourceOperandKey("activation", SourceDirection.INPUT, 0)
RESULT_KEY = SourceOperandKey("result", SourceDirection.OUTPUT, 0)


def _sequence(lanes: int) -> BeatSequence:
    return BeatSequence(
        lanes,
        tuple(tuple((fold * lanes + lane,) for lane in range(lanes)) for fold in range(2 // lanes)),
    )


def _requirements() -> ScheduledInputRequirements:
    requirements: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {
        ((), (0,)): 1,
        ((), (1,)): 1,
    }
    return ScheduledInputRequirements(
        requirements,
        schedule_domain=RectangularDomain(()),
        position_domain=RectangularDomain((2,)),
    )


def _availability() -> ScheduledOutputAvailability:
    availability: dict[tuple[int, ...], tuple[int, ...]] = {
        (0,): (),
        (1,): (),
    }
    return ScheduledOutputAvailability(
        availability,
        position_domain=RectangularDomain((2,)),
        schedule_domain=RectangularDomain(()),
    )


def construct_identity_region(
    shape: tuple[int, ...],
    datatype: Any,
    input_lanes: int,
    output_lanes: int,
) -> DataflowRegion:
    if shape != (2,) or input_lanes not in (1, 2) or output_lanes not in (1, 2):
        raise ValueError("the identity-chain fixture supports shape (2,) and lane widths 1 or 2")
    operand = Operand("value", datatype, shape)
    return DataflowRegion(
        LogicalSchedule(()),
        (
            InputInterface(
                Port("in", operand, _sequence(input_lanes)),
                _requirements(),
            ),
        ),
        (
            OutputInterface(
                Port("out", operand, _sequence(output_lanes)),
                _availability(),
            ),
        ),
    )


def construct_identity_consumer_region(
    shape: tuple[int, ...],
    datatype: Any,
    lanes: int,
    reference_region: str,
) -> DataflowRegion:
    if reference_region not in REFERENCE_REGIONS:
        raise ValueError("identity-chain reference region must be none, left, or right")
    operand = Operand("value", datatype, shape)
    output_datatype = datatype if reference_region == "none" else DataType["INT32"]
    inputs: tuple[InputInterface | InternalInput, ...] = (
        InputInterface(Port("in", operand, _sequence(lanes)), _requirements()),
    )
    if reference_region != "none":
        inputs = (*inputs, InternalInput(Operand("reference", datatype, shape), _requirements()))
    return DataflowRegion(
        LogicalSchedule(()),
        inputs,
        (
            OutputInterface(
                Port("out", Operand("result", output_datatype, shape), _sequence(lanes)),
                _availability(),
            ),
        ),
    )


class IdentityRegionKernel(Kernel):
    id = "identity_region"
    version = "1"

    shape = Input(tuple)
    datatype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    input_lanes = Input(int)
    output_lanes = Input(int)

    region = RegionDeclaration(
        family="test.identity_region",
        version="1",
        construct=construct_identity_region,
        shape=shape,
        datatype=datatype,
        input_lanes=input_lanes,
        output_lanes=output_lanes,
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        del parameters
        return ComponentABI("identity_region", ())


class IdentityConsumerKernel(Kernel):
    id = "identity_consumer"
    version = "1"

    shape = Input(tuple)
    datatype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    lanes = Input(int)
    reference_region = Input(str)

    region = RegionDeclaration(
        family="test.identity_consumer",
        version="1",
        construct=construct_identity_consumer_region,
        shape=shape,
        datatype=datatype,
        lanes=lanes,
        reference_region=reference_region,
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, object]) -> ComponentABI:
        del parameters
        return ComponentABI("identity_consumer", ())


class IdentityChainDesign(DataflowDesign):
    """Four real Regions: producer -> left -> right -> consumer."""

    id = "identity_chain"
    version = "1"

    shape = Input(tuple)
    datatype = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    activation_initializer = Input(FrozenInitializer, allow_absent=True)
    reference_region = Input(str)
    grouping = Decision(int, values=(1, 2))

    @derived(int, grouping=grouping)
    def unit_lanes(*, grouping: int) -> int:
        del grouping
        return 1

    producer = KernelChoice(
        Subspace(
            IdentityRegionKernel,
            shape=shape,
            datatype=datatype,
            input_lanes=unit_lanes,
            output_lanes=unit_lanes,
        )
    )
    left = KernelChoice(
        Subspace(
            IdentityRegionKernel,
            shape=shape,
            datatype=datatype,
            input_lanes=unit_lanes,
            output_lanes=grouping,
        )
    )
    right = KernelChoice(
        Subspace(
            IdentityRegionKernel,
            shape=shape,
            datatype=datatype,
            input_lanes=grouping,
            output_lanes=grouping,
        )
    )
    consumer = KernelChoice(
        Subspace(
            IdentityConsumerKernel,
            shape=shape,
            datatype=datatype,
            lanes=grouping,
            reference_region=reference_region,
        )
    )

    producer_to_left = NetworkEdge(producer.output("out"), EdgeSink(left.input("in")))
    left_to_right = NetworkEdge(left.output("out"), EdgeSink(right.input("in")))
    right_to_consumer = NetworkEdge(right.output("out"), EdgeSink(consumer.input("in")))

    activation = NetworkBoundary(producer.input("in"))
    result = NetworkBoundary(consumer.output("out"))


def _design(root: DataflowOp) -> IdentityChainDesign:
    return cast(IdentityChainDesign, root.design)


class IdentityChainOp(DataflowOp):
    family: ClassVar[str] = IDENTITY_CHAIN_SOURCE_FAMILY
    family_version: ClassVar[str] = IDENTITY_CHAIN_SOURCE_VERSION
    schema_version: ClassVar[int] = 1

    activation = OpInput(index=0, operand="value", correspondence=CoordinateMapping.IDENTITY)
    result = OpOutput(index=0, operand="value", correspondence=CoordinateMapping.IDENTITY)
    reference_region = Attribute(str, default="none")

    design = Subspace(
        IdentityChainDesign,
        shape=activation.shape,
        datatype=activation.datatype,
        activation_initializer=allow_absent(activation.initializer_value),
        reference_region=reference_region,
    )

    def selected_dataflow(self) -> ProjectionAssessment[DataflowNetwork] | None:
        return _design(self).dataflow

    def selected_design(self) -> object:
        return _design(self)

    def selected_source_semantics(self) -> object:
        return EncodedSourceSemantics(
            IDENTITY_CHAIN_SOURCE_SEMANTICS,
            IDENTITY_CHAIN_SOURCE_SEMANTICS_VERSION,
            {
                "operation": "identity",
                "reference_region": self.source.attributes["reference_region"],
            },
        )

    def selected_construction_identity(self, semantics: object) -> object:
        if not isinstance(semantics, IdentityChainSourceSemantics):
            raise DataflowOpError("identity-chain source semantics are invalid")
        return ConstructionIdentity(
            IDENTITY_CHAIN_CONSTRUCTION_FAMILY,
            IDENTITY_CHAIN_CONSTRUCTION_VERSION,
            "canonical",
        )

    def operand_references(
        self, network: DataflowNetwork
    ) -> Mapping[str, tuple[DataflowOperandRef, ...]]:
        del network
        return {
            "activation": (RegionInputRef("producer", "value"),),
            "result": (RegionOutputRef("consumer", "result"),),
        }

    def expected_for(self, source: Any) -> dict[str, tuple[tuple[int, ...] | None, Any]]:
        activation = source.operand("activation")
        datatype = (
            activation.datatype
            if self.source.attributes["reference_region"] == "none"
            else DataType["INT32"]
        )
        return {"result": (activation.shape, datatype)}

    def execute_node(self, context: Any, graph: Any) -> None:
        del graph
        context[self.onnx_node.output[0]] = np.asarray(context[self.onnx_node.input[0]]).copy()


@dataclass(frozen=True, slots=True)
class IdentityChainSourceSemantics:
    reference_region: str | None


@dataclass(frozen=True, slots=True)
class IdentityChainParameters:
    grouping: int
    shape: tuple[int, ...]
    logical_datatype: str
    carrier_dtype: int
    reference_region: str | None


def decode_identity_chain_semantics(
    value: EncodedSourceSemantics,
) -> IdentityChainSourceSemantics:
    payload = dict(cast(Mapping[str, object], value.payload))
    if (
        value.identity != IDENTITY_CHAIN_SOURCE_SEMANTICS
        or value.version != IDENTITY_CHAIN_SOURCE_SEMANTICS_VERSION
        or set(payload) != {"operation", "reference_region"}
        or payload["operation"] != "identity"
        or payload["reference_region"] not in REFERENCE_REGIONS
    ):
        raise ValueError("unsupported identity-chain source semantics")
    reference = payload["reference_region"]
    return IdentityChainSourceSemantics(None if reference == "none" else reference)


def _source_operand(source: SourceProvenance, key: SourceOperandKey) -> SourceValueRef:
    matches = tuple(item for item in source.operands if item.key == key)
    if len(matches) != 1:
        raise ValueError(f"identity-chain source is missing {key.operand_id!r}")
    return matches[0]


def derive_identity_chain_facts(
    identity: ConstructionIdentity,
    source: SourceProvenance,
    semantics: IdentityChainSourceSemantics,
    choices: tuple[RecordedChoice, ...],
) -> SelectionFacts[IdentityChainSourceSemantics, IdentityChainParameters]:
    if (identity.family, identity.version) != (
        IDENTITY_CHAIN_CONSTRUCTION_FAMILY,
        IDENTITY_CHAIN_CONSTRUCTION_VERSION,
    ):
        raise ValueError("unsupported identity-chain construction")
    if identity.form == "canonical":
        if identity.form_version != 1 or identity.form_arguments:
            raise ValueError("canonical identity-chain form takes no arguments")
    elif identity.form == IDENTITY_ELIDED_FORM:
        if identity.form_version != 1 or identity.form_arguments not in (
            (("region_id", "left"),),
            (("region_id", "right"),),
        ):
            raise ValueError("identity_elided requires region_id left or right")
    else:
        raise ValueError("unsupported identity-chain form")
    if (source.family, source.family_version) != (
        IDENTITY_CHAIN_SOURCE_FAMILY,
        IDENTITY_CHAIN_SOURCE_VERSION,
    ):
        raise ValueError("identity-chain source family changed")
    activation = _source_operand(source, ACTIVATION_KEY)
    result = _source_operand(source, RESULT_KEY)
    if (
        activation.shape != (2,)
        or result.shape != activation.shape
        or activation.carrier_dtype != TensorProto.FLOAT
        or result.carrier_dtype != activation.carrier_dtype
        or activation.logical_datatype != "INT8"
        or result.logical_datatype != ("INT8" if semantics.reference_region is None else "INT32")
        or result.initializer_present
    ):
        raise ValueError("identity-chain source operands must be matching INT8 vectors")
    if activation.initializer_present != (semantics.reference_region is not None):
        raise ValueError("identity-chain reference supply must match initializer presence")
    if tuple(item.path for item in choices) != ("design.grouping",):
        raise ValueError("identity-chain selection requires exactly design.grouping")
    grouping = choices[0].value
    if type(grouping) is not int or grouping not in (1, 2):
        raise ValueError("identity-chain grouping must be 1 or 2")
    return SelectionFacts(
        identity,
        source,
        semantics,
        choices,
        IdentityChainParameters(
            grouping,
            activation.shape,
            activation.logical_datatype,
            activation.carrier_dtype,
            semantics.reference_region,
        ),
    )


def _region_for(parameters: IdentityChainParameters, region_id: str) -> DataflowRegion:
    grouping = parameters.grouping
    if region_id == "consumer":
        return construct_identity_consumer_region(
            parameters.shape,
            DataType[parameters.logical_datatype],
            grouping,
            "none" if parameters.reference_region is None else parameters.reference_region,
        )
    widths = {
        "producer": (1, 1),
        "left": (1, grouping),
        "right": (grouping, grouping),
    }
    input_lanes, output_lanes = widths[region_id]
    return construct_identity_region(
        parameters.shape,
        DataType[parameters.logical_datatype],
        input_lanes,
        output_lanes,
    )


def _active_regions(identity: ConstructionIdentity) -> tuple[str, ...]:
    regions = ("producer", "left", "right", "consumer")
    if identity.form == "canonical":
        return regions
    removed = cast(str, dict(identity.form_arguments)["region_id"])
    return tuple(region_id for region_id in regions if region_id != removed)


def project_identity_chain_network(
    facts: SelectionFacts[IdentityChainSourceSemantics, IdentityChainParameters],
) -> DataflowNetwork:
    active = _active_regions(facts.construction)
    nodes = tuple(
        NetworkNode(region_id, _region_for(facts.parameters, region_id))
        for region_id in sorted(active)
    )
    edges = []
    for source, target in zip(active, active[1:]):
        source_region = _region_for(facts.parameters, source)
        edges.append(
            Edge(
                f"{source}_to_{target}",
                RegionEndpoint(source, "out"),
                (
                    SinkContract(
                        RegionEndpoint(target, "in"),
                        PositionMap.identity(
                            source_region.output_interface("out").port.beat_sequence.image_set
                        ),
                    ),
                ),
            )
        )
    producer = _region_for(facts.parameters, "producer")
    consumer = _region_for(facts.parameters, "consumer")
    return DataflowNetwork(
        nodes,
        tuple(sorted(edges, key=lambda edge: edge.id)),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint("producer", "in"),
                producer.input_interface("in").port.beat_sequence,
            ),
            BoundaryContract(
                "result",
                RegionEndpoint("consumer", "out"),
                consumer.output_interface("out").port.beat_sequence,
            ),
        ),
    )


def _tensor(name: str) -> Any:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [2])


def construct_identity_chain_snapshot(
    facts: SelectionFacts[IdentityChainSourceSemantics, IdentityChainParameters],
    construction_inputs: ConstructionInputs,
) -> SelectedGraphSnapshot:
    reference_region = facts.parameters.reference_region
    required_initializers = () if reference_region is None else (ACTIVATION_KEY,)
    frozen = validate_construction_inputs(
        facts,
        construction_inputs,
        required_initializers,
    )
    active = _active_regions(facts.construction)
    output_names = {
        "producer": "P",
        "left": "L",
        "right": "R",
    }
    nodes: list[Any] = []
    graph_nodes: list[GraphNodeBinding] = []
    interface_bindings: list[InterfaceBinding] = []
    ownership: list[ComputationOwner] = []
    value_info: list[Any] = []
    root_value = "X" if reference_region is None else "X_initializer"
    current = root_value
    domain = RectangularDomain(facts.parameters.shape)
    relation = PositionRelation.direct(domain, domain)
    for region_id in active:
        if region_id == "consumer":
            break
        node_id = f"{region_id}.identity"
        output = output_names[region_id]
        nodes.append(helper.make_node("Identity", [current], [output], name=node_id))
        graph_nodes.append(GraphNodeBinding(node_id, len(nodes) - 1))
        input_anchors = [GraphSlotRef(GraphSlotKind.NODE_INPUT, node_id, 0)]
        if region_id == "producer":
            input_anchors.insert(
                0,
                GraphSlotRef(
                    GraphSlotKind.GRAPH_INPUT
                    if reference_region is None
                    else GraphSlotKind.INITIALIZER,
                    IDENTITY_CHAIN_GRAPH_NAME if reference_region is None else root_value,
                    0 if reference_region is None else None,
                ),
            )
        output_anchors = [GraphSlotRef(GraphSlotKind.NODE_OUTPUT, node_id, 0)]
        interface_bindings.extend(
            (
                InterfaceBinding(
                    QualifiedInterfaceRef(
                        region_id,
                        InterfaceDirection.INPUT,
                        "value",
                        "in",
                    ),
                    current,
                    relation,
                    tuple(input_anchors),
                ),
                InterfaceBinding(
                    QualifiedInterfaceRef(
                        region_id,
                        InterfaceDirection.OUTPUT,
                        "value",
                        "out",
                    ),
                    output,
                    relation,
                    tuple(output_anchors),
                ),
            )
        )
        ownership.append(ComputationOwner(OwnerKind.REGION, region_id, (node_id,)))
        if output != "Y":
            value_info.append(_tensor(output))
        current = output

    consumer_nodes: tuple[str, ...]
    reference_value: str | None = None
    if reference_region is None:
        node_id = "consumer.identity"
        nodes.append(helper.make_node("Identity", [current], ["Y"], name=node_id))
        graph_nodes.append(GraphNodeBinding(node_id, len(nodes) - 1))
        consumer_nodes = (node_id,)
        consumer_input_anchors = (GraphSlotRef(GraphSlotKind.NODE_INPUT, node_id, 0),)
        consumer_output_anchor = GraphSlotRef(GraphSlotKind.NODE_OUTPUT, node_id, 0)
    else:
        removed = (
            cast(str, dict(facts.construction.form_arguments)["region_id"])
            if facts.construction.form == IDENTITY_ELIDED_FORM
            else None
        )
        effective_reference = (
            {"left": "producer", "right": "left"}[reference_region]
            if removed == reference_region
            else reference_region
        )
        reference_value = output_names[effective_reference]
        zero_node = "consumer.reference_zero"
        add_node = "consumer.add"
        nodes.append(
            helper.make_node(
                "Mul",
                [reference_value, "reference_zero_constant"],
                ["reference_zero"],
                name=zero_node,
            )
        )
        graph_nodes.append(GraphNodeBinding(zero_node, len(nodes) - 1))
        nodes.append(helper.make_node("Add", [current, "reference_zero"], ["Y"], name=add_node))
        graph_nodes.append(GraphNodeBinding(add_node, len(nodes) - 1))
        consumer_nodes = (zero_node, add_node)
        consumer_input_anchors = (GraphSlotRef(GraphSlotKind.NODE_INPUT, add_node, 0),)
        consumer_output_anchor = GraphSlotRef(GraphSlotKind.NODE_OUTPUT, add_node, 0)
        interface_bindings.append(
            InterfaceBinding(
                QualifiedInterfaceRef(
                    "consumer",
                    InterfaceDirection.INPUT,
                    "reference",
                    None,
                ),
                reference_value,
                relation,
                (GraphSlotRef(GraphSlotKind.NODE_INPUT, zero_node, 0),),
            )
        )
        value_info.append(_tensor("reference_zero"))

    interface_bindings.extend(
        (
            InterfaceBinding(
                QualifiedInterfaceRef(
                    "consumer",
                    InterfaceDirection.INPUT,
                    "value",
                    "in",
                ),
                current,
                relation,
                consumer_input_anchors,
            ),
            InterfaceBinding(
                QualifiedInterfaceRef(
                    "consumer",
                    InterfaceDirection.OUTPUT,
                    "result",
                    "out",
                ),
                "Y",
                relation,
                (
                    consumer_output_anchor,
                    GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, IDENTITY_CHAIN_GRAPH_NAME, 0),
                ),
            ),
        )
    )
    ownership.append(ComputationOwner(OwnerKind.REGION, "consumer", consumer_nodes))

    graph_inputs = [_tensor("X")] if reference_region is None else []
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                nodes,
                IDENTITY_CHAIN_GRAPH_NAME,
                graph_inputs,
                [_tensor("Y")],
                value_info=value_info,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    if reference_region is not None:
        set_frozen_initializer(model, root_value, frozen[ACTIVATION_KEY])
        model.set_initializer(
            "reference_zero_constant",
            np.zeros(facts.parameters.shape, dtype=np.float32),
        )
    for name in (
        *(item.name for item in graph_inputs),
        *(item.name for item in value_info),
        root_value,
        *(("reference_zero_constant",) if reference_region is not None else ()),
    ):
        model.set_tensor_datatype(name, DataType["INT8"])
    model.set_tensor_datatype(
        "Y",
        DataType["INT8" if reference_region is None else "INT32"],
    )
    if reference_region is not None:
        model.set_tensor_datatype("reference_zero", DataType["INT32"])
    source_bindings: tuple[SourceValueBinding, ...]
    supplies: tuple[RequiredSupply, ...]
    if reference_region is None:
        source_bindings = (
            SourceValueBinding(
                ACTIVATION_KEY,
                "X",
                relation,
                (GraphSlotRef(GraphSlotKind.GRAPH_INPUT, IDENTITY_CHAIN_GRAPH_NAME, 0),),
            ),
            SourceValueBinding(
                RESULT_KEY,
                "Y",
                relation,
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, IDENTITY_CHAIN_GRAPH_NAME, 0),),
            ),
        )
        supplies = ()
    else:
        assert reference_value is not None
        effective_reference = next(
            region_id
            for region_id in ("right", "left", "producer")
            if output_names[region_id] == reference_value
        )
        derivation_nodes = tuple(
            f"{region_id}.identity"
            for region_id in active
            if region_id != "consumer"
            and ("producer", "left", "right").index(region_id)
            <= ("producer", "left", "right").index(effective_reference)
        )
        source_bindings = (
            SourceValueBinding(
                ACTIVATION_KEY,
                reference_value,
                relation,
                (
                    GraphSlotRef(
                        GraphSlotKind.NODE_OUTPUT,
                        f"{effective_reference}.identity",
                        0,
                    ),
                ),
            ),
            SourceValueBinding(
                RESULT_KEY,
                "Y",
                relation,
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, IDENTITY_CHAIN_GRAPH_NAME, 0),),
            ),
        )
        supplies = (
            RequiredSupply(
                QualifiedInterfaceRef(
                    "consumer",
                    InterfaceDirection.INPUT,
                    "reference",
                    None,
                ),
                root_value,
                ACTIVATION_KEY,
                derivation_nodes,
            ),
        )
    declaration = SelectedGraphDeclaration(
        SELECTED_DECLARATION_ID,
        SELECTED_DECLARATION_VERSION,
        "",
        facts.construction,
        facts.source,
        facts.choices,
        tuple(graph_nodes),
        tuple(interface_bindings),
        source_bindings,
        supplies,
        tuple(ownership),
    )
    return build_selected_snapshot(model, declaration)


def verify_identity_chain_snapshot(
    snapshot: SelectedGraphSnapshot,
    facts: SelectionFacts[IdentityChainSourceSemantics, IdentityChainParameters],
) -> tuple[Finding, ...]:
    inputs = ConstructionInputs()
    if facts.parameters.reference_region is not None:
        frozen = frozen_initializer_for_source(snapshot, ACTIVATION_KEY)
        if frozen is None:
            return (
                Finding(
                    FindingKind.REJECTION,
                    "selected-identity-chain-source-root",
                    QualifiedPath("selected.identity_chain.activation"),
                    "initializer-backed identity chain has no unique source root",
                ),
            )
        inputs = ConstructionInputs(((ACTIVATION_KEY, frozen),))
    try:
        expected = construct_identity_chain_snapshot(facts, inputs)
    except (TypeError, ValueError) as error:
        return (
            Finding(
                FindingKind.REJECTION,
                "selected-identity-chain-expected-construction",
                QualifiedPath("selected.identity_chain"),
                str(error),
            ),
        )
    return verify_normalized_selected_snapshot(
        snapshot,
        expected,
        finding_code="selected-identity-chain-form-mismatch",
        path="selected.identity_chain",
        message="selected identity chain differs from its projected form",
    )


def _activation_initializer_required(facts: SelectionFacts[Any, Any]) -> bool:
    if not isinstance(facts.parameters, IdentityChainParameters):
        raise TypeError("identity-chain initializer predicate received the wrong parameters")
    return facts.parameters.reference_region is not None


IDENTITY_CHAIN_SELECTED_CONSTRUCTION = SelectedConstruction(
    family=IDENTITY_CHAIN_CONSTRUCTION_FAMILY,
    version=IDENTITY_CHAIN_CONSTRUCTION_VERSION,
    source_semantics_identity=IDENTITY_CHAIN_SOURCE_SEMANTICS,
    source_semantics_version=IDENTITY_CHAIN_SOURCE_SEMANTICS_VERSION,
    admitted_forms=("canonical", IDENTITY_ELIDED_FORM),
    choice_paths=("design.grouping",),
    initializer_inputs=(
        SelectedInitializerInput(
            ACTIVATION_KEY,
            IdentityChainDesign.activation_initializer,
            _activation_initializer_required,
        ),
    ),
    decode_source_semantics=decode_identity_chain_semantics,
    derive_facts=derive_identity_chain_facts,
    project=project_identity_chain_network,
    construct=construct_identity_chain_snapshot,
    verify=verify_identity_chain_snapshot,
)

IdentityChainDesign.selected_graph = SelectedGraph(IDENTITY_CHAIN_SELECTED_CONSTRUCTION)

IDENTITY_CHAIN_CONSTRUCTIONS = ConstructionRegistry(
    {
        (
            IDENTITY_CHAIN_CONSTRUCTION_FAMILY,
            IDENTITY_CHAIN_CONSTRUCTION_VERSION,
        ): IDENTITY_CHAIN_SELECTED_CONSTRUCTION
    },
    {
        (
            IDENTITY_CHAIN_CONSTRUCTION_FAMILY,
            IDENTITY_CHAIN_CONSTRUCTION_VERSION,
        ): operation_choice_schema(IdentityChainOp)
    },
)

_SAME_FORM_SOURCE_FORMS = ("canonical", IDENTITY_ELIDED_FORM)
IDENTITY_CHAIN_TRANSFORM_AUTHORIZATIONS = (
    SelectedTransformAuthorization(
        READABLE_NAMES_TRANSFORM,
        TRANSFORM_VERSION,
        _SAME_FORM_SOURCE_FORMS,
        None,
        ("replace_nodes", "rename_values"),
    ),
    SelectedTransformAuthorization(
        BOUNDED_CLEANUP_TRANSFORM,
        TRANSFORM_VERSION,
        _SAME_FORM_SOURCE_FORMS,
        None,
        (
            "replace_nodes",
            "node_order",
            "remove_graph_inputs",
            "remove_value_info",
            "remove_initializers",
            "set_initializers",
            "replace_quantization_annotations",
        ),
    ),
    SelectedTransformAuthorization(
        ELIDE_EQUAL_WIDTH_IDENTITY_TRANSFORM,
        TRANSFORM_VERSION,
        ("canonical",),
        IDENTITY_ELIDED_FORM,
        (
            "remove_nodes",
            "replace_nodes",
            "remove_value_info",
            "replace_quantization_annotations",
        ),
    ),
)
IDENTITY_CHAIN_TRANSFORMS = SelectedTransformRegistry(
    {
        (
            IDENTITY_CHAIN_CONSTRUCTION_FAMILY,
            IDENTITY_CHAIN_CONSTRUCTION_VERSION,
        ): IDENTITY_CHAIN_TRANSFORM_AUTHORIZATIONS
    }
)


def identity_chain_model(
    *,
    reference_region: str | None = None,
    activation: np.ndarray | None = None,
) -> ModelWrapper:
    if reference_region is not None and reference_region not in {"left", "right"}:
        raise ValueError("reference_region must be left, right, or None")
    attributes: dict[str, object] = {}
    if reference_region is not None:
        attributes["reference_region"] = reference_region
    node = helper.make_node(
        "IdentityChain",
        ["activation"],
        ["result"],
        domain=DATAFLOW_DOMAIN,
        name="identity_chain0",
        **attributes,
    )
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [node],
                "identity_chain_source",
                [_tensor("activation")],
                [_tensor("result")],
            ),
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid(DATAFLOW_DOMAIN, 1),
            ],
        )
    )
    model.set_tensor_datatype("activation", DataType["INT8"])
    model.set_tensor_datatype(
        "result",
        DataType["INT8" if reference_region is None else "INT32"],
    )
    if reference_region is not None:
        values = (
            np.asarray([3.0, -2.0], dtype=np.float32)
            if activation is None
            else np.asarray(activation, dtype=np.float32)
        )
        if values.shape != (2,):
            raise ValueError("identity-chain activation initializer must have shape (2,)")
        model.set_initializer("activation", values)
    assign_dataflow_scope_ids(model, domain=DATAFLOW_DOMAIN)
    return model


def configured_identity_chain(
    grouping: int = 1,
    *,
    reference_region: str | None = None,
    activation: np.ndarray | None = None,
) -> tuple[ModelWrapper, IdentityChainOp]:
    model = identity_chain_model(
        reference_region=reference_region,
        activation=activation,
    )
    bound = IdentityChainOp(model.graph.node[0]).bind(model, None)
    configured = bound.design.assign(IdentityChainDesign.grouping, grouping).root
    if not isinstance(configured, IdentityChainOp):
        raise TypeError("identity-chain configuration returned the wrong root type")
    return model, configured


def combined_construction_registry() -> ConstructionRegistry:
    entries = dict(DEFAULT_SELECTED_CONSTRUCTIONS.entries)
    schemas = dict(DEFAULT_SELECTED_CONSTRUCTIONS.choice_schemas)
    entries.update(IDENTITY_CHAIN_CONSTRUCTIONS.entries)
    schemas.update(IDENTITY_CHAIN_CONSTRUCTIONS.choice_schemas)
    return ConstructionRegistry(entries, schemas)


def combined_transform_registry() -> SelectedTransformRegistry:
    entries = dict(DEFAULT_SELECTED_TRANSFORMS.entries)
    entries.update(IDENTITY_CHAIN_TRANSFORMS.entries)
    return SelectedTransformRegistry(entries)


__all__ = [
    "IDENTITY_CHAIN_CONSTRUCTIONS",
    "IDENTITY_CHAIN_SELECTED_CONSTRUCTION",
    "IDENTITY_CHAIN_TRANSFORMS",
    "IDENTITY_ELIDED_FORM",
    "IdentityChainDesign",
    "IdentityChainOp",
    "combined_construction_registry",
    "combined_transform_registry",
    "configured_identity_chain",
    "identity_chain_model",
]
