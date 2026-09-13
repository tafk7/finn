# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Durable 4B-R1--R6 controls through the selected declaration-v2 API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
import json
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
import pytest
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.analysis.tensor_value_summary import (  # type: ignore[import-not-found]
    initializer_value_summaries,
    summarize_tensor_values,
)
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow.model import (
    AffineRankMap,
    BeatSequence,
    BoundaryContract,
    CoordinateSet,
    DataflowNetwork,
    DataflowRegion,
    ExplicitCoordinateMap,
    InputInterface,
    InternalInput,
    LogicalSchedule,
    NetworkNode,
    Operand,
    OutputInterface,
    Port,
    RectangularDomain,
    RegionEndpoint,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    SeparableAffineRequirements,
)
from finn.dataflow.ops.native import operation_choice_schema
from finn.dataflow.ops.replay.op import ActivationReplayOp
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
    DecodedSelectedGraph,
    SelectedGraphDeclaration,
    SelectedGraphError,
    SelectionFacts,
    SourceDirection,
    SourceOperandKey,
    SourceProvenance,
    SourceValueBinding,
    SourceValueRef,
    build_selected_snapshot,
    decode_selected_declaration,
    decode_selected_graph,
    encode_selected_declaration,
    set_frozen_initializer,
    validate_construction_inputs,
)
from finn.dataflow.ops.tensor_summary import FrozenInitializer


FAMILY = "test.selected.invariants"
CHOICE_FAMILY = f"{FAMILY}.choice"
SEMANTICS_IDENTITY = f"{FAMILY}.semantics"


def _tensor(name: str, shape: tuple[int, ...], carrier: int = TensorProto.FLOAT) -> Any:
    return helper.make_tensor_value_info(name, carrier, list(shape))


def _positions(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(index) for index in item) for item in np.ndindex(shape))


def _sequence(shape: tuple[int, ...]) -> BeatSequence:
    return BeatSequence(1, tuple((position,) for position in _positions(shape)))


def _requirements(shape: tuple[int, ...]) -> ScheduledInputRequirements:
    positions = _positions(shape)
    schedule = RectangularDomain((len(positions),))
    domain = RectangularDomain(shape)
    entries = tuple((((index,), position), 1) for index, position in enumerate(positions))
    return ScheduledInputRequirements(
        entries,
        schedule_domain=schedule,
        position_domain=domain,
    )


def _availability(shape: tuple[int, ...]) -> ScheduledOutputAvailability:
    positions = _positions(shape)
    return ScheduledOutputAvailability(
        {position: (index,) for index, position in enumerate(positions)},
        position_domain=RectangularDomain(shape),
        schedule_domain=RectangularDomain((len(positions),)),
    )


def _stream_network(
    shape: tuple[int, ...],
    *,
    compact: bool = False,
    beat_width: int = 1,
    zero_requirements: bool = False,
) -> DataflowNetwork:
    if compact:
        assert len(shape) == 1
        extent = shape[0]
        domain = RectangularDomain(shape)
        sequence = BeatSequence.affine(
            domain,
            elements_per_beat=beat_width,
            beat_count=extent,
            view_extents=(extent, beat_width),
            offset=0,
            coefficients=(1, 0),
        )
        requirements = ScheduledInputRequirements.affine(
            domain,
            domain,
            base=(0,),
            iteration_coefficients=((1,),),
        )
        availability = ScheduledOutputAvailability.affine(
            domain,
            domain,
            view_extents=shape,
            offset=0,
            coefficients=(1,),
        )
    else:
        sequence = _sequence(shape)
        requirements = _requirements(shape)
        availability = _availability(shape)
    if zero_requirements:
        requirements = ScheduledInputRequirements(
            (),
            schedule_domain=RectangularDomain((len(_positions(shape)),)),
            position_domain=RectangularDomain(shape),
        )
    region = DataflowRegion(
        LogicalSchedule((ScheduleLevel("i", len(_positions(shape)) if not compact else shape[0]),)),
        (
            InputInterface(
                Port("in", Operand("X", DataType["INT8"], shape), sequence),
                requirements,
            ),
        ),
        (
            OutputInterface(
                Port("out", Operand("Y", DataType["INT8"], shape), sequence),
                availability,
            ),
        ),
    )
    return DataflowNetwork(
        (NetworkNode("compute", region),),
        (),
        (
            BoundaryContract("input", RegionEndpoint("compute", "in"), sequence),
            BoundaryContract("output", RegionEndpoint("compute", "out"), sequence),
        ),
    )


def _internal_region(shape: tuple[int, ...], output_name: str) -> DataflowRegion:
    positions = _positions(shape)
    schedule_domain = RectangularDomain(())
    position_domain = RectangularDomain(shape)
    entries = tuple((((), position), 1) for position in positions)
    requirements = ScheduledInputRequirements(
        entries,
        schedule_domain=schedule_domain,
        position_domain=position_domain,
    )
    availability = ScheduledOutputAvailability(
        {position: () for position in positions},
        position_domain=position_domain,
        schedule_domain=schedule_domain,
    )
    return DataflowRegion(
        LogicalSchedule(()),
        (
            InternalInput(
                Operand("P", DataType["INT64" if shape == () else "INT8"], shape),
                requirements,
            ),
        ),
        (
            OutputInterface(
                Port(
                    "out",
                    Operand(output_name, DataType["INT64" if shape == () else "INT8"], shape),
                    _sequence(shape),
                ),
                availability,
            ),
        ),
    )


def _internal_network(shape: tuple[int, ...]) -> DataflowNetwork:
    return DataflowNetwork(
        (NetworkNode("compute", _internal_region(shape, "Y")),),
        (),
        (BoundaryContract("output", RegionEndpoint("compute", "out"), _sequence(shape)),),
    )


def _shared_internal_network(shape: tuple[int, ...]) -> DataflowNetwork:
    return DataflowNetwork(
        (
            NetworkNode("first", _internal_region(shape, "Y")),
            NetworkNode("second", _internal_region(shape, "Z")),
        ),
        (),
        (
            BoundaryContract("first_output", RegionEndpoint("first", "out"), _sequence(shape)),
            BoundaryContract("second_output", RegionEndpoint("second", "out"), _sequence(shape)),
        ),
    )


def _decode_semantics(value: EncodedSourceSemantics) -> Mapping[str, object]:
    if value.identity != SEMANTICS_IDENTITY or value.version != 1:
        raise ValueError("unsupported invariant-test source semantics")
    if not isinstance(value.payload, Mapping):
        raise ValueError("invariant-test source semantics must be an object")
    return value.payload


def _derive(
    identity: ConstructionIdentity,
    source: SourceProvenance,
    semantics: Mapping[str, object],
    choices: tuple[RecordedChoice, ...],
) -> SelectionFacts[Mapping[str, object], object]:
    return SelectionFacts(identity, source, semantics, choices, semantics["parameters"])


def _project(
    facts: SelectionFacts[Mapping[str, object], object],
) -> DataflowNetwork:
    semantics = facts.source_semantics
    case = semantics["case"]
    raw_shape = semantics["shape"]
    if not isinstance(raw_shape, tuple) or any(type(item) is not int for item in raw_shape):
        raise ValueError("invariant-test shape must be an integer tuple")
    shape = cast(tuple[int, ...], raw_shape)
    if case == "direct":
        return _stream_network(shape)
    if case == "repeated":
        return _stream_network(shape)
    if case == "compact":
        beat_width = semantics["beat_width"]
        if type(beat_width) is not int:
            raise ValueError("invariant-test beat width must be an integer")
        return _stream_network(shape, compact=True, beat_width=beat_width)
    if case == "zero":
        return _stream_network(shape, zero_requirements=True)
    if case in {"internal", "view"}:
        return _internal_network(shape)
    if case == "shared":
        return _shared_internal_network(shape)
    raise ValueError(f"unknown invariant-test case {case!r}")


CONSTRUCTION = SelectedConstruction(
    family=FAMILY,
    version="1",
    source_semantics_identity=SEMANTICS_IDENTITY,
    source_semantics_version=1,
    admitted_forms=("canonical", "bounded"),
    choice_paths=(),
    initializer_inputs=(),
    decode_source_semantics=_decode_semantics,
    derive_facts=_derive,
    project=_project,
    construct=lambda _facts, _inputs: pytest.fail("fixture construction is explicit"),
    verify=lambda _snapshot, _facts: (),
)
REGISTRY = ConstructionRegistry({(FAMILY, "1"): CONSTRUCTION})

_PE_SCHEMA = tuple(
    choice
    for choice in operation_choice_schema(ActivationReplayOp)
    if choice.choice.path == "design.pe"
)
assert len(_PE_SCHEMA) == 1
CHOICE_CONSTRUCTION = replace(
    CONSTRUCTION,
    family=CHOICE_FAMILY,
    choice_paths=("design.pe",),
)
CHOICE_REGISTRY = ConstructionRegistry(
    {(CHOICE_FAMILY, "1"): CHOICE_CONSTRUCTION},
    {(CHOICE_FAMILY, "1"): _PE_SCHEMA},
)


def _source(
    case: str,
    shape: tuple[int, ...],
    operands: tuple[SourceValueRef, ...],
    *,
    payload: dict[str, object] | None = None,
    problem: str = "problem",
    scope: str | None = "scope",
    schema: int = 1,
) -> SourceProvenance:
    semantics_payload: dict[str, object] = {
        "case": case,
        "shape": list(shape),
        "parameters": {"nested": [1, {"value": 2}]},
    }
    if payload is not None:
        semantics_payload.update(payload)
    return SourceProvenance.create(
        family=SEMANTICS_IDENTITY,
        family_version="1",
        schema_version=schema,
        problem_fingerprint=problem,
        scope_id=scope,
        operands=operands,
        semantics=EncodedSourceSemantics(SEMANTICS_IDENTITY, 1, semantics_payload),
    )


def _declaration(
    source: SourceProvenance,
    graph_nodes: tuple[str, ...],
    interface_bindings: tuple[InterfaceBinding, ...],
    source_bindings: tuple[SourceValueBinding, ...],
    ownership: tuple[ComputationOwner, ...],
    *,
    supplies: tuple[RequiredSupply, ...] = (),
    identity: ConstructionIdentity | None = None,
    choices: tuple[RecordedChoice, ...] = (),
) -> SelectedGraphDeclaration:
    return SelectedGraphDeclaration(
        SELECTED_DECLARATION_ID,
        SELECTED_DECLARATION_VERSION,
        "",
        identity or ConstructionIdentity(FAMILY, "1"),
        source,
        choices,
        tuple(GraphNodeBinding(node_id, index) for index, node_id in enumerate(graph_nodes)),
        interface_bindings,
        source_bindings,
        supplies,
        ownership,
    )


def _stream_fixture(
    *,
    case: str = "direct",
    extent: int = 2,
    beat_width: int = 1,
    repeated: bool = False,
    source: SourceProvenance | None = None,
    identity: ConstructionIdentity | None = None,
    choices: tuple[RecordedChoice, ...] = (),
) -> tuple[ModelWrapper, DataflowNetwork, SelectedGraphDeclaration]:
    shape = (extent,)
    op_type = "Add" if repeated else "Identity"
    inputs = ["X", "X"] if repeated else ["X"]
    node_id = "compute_node"
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [helper.make_node(op_type, inputs, ["Y"], name="display")],
                "selected",
                [_tensor("X", shape)],
                [_tensor("Y", shape)],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_tensor_datatype("X", DataType["INT8"])
    model.set_tensor_datatype("Y", DataType["INT8"])
    x_key = SourceOperandKey("activation", SourceDirection.INPUT, 0)
    y_key = SourceOperandKey("result", SourceDirection.OUTPUT, 0)
    if source is None:
        source = _source(
            case,
            shape,
            (
                SourceValueRef(x_key, shape, TensorProto.FLOAT, "INT8", None),
                SourceValueRef(y_key, shape, TensorProto.FLOAT, "INT8", None),
            ),
            payload={"beat_width": beat_width},
        )
    domain = RectangularDomain(shape)
    relation = PositionRelation.direct(domain, domain)
    input_anchors = (
        GraphSlotRef(GraphSlotKind.GRAPH_INPUT, "selected", 0),
        *(GraphSlotRef(GraphSlotKind.NODE_INPUT, node_id, index) for index in range(len(inputs))),
    )
    interfaces = (
        InterfaceBinding(
            QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "X", "in"),
            "X",
            relation,
            input_anchors,
        ),
        InterfaceBinding(
            QualifiedInterfaceRef("compute", InterfaceDirection.OUTPUT, "Y", "out"),
            "Y",
            relation,
            (
                GraphSlotRef(GraphSlotKind.NODE_OUTPUT, node_id, 0),
                GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),
            ),
        ),
    )
    source_bindings = (
        SourceValueBinding(
            x_key,
            "X",
            relation,
            (GraphSlotRef(GraphSlotKind.GRAPH_INPUT, "selected", 0),),
        ),
        SourceValueBinding(
            y_key,
            "Y",
            relation,
            (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
        ),
    )
    network = _project(
        _derive(
            identity or ConstructionIdentity(FAMILY, "1"),
            source,
            _decode_semantics(source.semantics),
            choices,
        )
    )
    declaration = _declaration(
        source,
        (node_id,),
        interfaces,
        source_bindings,
        (ComputationOwner(OwnerKind.REGION, "compute", (node_id,)),),
        identity=identity,
        choices=choices,
    )
    return model, network, declaration


def _internal_fixture(
    values: np.ndarray,
) -> tuple[ModelWrapper, DataflowNetwork, SelectedGraphDeclaration, FrozenInitializer]:
    shape = tuple(int(extent) for extent in values.shape)
    carrier = TensorProto.INT64 if shape == () else TensorProto.FLOAT
    logical = "INT64" if shape == () else "INT8"
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["W"], ["Y"], name="display")],
                "selected",
                [],
                [_tensor("Y", shape, carrier)],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_initializer("W", values)
    model.set_tensor_datatype("W", DataType[logical])
    model.set_tensor_datatype("Y", DataType[logical])
    tensor = next(item for item in model.graph.initializer if item.name == "W")
    frozen = FrozenInitializer.from_tensor_proto(tensor)
    w_key = SourceOperandKey("weight", SourceDirection.INPUT, 0)
    y_key = SourceOperandKey("result", SourceDirection.OUTPUT, 0)
    source = _source(
        "internal",
        shape,
        (
            SourceValueRef(w_key, shape, carrier, logical, frozen.summary.content_digest),
            SourceValueRef(y_key, shape, carrier, logical, None),
        ),
    )
    domain = RectangularDomain(shape)
    relation = PositionRelation.direct(domain, domain)
    required = QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "P", None)
    declaration = _declaration(
        source,
        ("compute_node",),
        (
            InterfaceBinding(required, "W", relation, ()),
            InterfaceBinding(
                QualifiedInterfaceRef("compute", InterfaceDirection.OUTPUT, "Y", "out"),
                "Y",
                relation,
                (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "compute_node", 0),),
            ),
        ),
        (
            SourceValueBinding(
                w_key,
                "W",
                relation,
                (GraphSlotRef(GraphSlotKind.INITIALIZER, "W", None),),
            ),
            SourceValueBinding(
                y_key,
                "Y",
                relation,
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
            ),
        ),
        (ComputationOwner(OwnerKind.REGION, "compute", ("compute_node",)),),
        supplies=(RequiredSupply(required, "W", w_key, ()),),
    )
    return model, _internal_network(shape), declaration, frozen


def _view_fixture(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    relation: PositionRelation,
    *,
    transpose_count: int = 0,
    reshape_shape: tuple[int, ...] | None = None,
) -> tuple[ModelWrapper, SelectedGraphDeclaration]:
    nodes = []
    node_ids = []
    value_info = []
    current = "W"
    current_shape = source_shape
    for index in range(transpose_count):
        node_id = f"transpose_{index}"
        output = f"T{index}"
        nodes.append(helper.make_node("Transpose", [current], [output], name=node_id, perm=[1, 0]))
        node_ids.append(node_id)
        current = output
        current_shape = tuple(reversed(current_shape))
        value_info.append(_tensor(current, current_shape))
    if reshape_shape is not None:
        nodes.append(helper.make_node("Reshape", [current, "shape"], ["V"], name="reshape"))
        node_ids.append("reshape")
        current = "V"
        current_shape = target_shape
        value_info.append(_tensor(current, current_shape))
    target_value = current
    nodes.append(helper.make_node("Identity", [target_value], ["Y"], name="display"))
    node_ids.append("compute_node")
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                nodes,
                "selected",
                [],
                [_tensor("Y", target_shape)],
                value_info=value_info,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    values = np.arange(np.prod(source_shape), dtype=np.float32).reshape(source_shape)
    model.set_initializer("W", values)
    if reshape_shape is not None:
        model.set_initializer("shape", np.array(reshape_shape, dtype=np.int64))
    for name in ("W", "Y", *(item.name for item in value_info)):
        model.set_tensor_datatype(name, DataType["INT8"])
    digest = initializer_value_summaries(model)["W"].content_digest
    w_key = SourceOperandKey("weight", SourceDirection.INPUT, 0)
    y_key = SourceOperandKey("result", SourceDirection.OUTPUT, 0)
    source = _source(
        "view",
        target_shape,
        (
            SourceValueRef(w_key, source_shape, TensorProto.FLOAT, "INT8", digest),
            SourceValueRef(y_key, target_shape, TensorProto.FLOAT, "INT8", None),
        ),
    )
    target_domain = RectangularDomain(target_shape)
    local = PositionRelation.direct(target_domain, target_domain)
    required = QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "P", None)
    source_anchor = (
        GraphSlotRef(GraphSlotKind.INITIALIZER, "W", None)
        if target_value == "W"
        else GraphSlotRef(GraphSlotKind.NODE_OUTPUT, node_ids[-2], 0)
    )
    declaration = _declaration(
        source,
        tuple(node_ids),
        (
            InterfaceBinding(required, target_value, local, ()),
            InterfaceBinding(
                QualifiedInterfaceRef("compute", InterfaceDirection.OUTPUT, "Y", "out"),
                "Y",
                local,
                (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "compute_node", 0),),
            ),
        ),
        (
            SourceValueBinding(w_key, target_value, relation, (source_anchor,)),
            SourceValueBinding(
                y_key,
                "Y",
                local,
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
            ),
        ),
        (ComputationOwner(OwnerKind.REGION, "compute", tuple(node_ids)),),
        supplies=(RequiredSupply(required, "W", w_key, tuple(node_ids[:-1])),),
    )
    return model, declaration


def _shared_root_fixture() -> tuple[ModelWrapper, DataflowNetwork, SelectedGraphDeclaration]:
    shape = (2,)
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                (
                    helper.make_node("Identity", ["W"], ["Y"], name="first"),
                    helper.make_node("Identity", ["W"], ["Z"], name="second"),
                ),
                "selected",
                [],
                [_tensor("Y", shape), _tensor("Z", shape)],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    model.set_initializer("W", np.array([3.0, -2.0], dtype=np.float32))
    for name in ("W", "Y", "Z"):
        model.set_tensor_datatype(name, DataType["INT8"])
    digest = initializer_value_summaries(model)["W"].content_digest
    w_key = SourceOperandKey("weight", SourceDirection.INPUT, 0)
    y_key = SourceOperandKey("first_result", SourceDirection.OUTPUT, 0)
    z_key = SourceOperandKey("second_result", SourceDirection.OUTPUT, 1)
    source = _source(
        "shared",
        shape,
        (
            SourceValueRef(w_key, shape, TensorProto.FLOAT, "INT8", digest),
            SourceValueRef(y_key, shape, TensorProto.FLOAT, "INT8", None),
            SourceValueRef(z_key, shape, TensorProto.FLOAT, "INT8", None),
        ),
    )
    domain = RectangularDomain(shape)
    relation = PositionRelation.direct(domain, domain)
    first_required = QualifiedInterfaceRef("first", InterfaceDirection.INPUT, "P", None)
    second_required = QualifiedInterfaceRef("second", InterfaceDirection.INPUT, "P", None)
    interfaces = (
        InterfaceBinding(first_required, "W", relation, ()),
        InterfaceBinding(
            QualifiedInterfaceRef("first", InterfaceDirection.OUTPUT, "Y", "out"),
            "Y",
            relation,
            (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "first_node", 0),),
        ),
        InterfaceBinding(second_required, "W", relation, ()),
        InterfaceBinding(
            QualifiedInterfaceRef("second", InterfaceDirection.OUTPUT, "Z", "out"),
            "Z",
            relation,
            (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "second_node", 0),),
        ),
    )
    declaration = _declaration(
        source,
        ("first_node", "second_node"),
        interfaces,
        (
            SourceValueBinding(
                w_key,
                "W",
                relation,
                (GraphSlotRef(GraphSlotKind.INITIALIZER, "W", None),),
            ),
            SourceValueBinding(
                y_key,
                "Y",
                relation,
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 0),),
            ),
            SourceValueBinding(
                z_key,
                "Z",
                relation,
                (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, "selected", 1),),
            ),
        ),
        (
            ComputationOwner(OwnerKind.REGION, "first", ("first_node",)),
            ComputationOwner(OwnerKind.REGION, "second", ("second_node",)),
        ),
        supplies=(
            RequiredSupply(first_required, "W", w_key, ()),
            RequiredSupply(second_required, "W", w_key, ()),
        ),
    )
    return model, _shared_internal_network(shape), declaration


def _decode(
    model: ModelWrapper,
    declaration: SelectedGraphDeclaration,
    *,
    registry: ConstructionRegistry = REGISTRY,
) -> DecodedSelectedGraph[Any, Any]:
    return decode_selected_graph(
        build_selected_snapshot(model, declaration), constructions=registry
    )


def test_v2_complete_ordered_slots_and_owners_refuse_omissions() -> None:
    model, network, declaration = _stream_fixture()
    assert _decode(model, declaration).network == network
    input_binding, output_binding = declaration.interface_bindings

    cases = (
        (
            replace(
                declaration,
                interface_bindings=(
                    replace(
                        input_binding,
                        anchors=tuple(
                            anchor
                            for anchor in input_binding.anchors
                            if anchor.kind is not GraphSlotKind.NODE_INPUT
                        ),
                    ),
                    output_binding,
                ),
            ),
            "selected.binding.consuming_anchor_missing",
        ),
        (
            replace(
                declaration,
                interface_bindings=(
                    input_binding,
                    replace(
                        output_binding,
                        anchors=tuple(
                            anchor
                            for anchor in output_binding.anchors
                            if anchor.kind is not GraphSlotKind.NODE_OUTPUT
                        ),
                    ),
                ),
            ),
            "selected.binding.defining_anchor_missing",
        ),
        (
            replace(
                declaration,
                source_bindings=(
                    replace(
                        declaration.source_bindings[0],
                        anchors=(GraphSlotRef(GraphSlotKind.GRAPH_INPUT, "wrong", 0),),
                    ),
                    declaration.source_bindings[1],
                ),
            ),
            "selected.binding.graph_owner",
        ),
        (
            replace(
                declaration,
                ownership=(),
            ),
            "selected.ownership.region",
        ),
        (
            replace(
                declaration,
                ownership=(*declaration.ownership, declaration.ownership[0]),
            ),
            "selected.ownership.coverage",
        ),
        (
            replace(
                declaration,
                ownership=(
                    *declaration.ownership,
                    ComputationOwner(OwnerKind.SOURCE_BOUNDARY, "absent", ()),
                ),
            ),
            "selected.ownership.source_boundary",
        ),
    )
    for corrupted, code in cases:
        with pytest.raises(SelectedGraphError) as error:
            _decode(model, corrupted)
        assert error.value.code == code


def test_v2_repeated_add_operand_slots_remain_distinct_and_complete() -> None:
    model, network, declaration = _stream_fixture(case="repeated", repeated=True)
    assert _decode(model, declaration).network == network
    input_binding = declaration.interface_bindings[0]
    assert tuple(
        anchor.index for anchor in input_binding.anchors if anchor.kind is GraphSlotKind.NODE_INPUT
    ) == (0, 1)

    omitted = replace(
        input_binding,
        anchors=tuple(
            anchor
            for anchor in input_binding.anchors
            if not (anchor.kind is GraphSlotKind.NODE_INPUT and anchor.index == 1)
        ),
    )
    with pytest.raises(SelectedGraphError) as missing:
        _decode(
            model,
            replace(
                declaration,
                interface_bindings=(omitted, declaration.interface_bindings[1]),
            ),
        )
    assert missing.value.code == "selected.binding.input_slot_coverage"

    duplicated = replace(
        input_binding,
        anchors=(*input_binding.anchors, GraphSlotRef(GraphSlotKind.NODE_INPUT, "compute_node", 0)),
    )
    with pytest.raises(SelectedGraphError) as duplicate:
        _decode(
            model,
            replace(
                declaration,
                interface_bindings=(duplicated, declaration.interface_bindings[1]),
            ),
        )
    assert duplicate.value.code == "selected.binding.anchor_duplicate"


def test_v2_initializer_content_and_ordered_derivation_remain_authoritative() -> None:
    model, network, declaration, _frozen = _internal_fixture(
        np.array([3.0, -2.0], dtype=np.float32)
    )
    snapshot = build_selected_snapshot(model, declaration)
    assert decode_selected_graph(snapshot, constructions=REGISTRY).network == network

    raw = json.loads(encode_selected_declaration(declaration))
    assert set(raw["supplies"][0]) == {
        "required_input",
        "initializer",
        "derivation_nodes",
    }
    for mutate in ("omit", "unknown"):
        corrupted = json.loads(json.dumps(raw))
        if mutate == "omit":
            del corrupted["supplies"][0]["derivation_nodes"]
        else:
            corrupted["supplies"][0]["covered_positions"] = {}
        with pytest.raises(SelectedGraphError) as malformed:
            decode_selected_declaration(json.dumps(corrupted))
        assert malformed.value.code == "selected.declaration.fields"

    changed = snapshot.model_copy()
    changed.set_initializer("W", np.array([99.0, 88.0], dtype=np.float32))
    changed.set_tensor_datatype("W", DataType["INT8"])
    with pytest.raises(SelectedGraphError) as stale:
        _decode(changed, snapshot.declaration)
    assert stale.value.code == "selected.supply.initializer_digest"

    source_domain = RectangularDomain((2, 3))
    target_domain = RectangularDomain((3, 2))
    view_model, view_declaration = _view_fixture(
        source_domain.extents,
        target_domain.extents,
        PositionRelation.transpose_2d(source_domain, target_domain),
        transpose_count=1,
    )
    disconnected = view_model.model.__class__()
    disconnected.ParseFromString(view_model.model.SerializeToString(deterministic=True))
    disconnected_model = ModelWrapper(disconnected)
    disconnected_model.set_initializer("U", np.arange(6, dtype=np.float32).reshape(2, 3))
    disconnected_model.set_tensor_datatype("U", DataType["INT8"])
    disconnected_model.graph.node[0].input[0] = "U"
    with pytest.raises(SelectedGraphError) as path:
        _decode(disconnected_model, view_declaration)
    assert path.value.code == "selected.source.initializer_derivation_disconnected"


@pytest.mark.parametrize(
    ("source_shape", "target_shape", "kind", "transpose_count", "reshape_shape", "code"),
    (
        ((2, 2), (2, 2), "direct", 0, None, None),
        ((2, 3), (3, 2), "transpose", 1, None, None),
        ((2, 3), (3, 2), "transpose", 3, None, None),
        ((2, 3), (3, 2), "reshape", 0, (3, 2), None),
        (
            (2, 2),
            (2, 2),
            "transpose",
            0,
            None,
            "selected.source.initializer_derivation_relation",
        ),
        (
            (2, 2),
            (2, 2),
            "transpose",
            2,
            None,
            "selected.source.initializer_derivation_relation",
        ),
        (
            (2, 2),
            (2, 2),
            "transpose",
            4,
            None,
            "selected.source.initializer_derivation_relation",
        ),
        (
            (2, 3),
            (3, 2),
            "reshape",
            0,
            (6,),
            "selected.source.initializer_derivation_shape",
        ),
    ),
)
def test_v2_static_initializer_views_match_declared_coordinate_semantics(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    kind: str,
    transpose_count: int,
    reshape_shape: tuple[int, ...] | None,
    code: str | None,
) -> None:
    source_domain = RectangularDomain(source_shape)
    target_domain = RectangularDomain(target_shape)
    relation = (
        PositionRelation.direct(source_domain, target_domain)
        if kind == "direct"
        else PositionRelation.row_major_reshape(source_domain, target_domain)
        if kind == "reshape"
        else PositionRelation.transpose_2d(source_domain, target_domain)
    )
    model, declaration = _view_fixture(
        source_shape,
        target_shape,
        relation,
        transpose_count=transpose_count,
        reshape_shape=reshape_shape,
    )
    if code is None:
        _decode(model, declaration)
    else:
        with pytest.raises(SelectedGraphError) as error:
            _decode(model, declaration)
        assert error.value.code == code


def test_v2_distinct_qualified_inputs_may_share_one_consistent_root() -> None:
    model, network, declaration = _shared_root_fixture()
    decoded = _decode(model, declaration)
    assert decoded.network == network
    assert {item.required_input.node_id for item in decoded.declaration.supplies} == {
        "first",
        "second",
    }
    assert {item.root_graph_value for item in decoded.declaration.supplies} == {"W"}
    assert {item.source for item in decoded.declaration.supplies} == {
        SourceOperandKey("weight", SourceDirection.INPUT, 0)
    }


@pytest.mark.parametrize(
    "entries",
    (
        [[[0], [0]], [[0], [1]]],
        [[[0], [0]]],
        [[[0], [0]], [[2], [1]]],
        [[[0], [0]], [[1], [2]]],
    ),
    ids=("duplicate", "missing", "source_out_of_bounds", "target_out_of_bounds"),
)
def test_v2_malformed_explicit_binding_maps_refuse(entries: list[object]) -> None:
    model, _network, declaration = _stream_fixture()
    raw = json.loads(encode_selected_declaration(declaration))
    raw["interface_bindings"][0]["relation"] = {
        "identity": "finn.dataflow.map",
        "version": 1,
        "kind": "explicit",
        "source_extents": [2],
        "target_extents": [2],
        "entries": entries,
    }
    with pytest.raises(SelectedGraphError) as error:
        corrupted = decode_selected_declaration(json.dumps(raw))
        _decode(model, corrupted)
    assert error.value.code in {"selected.declaration.relation", "selected.binding.relation"}


@pytest.mark.parametrize(
    "coordinate_map",
    (
        ExplicitCoordinateMap(
            (((0,), (0,)), ((1,), (0,))),
            source_domain=RectangularDomain((2,)),
            target_domain=RectangularDomain((2,)),
        ),
        ExplicitCoordinateMap(
            (((0,), (0,)), ((1,), (1,))),
            source_domain=RectangularDomain((2,)),
            target_domain=RectangularDomain((3,)),
        ),
        AffineRankMap.from_mixed_radix(
            RectangularDomain((2,)),
            view_extents=(2,),
            target=RectangularDomain((2,)),
            offset=0,
            coefficients=(0,),
        ),
    ),
    ids=("duplicate_target", "missing_target", "constant_affine"),
)
def test_position_relation_constructor_refuses_nonbijective_maps(
    coordinate_map: ExplicitCoordinateMap | AffineRankMap,
) -> None:
    with pytest.raises(ValueError, match="bijective"):
        PositionRelation(coordinate_map)


@pytest.mark.parametrize(
    "encoding",
    (
        {
            "identity": "finn.dataflow.map",
            "version": 1,
            "kind": "explicit",
            "source_extents": [2],
            "target_extents": [2],
            "entries": [[[0], [0]], [[1], [0]]],
        },
        {
            "identity": "finn.dataflow.map",
            "version": 1,
            "kind": "explicit",
            "source_extents": [2],
            "target_extents": [3],
            "entries": [[[0], [0]], [[1], [1]]],
        },
        {
            "identity": "finn.dataflow.map",
            "version": 1,
            "kind": "affine_rank",
            "source_extents": [2],
            "view_extents": [2],
            "target_extents": [2],
            "offset": 0,
            "coefficients": [0],
        },
    ),
    ids=("duplicate_target", "missing_target", "constant_affine"),
)
def test_v2_persisted_nonbijective_binding_maps_refuse(encoding: dict[str, object]) -> None:
    model, _network, declaration = _stream_fixture()
    raw = json.loads(encode_selected_declaration(declaration))
    raw["interface_bindings"][0]["relation"] = encoding
    with pytest.raises(SelectedGraphError) as error:
        corrupted = decode_selected_declaration(json.dumps(raw))
        _decode(model, corrupted)
    assert error.value.code == "selected.declaration.relation"


def test_v2_valid_permutation_and_reshape_relations_round_trip() -> None:
    domain = RectangularDomain((2,))
    permutation = PositionRelation(
        ExplicitCoordinateMap(
            (((0,), (1,)), ((1,), (0,))),
            source_domain=domain,
            target_domain=domain,
        )
    )
    model, network, declaration = _stream_fixture()
    input_binding = replace(declaration.interface_bindings[0], relation=permutation)
    decoded = _decode(
        model,
        replace(
            declaration,
            interface_bindings=(input_binding, declaration.interface_bindings[1]),
        ),
    )
    assert decoded.network == network
    assert decoded.declaration.interface_bindings[0].relation.mapped((0,)) == (1,)
    assert decoded.declaration.interface_bindings[0].relation.mapped((1,)) == (0,)

    source_domain = RectangularDomain((2, 3))
    target_domain = RectangularDomain((3, 2))
    reshape = PositionRelation.row_major_reshape(source_domain, target_domain)
    view_model, view_declaration = _view_fixture(
        source_domain.extents,
        target_domain.extents,
        reshape,
        reshape_shape=target_domain.extents,
    )
    view = _decode(view_model, view_declaration)
    assert view.declaration.source_bindings[0].relation.mapped((1, 2)) == (2, 1)


def test_position_relation_bijection_checks_stay_compact_and_preserve_empty_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("binding-relation validation entered an expansion iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    source = RectangularDomain((1_000_000, 1))
    target = RectangularDomain((1, 1_000_000))
    compact = PositionRelation.row_major_reshape(source, target)
    assert compact.mapped((999_999, 0)) == (0, 999_999)

    empty = RectangularDomain((0,))
    empty_relation = PositionRelation(
        ExplicitCoordinateMap((), source_domain=empty, target_domain=empty)
    )
    assert empty_relation.source_domain == empty_relation.target_domain == empty

    scalar = RectangularDomain(())
    scalar_relation = PositionRelation(
        ExplicitCoordinateMap((((), ()),), source_domain=scalar, target_domain=scalar)
    )
    assert scalar_relation.mapped(()) == ()


def test_v2_million_identity_projection_stays_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    extent = 1_000_000
    model, network, declaration = _stream_fixture(case="compact", extent=extent, beat_width=1)

    def refuse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("compact selected projection entered an expansion iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)
    snapshot = build_selected_snapshot(model, declaration)
    assert len(snapshot.model_bytes) < 10_000
    decoded = decode_selected_graph(snapshot, constructions=REGISTRY)
    region = decoded.network.nodes[0].region
    input_interface = region.inputs[0]
    assert isinstance(input_interface, InputInterface)
    assert decoded.network == network
    assert input_interface.port.beat_sequence.position_at(extent - 1, 0) == (extent - 1,)
    assert input_interface.requirements.required((extent - 1,), (extent - 1,)) == 1


def test_v2_projector_refuses_nonpositive_beat_width_before_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _network, declaration = _stream_fixture(case="compact", extent=1_000_000, beat_width=0)

    def refuse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid beat width entered an expansion iterator")

    monkeypatch.setattr(RectangularDomain, "iter_coordinates", refuse)
    monkeypatch.setattr(CoordinateSet, "iter_coordinates", refuse)
    monkeypatch.setattr(BeatSequence, "iter_beats", refuse)
    with pytest.raises(SelectedGraphError) as error:
        _decode(model, declaration)
    assert error.value.code.startswith("selected.projection.network.")
    assert error.value.code.endswith("beat.elements_per_beat_not_positive")


def test_v2_projected_zero_requirements_keep_one_typed_value_semantics() -> None:
    model, _network, declaration = _stream_fixture(case="zero")
    projected = _decode(model, declaration).network.nodes[0].region.inputs[0].requirements
    domain = RectangularDomain((2,))
    raw = SeparableAffineRequirements(
        domain,
        domain,
        base=(0,),
        iteration_coefficients=((1,),),
        multiplicity=0,
    )
    values = (
        projected,
        ScheduledInputRequirements.from_rule(raw),
        ScheduledInputRequirements.affine(
            domain,
            domain,
            base=(0,),
            iteration_coefficients=((1,),),
            multiplicity=0,
        ),
        ScheduledInputRequirements((), schedule_domain=domain, position_domain=domain),
    )
    assert all(value == projected for value in values)
    assert len({hash(value) for value in values}) == 1


def test_v2_scalar_construction_input_preserves_rank_carrier_bytes_and_digest() -> None:
    values = np.array(7, dtype=np.int64)
    model, network, declaration, frozen = _internal_fixture(values)
    decoded = _decode(model, declaration)
    key = SourceOperandKey("weight", SourceDirection.INPUT, 0)
    validated = validate_construction_inputs(
        decoded.selection_facts,
        ConstructionInputs(((key, frozen),)),
        (key,),
    )
    restored = validated[key]
    assert decoded.network == network
    assert restored.shape == ()
    assert restored.carrier_dtype == TensorProto.INT64
    assert restored.numpy_dtype == np.dtype(np.int64).str
    assert restored.contiguous_bytes == values.tobytes()
    assert restored.summary == summarize_tensor_values(values)
    assert (
        restored.summary.content_digest == declaration.source.operands[0].initializer_content_digest
    )

    copied = ModelWrapper(helper.make_model(helper.make_graph([], "copy", [], [])))
    set_frozen_initializer(copied, "scalar", restored)
    copied_value = copied.get_initializer("scalar")
    assert copied_value is not None
    assert copied_value.shape == ()
    assert copied_value.dtype == values.dtype
    assert copied_value.tobytes() == values.tobytes()

    vector = np.array([7], dtype=np.int64)
    vector_tensor = helper.make_tensor(
        "vector",
        TensorProto.INT64,
        [1],
        vector.tolist(),
    )
    wrong = FrozenInitializer.from_tensor_proto(vector_tensor)
    assert wrong.summary.content_digest == frozen.summary.content_digest
    with pytest.raises(SelectedGraphError) as construction_error:
        validate_construction_inputs(
            decoded.selection_facts,
            ConstructionInputs(((key, wrong),)),
            (key,),
        )
    assert construction_error.value.code == "selected.construction.initializer_shape"

    changed = decoded.snapshot.model_copy()
    changed.set_initializer("W", vector)
    changed.set_tensor_datatype("W", DataType["INT64"])
    with pytest.raises(SelectedGraphError) as selected_error:
        _decode(changed, decoded.declaration)
    assert selected_error.value.code == "selected.source.shape"


def test_v2_decoded_source_choices_parameters_forms_and_origin_are_deeply_immutable() -> None:
    payload_shape = [2]
    payload_nested: list[object] = [1, {"value": 2}]
    payload_parameters: dict[str, object] = {"nested": payload_nested}
    payload: dict[str, object] = {
        "case": "direct",
        "shape": payload_shape,
        "parameters": payload_parameters,
    }
    form_argument = {"candidate": ["left"]}
    identity = ConstructionIdentity(
        CHOICE_FAMILY,
        "1",
        "bounded",
        1,
        (("selection", form_argument),),
    )
    x_key = SourceOperandKey("activation", SourceDirection.INPUT, 0)
    y_key = SourceOperandKey("result", SourceDirection.OUTPUT, 0)
    source = SourceProvenance.create(
        family=SEMANTICS_IDENTITY,
        family_version="1",
        schema_version=7,
        problem_fingerprint="origin-problem",
        scope_id="origin-scope",
        operands=(
            SourceValueRef(x_key, (2,), TensorProto.FLOAT, "INT8", None),
            SourceValueRef(y_key, (2,), TensorProto.FLOAT, "INT8", None),
        ),
        semantics=EncodedSourceSemantics(SEMANTICS_IDENTITY, 1, payload),
    )
    original_source_fingerprint = source.semantic_fingerprint
    original_selection_fingerprint = SelectionFacts(
        identity,
        source,
        source.semantics.payload,
        (RecordedChoice("design.pe", 2),),
        cast(Mapping[str, object], source.semantics.payload)["parameters"],
    ).selection_fingerprint
    payload_shape.append(99)
    payload_nested.append(3)
    form_argument["candidate"].append("right")

    model, _network, declaration = _stream_fixture(
        source=source,
        identity=identity,
        choices=(RecordedChoice("design.pe", 2),),
    )
    decoded = _decode(model, declaration, registry=CHOICE_REGISTRY)
    semantics = decoded.selection_facts.source_semantics
    parameters = decoded.selection_facts.parameters
    stored_form = decoded.declaration.construction.form_arguments[0][1]
    assert isinstance(semantics, Mapping)
    assert isinstance(parameters, Mapping)
    assert isinstance(stored_form, Mapping)
    nested = parameters["nested"]
    assert isinstance(nested, tuple)
    nested_item = nested[1]
    assert isinstance(nested_item, Mapping)
    assert decoded.selection_facts.source is decoded.declaration.source
    assert semantics is decoded.declaration.source.semantics.payload
    assert semantics["shape"] == (2,)
    assert semantics["parameters"] is parameters
    assert nested == (1, {"value": 2})
    assert stored_form["candidate"] == ("left",)
    assert decoded.selection_facts.choices[0].value == 2
    assert decoded.declaration.source.problem_fingerprint == "origin-problem"
    assert decoded.declaration.source.scope_id == "origin-scope"
    assert decoded.declaration.source.schema_version == 7
    assert decoded.declaration.source.semantic_fingerprint == original_source_fingerprint
    assert decoded.selection_facts.selection_fingerprint == original_selection_fingerprint

    with pytest.raises(TypeError):
        semantics["case"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        nested_item["value"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        stored_form["candidate"] = ()  # type: ignore[index]
    with pytest.raises((FrozenInstanceError, TypeError)):
        setattr(decoded.selection_facts.choices[0], "value", 3)
    with pytest.raises((FrozenInstanceError, TypeError)):
        setattr(decoded.declaration.source, "scope_id", "changed")
