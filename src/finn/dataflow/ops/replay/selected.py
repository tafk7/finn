# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Selected ONNX construction for standalone activation Replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
from onnx import helper  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Finding, FindingKind, QualifiedPath
from finn.dataflow.model.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.model.maps import RectangularDomain
from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    NetworkNode,
    RegionEndpoint,
)
from finn.dataflow.ops.mvau.regions import construct_activation_replay_region
from finn.dataflow.ops.selected import (
    SELECTED_DECLARATION_ID,
    SELECTED_DECLARATION_VERSION,
    ComputationOwner,
    ConstructionIdentity,
    ConstructionInputs,
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
    SelectedConstruction,
    SelectedGraphDeclaration,
    SelectedGraphSnapshot,
    SelectionFacts,
    SourceDirection,
    SourceOperandKey,
    SourceProvenance,
    SourceValueBinding,
    SourceValueRef,
    build_selected_snapshot,
    validate_construction_inputs,
)
from finn.dataflow.ops.selected_verification import verify_normalized_selected_snapshot
from finn.dataflow.ops.selected_transforms import (
    BOUNDED_CLEANUP_TRANSFORM,
    READABLE_NAMES_TRANSFORM,
    TRANSFORM_VERSION,
    SelectedTransformAuthorization,
)

REPLAY_CONSTRUCTION_FAMILY = "finn.dataflow.selected.activation_replay"
REPLAY_CONSTRUCTION_VERSION = "1"
REPLAY_SOURCE_SEMANTICS = "finn.dataflow.source.activation_replay"
REPLAY_SOURCE_SEMANTICS_VERSION = 1
REPLAY_GRAPH_NAME = "selected_activation_replay"

ACTIVATION_KEY = SourceOperandKey("activation", SourceDirection.INPUT, 0)
EXPANDED_KEY = SourceOperandKey("expanded", SourceDirection.OUTPUT, 0)


@dataclass(frozen=True, slots=True)
class ReplaySourceSemantics:
    repeat_count: int


@dataclass(frozen=True, slots=True)
class ReplaySelectionParameters:
    rows: int
    matrix_width: int
    replay_count: int
    simd: int
    activation_datatype: str
    source_input_shape: tuple[int, ...]
    source_output_shape: tuple[int, ...]


def encode_replay_source_semantics(
    value: ReplaySourceSemantics,
) -> EncodedSourceSemantics:
    if type(value.repeat_count) is not int or value.repeat_count < 1:
        raise ValueError("Replay repeat_count must be a positive integer")
    return EncodedSourceSemantics(
        REPLAY_SOURCE_SEMANTICS,
        REPLAY_SOURCE_SEMANTICS_VERSION,
        {"repeat_count": value.repeat_count},
    )


def decode_replay_source_semantics(
    value: EncodedSourceSemantics,
) -> ReplaySourceSemantics:
    if (
        value.identity != REPLAY_SOURCE_SEMANTICS
        or value.version != REPLAY_SOURCE_SEMANTICS_VERSION
        or not isinstance(value.payload, Mapping)
        or set(value.payload) != {"repeat_count"}
    ):
        raise ValueError("unsupported Replay source-semantics payload")
    count = value.payload["repeat_count"]
    if type(count) is not int or count < 1:
        raise ValueError("Replay repeat_count must be a positive integer")
    return ReplaySourceSemantics(count)


def _choice(choices: tuple[RecordedChoice, ...], path: str) -> object:
    matches = tuple(item.value for item in choices if item.path == path)
    if len(matches) != 1:
        raise ValueError(f"Replay selection needs exactly one {path!r} choice")
    return matches[0]


def _operand(source: SourceProvenance, key: SourceOperandKey) -> SourceValueRef:
    matches = tuple(item for item in source.operands if item.key == key)
    if len(matches) != 1:
        raise ValueError(f"Replay source is missing {key.operand_id!r}")
    return matches[0]


def derive_replay_facts(
    identity: ConstructionIdentity,
    source: SourceProvenance,
    semantics: ReplaySourceSemantics,
    choices: tuple[RecordedChoice, ...],
) -> SelectionFacts[ReplaySourceSemantics, ReplaySelectionParameters]:
    if identity != ConstructionIdentity(
        REPLAY_CONSTRUCTION_FAMILY,
        REPLAY_CONSTRUCTION_VERSION,
        "canonical",
    ):
        raise ValueError("unsupported Replay construction identity")
    if source.family != "finn.dataflow.activation_replay" or source.family_version != "2":
        raise ValueError("selected Replay requires operation family version 2")
    activation = _operand(source, ACTIVATION_KEY)
    expanded = _operand(source, EXPANDED_KEY)
    if len(activation.shape) < 2 or any(extent <= 0 for extent in activation.shape):
        raise ValueError("Replay activation must have positive rank-two-or-greater shape")
    rows = prod(activation.shape[:-1])
    width = activation.shape[-1]
    expected_output = (rows * semantics.repeat_count, width)
    if expanded.shape != expected_output:
        raise ValueError(f"Replay source output must have shape {expected_output}")
    if (
        activation.logical_datatype != expanded.logical_datatype
        or activation.carrier_dtype != expanded.carrier_dtype
    ):
        raise ValueError("Replay input and output datatypes must agree")
    pe = cast(int, _choice(choices, "design.pe"))
    simd = cast(int, _choice(choices, "design.simd"))
    if pe != 1:
        raise ValueError("standalone Replay requires PE=1")
    if simd < 1 or width % simd:
        raise ValueError("Replay SIMD must be a positive divisor of matrix width")
    parameters = ReplaySelectionParameters(
        rows,
        width,
        semantics.repeat_count,
        int(simd),
        activation.logical_datatype,
        activation.shape,
        expanded.shape,
    )
    return SelectionFacts(identity, source, semantics, choices, parameters)


def _tensor(name: str, carrier: int, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, carrier, list(shape))


def project_replay_network(
    facts: SelectionFacts[ReplaySourceSemantics, ReplaySelectionParameters],
) -> DataflowNetwork:
    """Reconstruct the logical Replay contract from detached selected facts."""

    if (
        facts.construction.form != "canonical"
        or facts.construction.form_version != 1
        or facts.construction.form_arguments
    ):
        raise ValueError("unsupported Replay construction form")
    parameters = facts.parameters
    datatype = resolve_qonnx_datatype_name(parameters.activation_datatype)
    replay = construct_activation_replay_region(
        parameters.rows,
        parameters.matrix_width,
        parameters.replay_count,
        datatype,
        1,
        parameters.simd,
    )
    return DataflowNetwork(
        (NetworkNode("replay", replay),),
        (),
        (
            BoundaryContract(
                "activation",
                RegionEndpoint("replay", "activation_in"),
                replay.input_interface("activation_in").port.beat_sequence,
            ),
            BoundaryContract(
                "expanded",
                RegionEndpoint("replay", "activation_out"),
                replay.output_interface("activation_out").port.beat_sequence,
            ),
        ),
    )


def construct_replay_snapshot(
    facts: SelectionFacts[ReplaySourceSemantics, ReplaySelectionParameters],
    construction_inputs: ConstructionInputs,
) -> SelectedGraphSnapshot:
    validate_construction_inputs(facts, construction_inputs, ())
    parameters = facts.parameters
    carrier_dtype = _operand(facts.source, ACTIVATION_KEY).carrier_dtype
    datatype = resolve_qonnx_datatype_name(parameters.activation_datatype)
    nodes = []
    node_records = []
    ownership = []
    value_info = []

    current = "X"
    graph_input = "X"
    if parameters.source_input_shape != (parameters.rows, parameters.matrix_width):
        graph_input = "X_source"
        current = "X"
        node_id = "source.activation.reshape"
        nodes.append(helper.make_node("Reshape", [graph_input, "shape_X"], [current], name=node_id))
        node_records.append(GraphNodeBinding(node_id, len(nodes) - 1))
        ownership.append(ComputationOwner(OwnerKind.SOURCE_BOUNDARY, "activation", (node_id,)))
        value_info.append(
            _tensor(
                current,
                carrier_dtype,
                (parameters.rows, parameters.matrix_width),
            )
        )

    replay_nodes = []
    for node_id, op_type, inputs, outputs in (
        ("replay.unsqueeze", "Unsqueeze", [current, "axes_replay"], ["X1"]),
        ("replay.expand", "Expand", ["X1", "shape_XRF"], ["XRF"]),
        ("replay.reshape", "Reshape", ["XRF", "shape_XR"], ["XR"]),
    ):
        nodes.append(helper.make_node(op_type, inputs, outputs, name=node_id))
        node_records.append(GraphNodeBinding(node_id, len(nodes) - 1))
        replay_nodes.append(node_id)
    ownership.append(ComputationOwner(OwnerKind.REGION, "replay", tuple(replay_nodes)))

    value_info.extend(
        (
            _tensor(
                "X1",
                carrier_dtype,
                (parameters.rows, 1, parameters.matrix_width),
            ),
            _tensor(
                "XRF",
                carrier_dtype,
                (parameters.rows, parameters.replay_count, parameters.matrix_width),
            ),
        )
    )
    model = ModelWrapper(
        helper.make_model(
            helper.make_graph(
                nodes,
                REPLAY_GRAPH_NAME,
                [
                    _tensor(
                        graph_input,
                        carrier_dtype,
                        parameters.source_input_shape,
                    )
                ],
                [_tensor("XR", carrier_dtype, parameters.source_output_shape)],
                value_info=value_info,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
    )
    constant_values = {
        "axes_replay": (1,),
        "shape_XRF": (
            parameters.rows,
            parameters.replay_count,
            parameters.matrix_width,
        ),
        "shape_XR": parameters.source_output_shape,
    }
    if graph_input != "X":
        constant_values["shape_X"] = (parameters.rows, parameters.matrix_width)
    for name, value in constant_values.items():
        model.set_initializer(name, np.asarray(value, dtype=np.int64))
    for name in (graph_input, "X", "X1", "XRF", "XR"):
        if name == graph_input or model.get_tensor_shape(name) is not None:
            model.set_tensor_datatype(name, datatype)

    x_domain = RectangularDomain((parameters.rows, parameters.matrix_width))
    xr_domain = RectangularDomain(parameters.source_output_shape)
    interface_bindings = (
        InterfaceBinding(
            QualifiedInterfaceRef("replay", InterfaceDirection.INPUT, "X", "activation_in"),
            "X",
            PositionRelation.direct(x_domain, x_domain),
            (GraphSlotRef(GraphSlotKind.NODE_INPUT, "replay.unsqueeze", 0),),
        ),
        InterfaceBinding(
            QualifiedInterfaceRef("replay", InterfaceDirection.OUTPUT, "XR", "activation_out"),
            "XR",
            PositionRelation.direct(xr_domain, xr_domain),
            (
                GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "replay.reshape", 0),
                GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, REPLAY_GRAPH_NAME, 0),
            ),
        ),
    )
    source_bindings = (
        SourceValueBinding(
            ACTIVATION_KEY,
            "X",
            (
                PositionRelation.direct(RectangularDomain(parameters.source_input_shape), x_domain)
                if parameters.source_input_shape == (parameters.rows, parameters.matrix_width)
                else PositionRelation.row_major_reshape(
                    RectangularDomain(parameters.source_input_shape), x_domain
                )
            ),
            (
                GraphSlotRef(GraphSlotKind.GRAPH_INPUT, REPLAY_GRAPH_NAME, 0)
                if graph_input == "X"
                else GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "source.activation.reshape", 0),
            ),
        ),
        SourceValueBinding(
            EXPANDED_KEY,
            "XR",
            PositionRelation.direct(xr_domain, xr_domain),
            (GraphSlotRef(GraphSlotKind.GRAPH_OUTPUT, REPLAY_GRAPH_NAME, 0),),
        ),
    )
    declaration = SelectedGraphDeclaration(
        SELECTED_DECLARATION_ID,
        SELECTED_DECLARATION_VERSION,
        "",
        facts.construction,
        facts.source,
        facts.choices,
        tuple(node_records),
        interface_bindings,
        source_bindings,
        (),
        tuple(ownership),
    )
    return build_selected_snapshot(model, declaration)


def verify_replay_snapshot(
    snapshot: SelectedGraphSnapshot,
    facts: SelectionFacts[ReplaySourceSemantics, ReplaySelectionParameters],
) -> tuple[Finding, ...]:
    try:
        expected = construct_replay_snapshot(facts, ConstructionInputs())
    except (TypeError, ValueError) as error:
        return (
            Finding(
                FindingKind.REJECTION,
                "selected-replay-expected-construction",
                QualifiedPath("selected.replay"),
                str(error),
            ),
        )
    return verify_normalized_selected_snapshot(
        snapshot,
        expected,
        finding_code="selected-replay-canonical-mismatch",
        path="selected.replay",
        message="selected Replay graph differs from the construction derived from frozen facts",
    )


REPLAY_SELECTED_CONSTRUCTION = SelectedConstruction(
    family=REPLAY_CONSTRUCTION_FAMILY,
    version=REPLAY_CONSTRUCTION_VERSION,
    source_semantics_identity=REPLAY_SOURCE_SEMANTICS,
    source_semantics_version=REPLAY_SOURCE_SEMANTICS_VERSION,
    admitted_forms=("canonical",),
    choice_paths=("design.pe", "design.simd"),
    initializer_inputs=(),
    decode_source_semantics=decode_replay_source_semantics,
    derive_facts=derive_replay_facts,
    project=project_replay_network,
    construct=construct_replay_snapshot,
    verify=verify_replay_snapshot,
)

REPLAY_SELECTED_TRANSFORM_AUTHORIZATIONS = (
    SelectedTransformAuthorization(
        READABLE_NAMES_TRANSFORM,
        TRANSFORM_VERSION,
        ("canonical",),
        None,
        ("replace_nodes", "rename_values"),
    ),
    SelectedTransformAuthorization(
        BOUNDED_CLEANUP_TRANSFORM,
        TRANSFORM_VERSION,
        ("canonical",),
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
)

__all__ = [
    "ACTIVATION_KEY",
    "EXPANDED_KEY",
    "REPLAY_CONSTRUCTION_FAMILY",
    "REPLAY_SELECTED_CONSTRUCTION",
    "REPLAY_SELECTED_TRANSFORM_AUTHORIZATIONS",
    "REPLAY_SOURCE_SEMANTICS",
    "ReplaySelectionParameters",
    "ReplaySourceSemantics",
    "construct_replay_snapshot",
    "decode_replay_source_semantics",
    "derive_replay_facts",
    "encode_replay_source_semantics",
    "verify_replay_snapshot",
]
