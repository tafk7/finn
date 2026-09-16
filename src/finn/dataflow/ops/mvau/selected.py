# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Selected ONNX construction for the standard Replay-to-dot-product MVAU."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]

from finn.dataflow._engine import Finding, FindingKind, QualifiedPath
from finn.dataflow.analysis.integer_dot import (
    DotProductPremise,
    FixedWeightPremise,
    check_integer_dot_product_support,
    decode_dot_product_premise,
)
from finn.dataflow.model.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.model.maps import RectangularDomain
from finn.dataflow.model.network import DataflowNetwork
from finn.dataflow.ops.mvau.computation import AccumulationMode, ActivationMode
from finn.dataflow.ops.mvau.numerics import (
    ACTIVATION_IDENTITY,
    WEIGHT_IDENTITY,
    carrier_name,
    integer_graph_profile_fingerprint,
    integer_type,
)
from finn.dataflow.ops.mvau.designs.supply import WeightSupply
from finn.dataflow.ops.mvau.networks import (
    construct_decomposed_mvau_network,
    construct_decoupled_mvau_network,
    construct_embedded_mvau_network,
)
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
    construct_embedded_dot_product_region,
    construct_weight_stream_region,
)
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
    RequiredSupply,
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
    set_frozen_initializer,
    validate_construction_inputs,
)
from finn.dataflow.ops.selected_verification import (
    frozen_initializer_for_source,
    verify_normalized_selected_snapshot,
)
from finn.dataflow.ops.selected_transforms import (
    BOUNDED_CLEANUP_TRANSFORM,
    READABLE_NAMES_TRANSFORM,
    TRANSFORM_VERSION,
    SelectedTransformAuthorization,
)

MVAU_CONSTRUCTION_FAMILY = "finn.dataflow.selected.mvau.dot_product"
MVAU_CONSTRUCTION_VERSION = "2"
MVAU_SOURCE_SEMANTICS = "finn.dataflow.source.mvau"
MVAU_SOURCE_SEMANTICS_VERSION = 3
MVAU_GRAPH_NAME = "selected_mvau_dot_product"

ACTIVATION_KEY = SourceOperandKey("activation", SourceDirection.INPUT, 0)
WEIGHT_KEY = SourceOperandKey("weight", SourceDirection.INPUT, 1)
OUTPUT_KEY = SourceOperandKey("output", SourceDirection.OUTPUT, 0)


@dataclass(frozen=True, slots=True)
class MvauSourceSemantics:
    accumulation: AccumulationMode
    activation: ActivationMode
    accumulator_datatype: str
    output_datatype: str
    activation_bias: int | None
    integer_support_fingerprint: str | None = None
    integer_output_carrier: str | None = None
    integer_premise: object | None = None


@dataclass(frozen=True, slots=True)
class MvauSelectionParameters:
    rows: int
    matrix_width: int
    matrix_height: int
    neuron_folds: int
    pe: int
    simd: int
    weight_supply: WeightSupply
    activation_datatype: str
    weight_datatype: str
    accumulator_datatype: str
    output_datatype: str
    source_activation_shape: tuple[int, ...]
    source_weight_shape: tuple[int, ...]
    source_output_shape: tuple[int, ...]
    source_weight_initializer_digest: str | None
    integer_support_fingerprint: str | None
    integer_output_carrier: str | None
    integer_premise: DotProductPremise | None


def encode_mvau_source_semantics(value: MvauSourceSemantics) -> EncodedSourceSemantics:
    if value.activation is ActivationMode.NONE and value.activation_bias is not None:
        raise ValueError("an unfused MVAU semantic descriptor has no activation bias")
    if value.activation is ActivationMode.MULTITHRESHOLD and type(value.activation_bias) is not int:
        raise ValueError("a fused MVAU semantic descriptor needs an integer activation bias")
    if value.accumulation is AccumulationMode.INTEGER and value.activation is ActivationMode.NONE:
        if not value.integer_support_fingerprint:
            raise ValueError("plain-integer MVAU semantics need a numerical premise fingerprint")
        if value.integer_output_carrier not in {"INT32", "INT64"}:
            raise ValueError("plain-integer MVAU semantics need an INT32 or INT64 carrier")
        decode_dot_product_premise(value.integer_premise)
    elif (
        value.integer_support_fingerprint is not None
        or value.integer_output_carrier is not None
        or value.integer_premise is not None
    ):
        raise ValueError("non-plain-integer MVAU semantics must not claim integer support")
    return EncodedSourceSemantics(
        MVAU_SOURCE_SEMANTICS,
        MVAU_SOURCE_SEMANTICS_VERSION,
        {
            "accumulation": value.accumulation.value,
            "activation": value.activation.value,
            "accumulator_datatype": value.accumulator_datatype,
            "output_datatype": value.output_datatype,
            "activation_bias": value.activation_bias,
            "integer_support_fingerprint": value.integer_support_fingerprint,
            "integer_output_carrier": value.integer_output_carrier,
            "integer_premise": value.integer_premise,
        },
    )


def decode_mvau_source_semantics(value: EncodedSourceSemantics) -> MvauSourceSemantics:
    fields = {
        "accumulation",
        "activation",
        "accumulator_datatype",
        "output_datatype",
        "activation_bias",
        "integer_support_fingerprint",
        "integer_output_carrier",
        "integer_premise",
    }
    if (
        value.identity != MVAU_SOURCE_SEMANTICS
        or value.version != MVAU_SOURCE_SEMANTICS_VERSION
        or not isinstance(value.payload, Mapping)
        or set(value.payload) != fields
    ):
        raise ValueError("unsupported MVAU source-semantics payload")
    accumulation = AccumulationMode(str(value.payload["accumulation"]))
    activation = ActivationMode(str(value.payload["activation"]))
    accumulator = value.payload["accumulator_datatype"]
    output = value.payload["output_datatype"]
    bias = value.payload["activation_bias"]
    support_fingerprint = value.payload["integer_support_fingerprint"]
    integer_output_carrier = value.payload["integer_output_carrier"]
    integer_premise = value.payload["integer_premise"]
    if type(accumulator) is not str or not accumulator or type(output) is not str or not output:
        raise ValueError("MVAU semantic datatypes must be non-empty strings")
    if activation is ActivationMode.NONE:
        if bias is not None:
            raise ValueError("an unfused MVAU semantic descriptor has no activation bias")
    elif type(bias) is not int:
        raise ValueError("a fused MVAU semantic descriptor needs an integer activation bias")
    if accumulation is AccumulationMode.INTEGER and activation is ActivationMode.NONE:
        if type(support_fingerprint) is not str or not support_fingerprint:
            raise ValueError("plain-integer MVAU semantics need a numerical support fingerprint")
        if integer_output_carrier not in {"INT32", "INT64"}:
            raise ValueError("plain-integer MVAU semantics need an INT32 or INT64 carrier")
        decode_dot_product_premise(integer_premise)
    elif (
        support_fingerprint is not None
        or integer_output_carrier is not None
        or integer_premise is not None
    ):
        raise ValueError("non-plain-integer MVAU semantics must not claim integer support")
    return MvauSourceSemantics(
        accumulation,
        activation,
        accumulator,
        output,
        bias,
        support_fingerprint,
        integer_output_carrier,
        integer_premise,
    )


def _choice(choices: tuple[RecordedChoice, ...], path: str) -> object:
    matches = tuple(item.value for item in choices if item.path == path)
    if len(matches) != 1:
        raise ValueError(f"MVAU selection needs exactly one {path!r} choice")
    return matches[0]


def _operand(source: SourceProvenance, key: SourceOperandKey) -> SourceValueRef:
    matches = tuple(item for item in source.operands if item.key == key)
    if len(matches) != 1:
        raise ValueError(f"MVAU source is missing {key.operand_id!r}")
    return matches[0]


def _validated_integer_support(
    source: SourceProvenance,
    semantics: MvauSourceSemantics,
) -> DotProductPremise:
    """Re-establish detached numerical admission from the complete premise."""

    premise = decode_dot_product_premise(semantics.integer_premise)
    activation = _operand(source, ACTIVATION_KEY)
    weight = _operand(source, WEIGHT_KEY)
    output = _operand(source, OUTPUT_KEY)
    if source.scope_id is None:
        raise ValueError("selected integer MVAU requires an invocation scope")
    expected = (
        (premise.activation_source, ACTIVATION_IDENTITY, "activation source"),
        (premise.weight_source, WEIGHT_IDENTITY, "weight source"),
    )
    for actual, wanted, label in expected:
        if actual != wanted:
            raise ValueError(f"selected integer MVAU {label} identity differs")
    if premise.invocation_scope.scope_id != source.scope_id:
        raise ValueError("selected integer MVAU premise invocation scope differs")
    if premise.activation_shape != activation.shape or premise.weight_shape != weight.shape:
        raise ValueError("selected integer MVAU premise input shapes differ from source")
    if premise.output_shape != output.shape:
        raise ValueError("selected integer MVAU premise output shape differs from source")
    if premise.activation_source_carrier != carrier_name(activation.carrier_dtype):
        raise ValueError("selected integer MVAU activation carrier differs from source")
    if premise.weight_source_carrier != carrier_name(weight.carrier_dtype):
        raise ValueError("selected integer MVAU weight carrier differs from source")
    activation_type = integer_type(resolve_qonnx_datatype_name(activation.logical_datatype))
    weight_type = integer_type(resolve_qonnx_datatype_name(weight.logical_datatype))
    accumulator_type = integer_type(resolve_qonnx_datatype_name(semantics.accumulator_datatype))
    result_type = integer_type(resolve_qonnx_datatype_name(semantics.output_datatype))
    if premise.activation_logical_type != activation_type:
        raise ValueError("selected integer MVAU activation logical type differs from source")
    if premise.activation_range != activation_type.value_range:
        raise ValueError("selected integer MVAU activation range is not the full source domain")
    if premise.weight_logical_type != weight_type:
        raise ValueError("selected integer MVAU weight logical type differs from source")
    if premise.accumulator_type != accumulator_type:
        raise ValueError("selected integer MVAU accumulator type differs from source semantics")
    if premise.result_type != result_type:
        raise ValueError("selected integer MVAU result type differs from source semantics")
    if output.logical_datatype != semantics.output_datatype:
        raise ValueError("selected integer MVAU output datatype differs from source")
    if isinstance(premise.weights, FixedWeightPremise):
        if weight.initializer_content_digest != premise.weights.content_digest:
            raise ValueError("selected integer MVAU fixed-weight digest differs from source")
    report = check_integer_dot_product_support(
        premise=premise,
        selected_internal_bits=premise.accumulator_type.bit_width,
        selected_output_carrier=semantics.integer_output_carrier or "",
        target_max_accumulator_bits=64,
    )
    if not report.supported or report.support is None:
        reasons = ", ".join(item.code for item in report.findings) or "unsupported"
        raise ValueError(f"selected integer MVAU numerical support refused: {reasons}")
    expected_fingerprint = integer_graph_profile_fingerprint(report.support)
    if semantics.integer_support_fingerprint != expected_fingerprint:
        raise ValueError("selected integer MVAU support fingerprint differs from its premise")
    return premise


def derive_mvau_facts(
    identity: ConstructionIdentity,
    source: SourceProvenance,
    semantics: MvauSourceSemantics,
    choices: tuple[RecordedChoice, ...],
) -> SelectionFacts[MvauSourceSemantics, MvauSelectionParameters]:
    if identity != ConstructionIdentity(
        MVAU_CONSTRUCTION_FAMILY,
        MVAU_CONSTRUCTION_VERSION,
        "canonical",
    ):
        raise ValueError("unsupported MVAU construction identity")
    if semantics.activation is not ActivationMode.NONE:
        raise ValueError("the selected dot-product Design has no fused activation")
    if semantics.output_datatype != semantics.accumulator_datatype:
        raise ValueError("unfused MVAU output datatype must equal accumulator datatype")
    if source.family != "finn.dataflow.mvau" or source.family_version != "1":
        raise ValueError("selected MVAU requires operation family version 1")
    if source.schema_version != 5:
        raise ValueError("version-2 selected MVAU requires native schema 5")
    activation = _operand(source, ACTIVATION_KEY)
    weight = _operand(source, WEIGHT_KEY)
    output = _operand(source, OUTPUT_KEY)
    if not activation.shape or any(extent <= 0 for extent in activation.shape):
        raise ValueError("MVAU activation must have positive rank-one-or-greater shape")
    if len(weight.shape) != 2 or any(extent <= 0 for extent in weight.shape):
        raise ValueError("MVAU weight must have a positive rank-two shape")
    rows = prod(activation.shape[:-1]) if len(activation.shape) > 1 else 1
    width = activation.shape[-1]
    if weight.shape[0] != width:
        raise ValueError("MVAU activation width and weight rows differ")
    height = weight.shape[1]
    expected_output = (*activation.shape[:-1], height)
    if output.shape != expected_output:
        raise ValueError(f"MVAU source output must have shape {expected_output}")
    if output.logical_datatype != semantics.output_datatype:
        raise ValueError("MVAU source output datatype differs from normalized semantics")
    if semantics.accumulation is AccumulationMode.XNOR_POPCOUNT and (
        activation.logical_datatype != "BINARY" or weight.logical_datatype != "BINARY"
    ):
        raise ValueError("selected XNOR popcount requires BINARY activation and weight operands")
    if semantics.accumulation is AccumulationMode.BIPOLAR_POPCOUNT and (
        activation.logical_datatype != "BIPOLAR" or weight.logical_datatype != "BIPOLAR"
    ):
        raise ValueError(
            "selected bipolar popcount requires BIPOLAR activation and weight operands"
        )
    if semantics.accumulation is AccumulationMode.INTEGER and (
        activation.logical_datatype == "BIPOLAR" and weight.logical_datatype == "BIPOLAR"
    ):
        raise ValueError("BIPOLAR activation and weight operands require bipolar popcount")
    integer_premise = (
        _validated_integer_support(source, semantics)
        if semantics.accumulation is AccumulationMode.INTEGER
        else None
    )
    if semantics.accumulation is AccumulationMode.INTEGER:
        if semantics.integer_output_carrier not in {"INT32", "INT64"}:
            raise ValueError("selected integer MVAU has no checked output carrier")
        if not semantics.integer_support_fingerprint:
            raise ValueError("selected integer MVAU has no numerical support fingerprint")
    elif any(
        carrier != TensorProto.FLOAT
        for carrier in (
            activation.carrier_dtype,
            weight.carrier_dtype,
            output.carrier_dtype,
        )
    ):
        raise ValueError("selected popcount MVAU supports FLOAT ONNX carriers")
    design_case = _choice(choices, "design.case")
    if design_case != "dot_product":
        raise ValueError("selected MVAU construction requires the dot_product Design")
    pe = cast(int, _choice(choices, "design.dot_product.pe"))
    simd = cast(int, _choice(choices, "design.dot_product.simd"))
    supply_value = _choice(choices, "design.dot_product.weight_supply")
    compute = cast(str, _choice(choices, "design.dot_product.compute.kernel"))
    if not isinstance(supply_value, WeightSupply):
        raise TypeError("MVAU weight supply choice was not decoded nominally")
    supply = supply_value
    expected_compute = "dotp_axi_embedded" if supply is WeightSupply.EMBEDDED else "dotp_axi"
    if compute != expected_compute:
        raise ValueError(f"{supply.value} supply requires compute candidate {expected_compute}")
    if pe < 1 or height % pe:
        raise ValueError("MVAU PE must be a positive divisor of matrix height")
    if simd < 1 or width % simd:
        raise ValueError("MVAU SIMD must be a positive divisor of matrix width")
    if supply is not WeightSupply.EXTERNAL and not weight.initializer_present:
        raise ValueError(f"{supply.value} weight supply requires a source initializer")
    parameters = MvauSelectionParameters(
        rows,
        width,
        height,
        height // int(pe),
        int(pe),
        int(simd),
        supply,
        activation.logical_datatype,
        weight.logical_datatype,
        semantics.accumulator_datatype,
        semantics.output_datatype,
        activation.shape,
        weight.shape,
        output.shape,
        weight.initializer_content_digest,
        semantics.integer_support_fingerprint,
        semantics.integer_output_carrier,
        integer_premise,
    )
    return SelectionFacts(identity, source, semantics, choices, parameters)


def project_mvau_network(
    facts: SelectionFacts[MvauSourceSemantics, MvauSelectionParameters],
) -> DataflowNetwork:
    """Reconstruct the logical MVAU contract from detached selected facts."""

    if (
        facts.construction.form != "canonical"
        or facts.construction.form_version != 1
        or facts.construction.form_arguments
    ):
        raise ValueError("unsupported MVAU construction form")
    parameters = facts.parameters
    activation_type = resolve_qonnx_datatype_name(parameters.activation_datatype)
    weight_type = resolve_qonnx_datatype_name(parameters.weight_datatype)
    output_type = resolve_qonnx_datatype_name(parameters.output_datatype)
    replay = construct_activation_replay_region(
        parameters.rows,
        parameters.matrix_width,
        parameters.matrix_height,
        activation_type,
        parameters.pe,
        parameters.simd,
    )
    if parameters.weight_supply is WeightSupply.EMBEDDED:
        compute = construct_embedded_dot_product_region(
            parameters.rows,
            parameters.matrix_width,
            parameters.matrix_height,
            activation_type,
            weight_type,
            output_type,
            parameters.pe,
            parameters.simd,
        )
        return construct_embedded_mvau_network(replay, compute)
    compute = construct_dot_product_region(
        parameters.rows,
        parameters.matrix_width,
        parameters.matrix_height,
        activation_type,
        weight_type,
        output_type,
        parameters.pe,
        parameters.simd,
    )
    if parameters.weight_supply is WeightSupply.DECOUPLED:
        memory = construct_weight_stream_region(
            parameters.rows,
            parameters.matrix_width,
            parameters.matrix_height,
            weight_type,
            parameters.pe,
            parameters.simd,
        )
        return construct_decoupled_mvau_network(replay, compute, memory)
    return construct_decomposed_mvau_network(replay, compute)


def _tensor(name: str, carrier: int, shape: tuple[int, ...]) -> Any:
    return helper.make_tensor_value_info(name, carrier, list(shape))


def construct_mvau_snapshot(
    facts: SelectionFacts[MvauSourceSemantics, MvauSelectionParameters],
    construction_inputs: ConstructionInputs,
) -> SelectedGraphSnapshot:
    parameters = facts.parameters
    source_activation_carrier = _operand(facts.source, ACTIVATION_KEY).carrier_dtype
    source_weight_carrier = _operand(facts.source, WEIGHT_KEY).carrier_dtype
    source_output_carrier = _operand(facts.source, OUTPUT_KEY).carrier_dtype
    integer_path = facts.source_semantics.accumulation is AccumulationMode.INTEGER
    activation_carrier = TensorProto.INT64 if integer_path else source_activation_carrier
    weight_carrier = TensorProto.INT64 if integer_path else source_weight_carrier
    output_carrier = (
        TensorProto.INT32
        if integer_path and parameters.integer_output_carrier == "INT32"
        else TensorProto.INT64
        if integer_path and parameters.integer_output_carrier == "INT64"
        else source_output_carrier
    )
    required_inputs = () if parameters.weight_supply is WeightSupply.EXTERNAL else (WEIGHT_KEY,)
    frozen = validate_construction_inputs(facts, construction_inputs, required_inputs)
    activation_type = resolve_qonnx_datatype_name(parameters.activation_datatype)
    weight_type = resolve_qonnx_datatype_name(parameters.weight_datatype)
    output_type = resolve_qonnx_datatype_name(parameters.output_datatype)

    nodes = []
    node_records = []
    owner_nodes: dict[tuple[OwnerKind, str, str], list[str]] = defaultdict(list)
    value_info = []
    logical_types = {}
    constant_values: dict[str, tuple[int, ...]] = {}

    def add_value(name: str, carrier: int, shape: tuple[int, ...], logical: object) -> None:
        value_info.append(_tensor(name, carrier, shape))
        logical_types[name] = logical

    def add_node(
        node_id: str,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        owner_kind: OwnerKind,
        owner_id: str,
        rule: str,
        **attributes: object,
    ) -> None:
        nodes.append(helper.make_node(op_type, inputs, outputs, name=node_id, **attributes))
        node_records.append(GraphNodeBinding(node_id, len(nodes) - 1))
        owner_nodes[(owner_kind, owner_id, rule)].append(node_id)

    source_activation_shape = parameters.source_activation_shape
    source_activation = (
        "X" if source_activation_shape == (parameters.rows, parameters.matrix_width) else "X_source"
    )
    region_activation = source_activation
    if integer_path:
        region_activation = "X_integer"
        add_node(
            "source.activation.cast",
            "Cast",
            [source_activation],
            [region_activation],
            OwnerKind.SOURCE_BOUNDARY,
            "activation",
            "checked_integer_conversion.v1",
            to=TensorProto.INT64,
        )
        add_value(
            region_activation,
            TensorProto.INT64,
            source_activation_shape,
            activation_type,
        )
    if source_activation_shape != (parameters.rows, parameters.matrix_width):
        constant_values["shape_X"] = (parameters.rows, parameters.matrix_width)
        add_node(
            "source.activation.reshape",
            "Reshape",
            [region_activation, "shape_X"],
            ["X"],
            OwnerKind.SOURCE_BOUNDARY,
            "activation",
            "checked_integer_conversion.v1" if integer_path else "row_major.v1",
        )
        add_value(
            "X",
            activation_carrier,
            (parameters.rows, parameters.matrix_width),
            activation_type,
        )
        region_activation = "X"

    constant_values.update(
        {
            "axes_replay": (1,),
            "shape_XRF": (
                parameters.rows,
                parameters.neuron_folds,
                parameters.matrix_width,
            ),
            "shape_XR": (
                parameters.rows * parameters.neuron_folds,
                parameters.matrix_width,
            ),
        }
    )
    add_node(
        "replay.unsqueeze",
        "Unsqueeze",
        [region_activation, "axes_replay"],
        ["X1"],
        OwnerKind.REGION,
        "replay",
        "activation_replay.v1",
    )
    add_node(
        "replay.expand",
        "Expand",
        ["X1", "shape_XRF"],
        ["XRF"],
        OwnerKind.REGION,
        "replay",
        "activation_replay.v1",
    )
    add_node(
        "replay.reshape",
        "Reshape",
        ["XRF", "shape_XR"],
        ["XR"],
        OwnerKind.REGION,
        "replay",
        "activation_replay.v1",
    )
    add_value(
        "X1",
        activation_carrier,
        (parameters.rows, 1, parameters.matrix_width),
        activation_type,
    )
    add_value(
        "XRF",
        activation_carrier,
        (parameters.rows, parameters.neuron_folds, parameters.matrix_width),
        activation_type,
    )
    add_value(
        "XR",
        activation_carrier,
        (parameters.rows * parameters.neuron_folds, parameters.matrix_width),
        activation_type,
    )

    weight_owner = (
        (OwnerKind.SOURCE_BOUNDARY, "weight", "transpose_2d.v1")
        if parameters.weight_supply is WeightSupply.EXTERNAL
        else (OwnerKind.REGION, "compute", "transpose_2d.v1")
        if parameters.weight_supply is WeightSupply.EMBEDDED
        else (OwnerKind.REGION, "memory", "transpose_2d.v1")
    )
    weight_source_value = "W_source"
    if integer_path:
        weight_source_value = "W_integer"
        add_node(
            "weight.to_integer",
            "Cast",
            ["W_source"],
            [weight_source_value],
            *weight_owner,
            to=TensorProto.INT64,
        )
        add_value(
            weight_source_value,
            TensorProto.INT64,
            parameters.source_weight_shape,
            weight_type,
        )
    add_node(
        "weight.to_region",
        "Transpose",
        [weight_source_value],
        ["W"],
        *weight_owner,
        perm=[1, 0],
    )
    add_value(
        "W",
        weight_carrier,
        (parameters.matrix_height, parameters.matrix_width),
        weight_type,
    )

    constant_values["shape_X_compute"] = (
        parameters.rows,
        parameters.neuron_folds,
        1,
        parameters.matrix_width,
    )
    add_node(
        "compute.activation.reshape",
        "Reshape",
        ["XR", "shape_X_compute"],
        ["X_compute"],
        OwnerKind.REGION,
        "compute",
        "dot_product.v1",
    )
    add_value(
        "X_compute",
        activation_carrier,
        (
            parameters.rows,
            parameters.neuron_folds,
            1,
            parameters.matrix_width,
        ),
        activation_type,
    )

    if integer_path:
        constant_values["shape_W_folded"] = (
            parameters.neuron_folds,
            parameters.pe,
            parameters.matrix_width,
        )
        add_node(
            "compute.weight.reshape",
            "Reshape",
            ["W", "shape_W_folded"],
            ["W_folded"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
        )
        add_node(
            "compute.weight.transpose",
            "Transpose",
            ["W_folded"],
            ["W_matmul"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
            perm=[0, 2, 1],
        )
        add_node(
            "compute.matmul",
            "MatMul",
            ["X_compute", "W_matmul"],
            ["Y_folded_4d"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
        )
        add_value(
            "W_folded",
            weight_carrier,
            (parameters.neuron_folds, parameters.pe, parameters.matrix_width),
            weight_type,
        )
        add_value(
            "W_matmul",
            weight_carrier,
            (parameters.neuron_folds, parameters.matrix_width, parameters.pe),
            weight_type,
        )
        add_value(
            "Y_folded_4d",
            TensorProto.INT64,
            (parameters.rows, parameters.neuron_folds, 1, parameters.pe),
            output_type,
        )
        folded_result = "Y_folded_4d"
    else:
        constant_values["shape_W_folded"] = (
            1,
            parameters.neuron_folds,
            parameters.pe,
            parameters.matrix_width,
        )
        constant_values["axes_reduce"] = (-1,)
        add_node(
            "compute.weight.reshape",
            "Reshape",
            ["W", "shape_W_folded"],
            ["W_folded"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
        )
        add_node(
            "compute.equal",
            "Equal",
            ["X_compute", "W_folded"],
            ["matches"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
        )
        add_node(
            "compute.cast",
            "Cast",
            ["matches"],
            ["match_counts"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
            to=TensorProto.FLOAT,
        )
        add_node(
            "compute.reduce",
            "ReduceSum",
            ["match_counts", "axes_reduce"],
            ["Y_folded"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
            keepdims=0,
        )
        add_value(
            "W_folded",
            weight_carrier,
            (1, parameters.neuron_folds, parameters.pe, parameters.matrix_width),
            weight_type,
        )
        add_value(
            "matches",
            TensorProto.BOOL,
            (
                parameters.rows,
                parameters.neuron_folds,
                parameters.pe,
                parameters.matrix_width,
            ),
            resolve_qonnx_datatype_name("BINARY"),
        )
        add_value(
            "match_counts",
            TensorProto.FLOAT,
            (
                parameters.rows,
                parameters.neuron_folds,
                parameters.pe,
                parameters.matrix_width,
            ),
            output_type,
        )
        add_value(
            "Y_folded",
            TensorProto.FLOAT,
            (parameters.rows, parameters.neuron_folds, parameters.pe),
            output_type,
        )
        folded_result = "Y_folded"

    constant_values["shape_Y"] = (parameters.rows, parameters.matrix_height)
    reshaped_result = "Y_integer" if integer_path else "Y"
    add_node(
        "compute.output.reshape",
        "Reshape",
        [folded_result, "shape_Y"],
        [reshaped_result],
        OwnerKind.REGION,
        "compute",
        "dot_product.v1",
    )
    add_value(
        reshaped_result,
        TensorProto.INT64 if integer_path else output_carrier,
        (parameters.rows, parameters.matrix_height),
        output_type,
    )
    if integer_path:
        add_node(
            "compute.output.cast",
            "Cast" if output_carrier == TensorProto.INT32 else "Identity",
            [reshaped_result],
            ["Y"],
            OwnerKind.REGION,
            "compute",
            "dot_product.v1",
            **({"to": output_carrier} if output_carrier == TensorProto.INT32 else {}),
        )
        add_value(
            "Y",
            output_carrier,
            (parameters.rows, parameters.matrix_height),
            output_type,
        )

    graph_output = "Y"
    if parameters.source_output_shape != (parameters.rows, parameters.matrix_height):
        graph_output = "Y_source"
        constant_values["shape_Y_source"] = parameters.source_output_shape
        add_node(
            "source.output.reshape",
            "Reshape",
            ["Y", "shape_Y_source"],
            [graph_output],
            OwnerKind.SOURCE_BOUNDARY,
            "output",
            "row_major.v1",
        )

    graph_inputs = [
        _tensor(
            source_activation,
            source_activation_carrier,
            parameters.source_activation_shape,
        )
    ]
    if parameters.weight_supply is WeightSupply.EXTERNAL:
        graph_inputs.append(
            _tensor(
                "W_source",
                source_weight_carrier,
                parameters.source_weight_shape,
            )
        )
    external_names = {item.name for item in graph_inputs} | {graph_output}
    graph = helper.make_graph(
        nodes,
        MVAU_GRAPH_NAME,
        graph_inputs,
        [
            _tensor(
                graph_output,
                output_carrier,
                parameters.source_output_shape,
            )
        ],
        value_info=[item for item in value_info if item.name not in external_names],
    )
    model = ModelWrapper(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    if parameters.weight_supply is not WeightSupply.EXTERNAL:
        set_frozen_initializer(model, "W_source", frozen[WEIGHT_KEY])
    for name, value in constant_values.items():
        model.set_initializer(name, np.asarray(value, dtype=np.int64))

    logical_types[source_activation] = activation_type
    logical_types["W_source"] = weight_type
    logical_types[graph_output] = output_type
    for name, datatype in logical_types.items():
        model.set_tensor_datatype(name, datatype)

    x_domain = RectangularDomain((parameters.rows, parameters.matrix_width))
    xr_domain = RectangularDomain(
        (parameters.rows * parameters.neuron_folds, parameters.matrix_width)
    )
    w_domain = RectangularDomain((parameters.matrix_height, parameters.matrix_width))
    y_domain = RectangularDomain((parameters.rows, parameters.matrix_height))
    interface_bindings = [
        InterfaceBinding(
            QualifiedInterfaceRef("replay", InterfaceDirection.INPUT, "X", "activation_in"),
            region_activation,
            PositionRelation.direct(x_domain, x_domain),
            (GraphSlotRef(GraphSlotKind.NODE_INPUT, "replay.unsqueeze", 0),),
        ),
        InterfaceBinding(
            QualifiedInterfaceRef("replay", InterfaceDirection.OUTPUT, "XR", "activation_out"),
            "XR",
            PositionRelation.direct(xr_domain, xr_domain),
            (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "replay.reshape", 0),),
        ),
        InterfaceBinding(
            QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "XR", "activation"),
            "XR",
            PositionRelation.direct(xr_domain, xr_domain),
            (GraphSlotRef(GraphSlotKind.NODE_INPUT, "compute.activation.reshape", 0),),
        ),
        InterfaceBinding(
            QualifiedInterfaceRef("compute", InterfaceDirection.OUTPUT, "Y", "output"),
            "Y",
            PositionRelation.direct(y_domain, y_domain),
            (
                GraphSlotRef(
                    GraphSlotKind.NODE_OUTPUT,
                    "compute.output.cast" if integer_path else "compute.output.reshape",
                    0,
                ),
            ),
        ),
    ]
    weight_use = GraphSlotRef(GraphSlotKind.NODE_INPUT, "compute.weight.reshape", 0)
    if parameters.weight_supply is WeightSupply.EXTERNAL:
        interface_bindings.append(
            InterfaceBinding(
                QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "W", "weight"),
                "W",
                PositionRelation.direct(w_domain, w_domain),
                (weight_use,),
            )
        )
    elif parameters.weight_supply is WeightSupply.EMBEDDED:
        interface_bindings.append(
            InterfaceBinding(
                QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "W", None),
                "W",
                PositionRelation.direct(w_domain, w_domain),
                (),
            )
        )
    else:
        interface_bindings.extend(
            (
                InterfaceBinding(
                    QualifiedInterfaceRef("memory", InterfaceDirection.INPUT, "W", None),
                    "W",
                    PositionRelation.direct(w_domain, w_domain),
                    (),
                ),
                InterfaceBinding(
                    QualifiedInterfaceRef("memory", InterfaceDirection.OUTPUT, "W", "weight"),
                    "W",
                    PositionRelation.direct(w_domain, w_domain),
                    (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "weight.to_region", 0),),
                ),
                InterfaceBinding(
                    QualifiedInterfaceRef("compute", InterfaceDirection.INPUT, "W", "weight"),
                    "W",
                    PositionRelation.direct(w_domain, w_domain),
                    (weight_use,),
                ),
            )
        )

    source_bindings = [
        SourceValueBinding(
            ACTIVATION_KEY,
            region_activation,
            (
                PositionRelation.direct(
                    RectangularDomain(parameters.source_activation_shape), x_domain
                )
                if parameters.source_activation_shape == (parameters.rows, parameters.matrix_width)
                else PositionRelation.row_major_reshape(
                    RectangularDomain(parameters.source_activation_shape), x_domain
                )
            ),
            (
                GraphSlotRef(
                    GraphSlotKind.NODE_OUTPUT,
                    "source.activation.reshape"
                    if source_activation_shape != (parameters.rows, parameters.matrix_width)
                    else "source.activation.cast",
                    0,
                )
                if integer_path
                else GraphSlotRef(GraphSlotKind.GRAPH_INPUT, MVAU_GRAPH_NAME, 0)
                if source_activation == "X"
                else GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "source.activation.reshape", 0),
            ),
        ),
        SourceValueBinding(
            OUTPUT_KEY,
            "Y",
            (
                PositionRelation.direct(RectangularDomain(parameters.source_output_shape), y_domain)
                if parameters.source_output_shape == (parameters.rows, parameters.matrix_height)
                else PositionRelation.row_major_reshape(
                    RectangularDomain(parameters.source_output_shape), y_domain
                )
            ),
            (
                GraphSlotRef(
                    GraphSlotKind.NODE_OUTPUT,
                    "compute.output.cast" if integer_path else "compute.output.reshape",
                    0,
                ),
            ),
        ),
    ]
    if parameters.weight_supply is WeightSupply.EXTERNAL:
        source_bindings.append(
            SourceValueBinding(
                WEIGHT_KEY,
                "W",
                PositionRelation.transpose_2d(
                    RectangularDomain(parameters.source_weight_shape), w_domain
                ),
                (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "weight.to_region", 0),),
            )
        )
    else:
        source_bindings.append(
            SourceValueBinding(
                WEIGHT_KEY,
                "W",
                PositionRelation.transpose_2d(
                    RectangularDomain(parameters.source_weight_shape), w_domain
                ),
                (GraphSlotRef(GraphSlotKind.NODE_OUTPUT, "weight.to_region", 0),),
            )
        )

    supplies: tuple[RequiredSupply, ...] = ()
    if parameters.weight_supply is not WeightSupply.EXTERNAL:
        required_ref = QualifiedInterfaceRef(
            "compute" if parameters.weight_supply is WeightSupply.EMBEDDED else "memory",
            InterfaceDirection.INPUT,
            "W",
            None,
        )
        supplies = (
            RequiredSupply(
                required_ref,
                "W_source",
                WEIGHT_KEY,
                (
                    ("weight.to_integer", "weight.to_region")
                    if integer_path
                    else ("weight.to_region",)
                ),
            ),
        )

    ownership = tuple(
        ComputationOwner(kind, owner, tuple(node_ids))
        for (kind, owner, rule), node_ids in owner_nodes.items()
    )
    declaration = SelectedGraphDeclaration(
        SELECTED_DECLARATION_ID,
        SELECTED_DECLARATION_VERSION,
        "",
        facts.construction,
        facts.source,
        facts.choices,
        tuple(node_records),
        tuple(interface_bindings),
        tuple(source_bindings),
        supplies,
        ownership,
    )
    return build_selected_snapshot(model, declaration)


def verify_mvau_snapshot(
    snapshot: SelectedGraphSnapshot,
    facts: SelectionFacts[MvauSourceSemantics, MvauSelectionParameters],
) -> tuple[Finding, ...]:
    inputs = ConstructionInputs()
    if facts.parameters.weight_supply is not WeightSupply.EXTERNAL:
        frozen = frozen_initializer_for_source(snapshot, WEIGHT_KEY)
        if frozen is None:
            return (
                Finding(
                    FindingKind.REJECTION,
                    "selected-mvau-weight-root",
                    QualifiedPath("selected.mvau.weight"),
                    "selected local-weight graph has no unique W_source initializer",
                ),
            )
        inputs = ConstructionInputs(((WEIGHT_KEY, frozen),))
    try:
        expected = construct_mvau_snapshot(facts, inputs)
    except (TypeError, ValueError) as error:
        return (
            Finding(
                FindingKind.REJECTION,
                "selected-mvau-expected-construction",
                QualifiedPath("selected.mvau"),
                str(error),
            ),
        )
    return verify_normalized_selected_snapshot(
        snapshot,
        expected,
        finding_code="selected-mvau-canonical-mismatch",
        path="selected.mvau",
        message="selected MVAU graph differs from the construction derived from frozen facts",
    )


MVAU_SELECTED_CONSTRUCTION = SelectedConstruction(
    family=MVAU_CONSTRUCTION_FAMILY,
    version=MVAU_CONSTRUCTION_VERSION,
    source_semantics_identity=MVAU_SOURCE_SEMANTICS,
    source_semantics_version=MVAU_SOURCE_SEMANTICS_VERSION,
    admitted_forms=("canonical",),
    choice_paths=(
        "design.case",
        "design.dot_product.pe",
        "design.dot_product.simd",
        "design.dot_product.weight_supply",
        "design.dot_product.compute.kernel",
    ),
    initializer_inputs=(),
    decode_source_semantics=decode_mvau_source_semantics,
    derive_facts=derive_mvau_facts,
    project=project_mvau_network,
    construct=construct_mvau_snapshot,
    verify=verify_mvau_snapshot,
)

MVAU_SELECTED_TRANSFORM_AUTHORIZATIONS = (
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
    "MVAU_CONSTRUCTION_FAMILY",
    "MVAU_SELECTED_CONSTRUCTION",
    "MVAU_SELECTED_TRANSFORM_AUTHORIZATIONS",
    "MVAU_SOURCE_SEMANTICS",
    "OUTPUT_KEY",
    "WEIGHT_KEY",
    "MvauSelectionParameters",
    "MvauSourceSemantics",
    "construct_mvau_snapshot",
    "decode_mvau_source_semantics",
    "derive_mvau_facts",
    "encode_mvau_source_semantics",
    "verify_mvau_snapshot",
]
