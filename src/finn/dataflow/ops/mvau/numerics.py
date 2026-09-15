# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Thin MVAU adapters for the hierarchy-independent integer-dot proof."""

from __future__ import annotations

from math import prod
import hashlib
import json
from typing import Any

from onnx import TensorProto  # type: ignore[import-not-found]

from finn.dataflow.analysis.integer_dot import (
    DotProductPremise,
    FixedWeightPremise,
    IntegerRange,
    IntegerSupportReport,
    IntegerSupport,
    IntegerType,
    InvocationScope,
    OperandIdentity,
    RuntimeIntegerTensor,
    RuntimeWeightPromise,
    check_integer_dot_product_support,
    execute_integer_dot_product,
    validate_integer_dot_product_operands,
)
from finn.dataflow.model.datatypes import QONNXDataType
from finn.dataflow.ops.source import SourceNode, SourceOperand
from finn.dataflow.ops.tensor_summary import FrozenInitializer


ACTIVATION_IDENTITY = OperandIdentity("activation", "input", 0)
WEIGHT_IDENTITY = OperandIdentity("weight", "input", 1)


_CARRIER_NAMES = {
    TensorProto.BOOL: "BOOL",
    TensorProto.INT8: "INT8",
    TensorProto.UINT8: "UINT8",
    TensorProto.INT16: "INT16",
    TensorProto.UINT16: "UINT16",
    TensorProto.INT32: "INT32",
    TensorProto.UINT32: "UINT32",
    TensorProto.INT64: "INT64",
    TensorProto.UINT64: "UINT64",
    TensorProto.FLOAT16: "FLOAT16",
    TensorProto.FLOAT: "FLOAT32",
    TensorProto.DOUBLE: "FLOAT64",
}


def carrier_name(carrier_dtype: int) -> str:
    try:
        return _CARRIER_NAMES[carrier_dtype]
    except KeyError as error:
        raise ValueError(f"unsupported ONNX tensor carrier {carrier_dtype}") from error


def array_carrier_name(value: Any) -> str:
    import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415

    dtype = np.asarray(value).dtype
    names = {
        np.dtype(np.bool_): "BOOL",
        np.dtype(np.int8): "INT8",
        np.dtype(np.uint8): "UINT8",
        np.dtype(np.int16): "INT16",
        np.dtype(np.uint16): "UINT16",
        np.dtype(np.int32): "INT32",
        np.dtype(np.uint32): "UINT32",
        np.dtype(np.int64): "INT64",
        np.dtype(np.uint64): "UINT64",
        np.dtype(np.float16): "FLOAT16",
        np.dtype(np.float32): "FLOAT32",
        np.dtype(np.float64): "FLOAT64",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported runtime tensor carrier {dtype}") from error


def integer_type(datatype: QONNXDataType) -> IntegerType:
    if not datatype.is_integer():
        raise ValueError(f"{datatype.name} is not an integer datatype")
    minimum, maximum = datatype.min(), datatype.max()
    if type(minimum) is not int or type(maximum) is not int:
        raise ValueError(f"{datatype.name} does not expose exact integer endpoints")
    return IntegerType(
        datatype.name,
        IntegerRange(minimum, maximum),
        datatype.bitwidth(),
        bool(datatype.signed()),
    )


def target_accumulator_bits(target: object) -> int:
    name = getattr(target, "value", str(target))
    try:
        return {"DSP48E1": 48, "DSP48E2": 48, "DSP58": 58}[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported target DSP {name!r}") from error


def integer_graph_profile_fingerprint(support: IntegerSupport) -> str:
    """Identity of selected arithmetic bytes, excluding per-use applicability."""

    bounds = support.bounds
    value = {
        "product": [bounds.product.minimum, bounds.product.maximum],
        "intermediate": [
            bounds.every_intermediate.minimum,
            bounds.every_intermediate.maximum,
        ],
        "result": [bounds.result.minimum, bounds.result.maximum],
        "internal_bits": support.internal_bits,
        "operand_carrier": support.operand_carrier,
        "matmul_result_carrier": support.matmul_result_carrier,
        "output_carrier": support.output_carrier,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fixed_weights(initializer: FrozenInitializer, carrier: str) -> FixedWeightPremise:
    values = []
    for raw in initializer.array_copy().reshape(-1):
        item = raw.item() if hasattr(raw, "item") else raw
        if isinstance(item, bool):
            raise ValueError("fixed weights contain Boolean values")
        if isinstance(item, int):
            values.append(item)
        elif isinstance(item, float) and item.is_integer():
            values.append(int(item))
        else:
            raise ValueError("fixed weights contain non-integral values")
    return FixedWeightPremise(tuple(values), initializer.summary.content_digest, carrier)


def mvau_integer_premise_from_operands(
    activation: SourceOperand,
    weight: SourceOperand,
    *,
    accumulator_datatype: QONNXDataType,
    output_datatype: QONNXDataType,
    invocation_scope: InvocationScope,
    runtime_writable: bool,
    runtime_promise: RuntimeWeightPromise | None,
) -> DotProductPremise:
    activation_type = integer_type(activation.datatype)
    weight_type = integer_type(weight.datatype)
    accumulator_type = integer_type(accumulator_datatype)
    output_type = integer_type(output_datatype)
    activation_carrier = carrier_name(activation.carrier_dtype)
    weight_carrier = carrier_name(weight.carrier_dtype)
    if runtime_writable:
        if runtime_promise is None:
            raise ValueError("runtime-writable weights require an explicit range promise")
        weights: FixedWeightPremise | RuntimeWeightPromise = runtime_promise
    else:
        if weight.initializer_value is None:
            raise ValueError("fixed-weight integer support requires an immutable initializer")
        weights = _fixed_weights(weight.initializer_value, weight_carrier)
    if len(weight.shape) != 2 or not activation.shape:
        raise ValueError("integer MVAU requires rank-two weights and a ranked activation")
    output_shape = (*activation.shape[:-1], weight.shape[1])
    return DotProductPremise(
        activation_type.value_range,
        activation_type,
        activation_carrier,
        activation.shape,
        ACTIVATION_IDENTITY,
        weights,
        weight_type,
        weight_carrier,
        weight.shape,
        WEIGHT_IDENTITY,
        accumulator_type,
        output_type,
        output_shape,
        invocation_scope,
    )


def mvau_integer_premise(
    source: SourceNode,
    *,
    invocation_scope: InvocationScope,
    runtime_writable: bool,
    runtime_promise: RuntimeWeightPromise | None,
) -> DotProductPremise:
    return mvau_integer_premise_from_operands(
        source.operand("activation"),
        source.operand("weight"),
        accumulator_datatype=source.attributes["accumulator_type"],  # type: ignore[arg-type]
        output_datatype=source.attributes["output_type"],  # type: ignore[arg-type]
        invocation_scope=invocation_scope,
        runtime_writable=runtime_writable,
        runtime_promise=runtime_promise,
    )


def check_mvau_integer_support(
    source: SourceNode,
    *,
    invocation_scope: InvocationScope,
    runtime_writable: bool,
    runtime_promise: RuntimeWeightPromise | None,
    target: object,
) -> IntegerSupportReport:
    from finn.dataflow.analysis.integer_dot import NumericalFinding  # noqa: PLC0415

    try:
        premise = mvau_integer_premise(
            source,
            invocation_scope=invocation_scope,
            runtime_writable=runtime_writable,
            runtime_promise=runtime_promise,
        )
        from finn.dataflow.analysis.integer_dot import analyze_integer_dot_product  # noqa: PLC0415

        bounds = analyze_integer_dot_product(premise)
        output_carrier = (
            "INT32"
            if IntegerRange(-(1 << 31), (1 << 31) - 1).contains_range(bounds.result)
            else "INT64"
        )
        return check_integer_dot_product_support(
            premise=premise,
            selected_internal_bits=premise.accumulator_type.bit_width,
            selected_output_carrier=output_carrier,
            target_max_accumulator_bits=target_accumulator_bits(target),
        )
    except (TypeError, ValueError) as error:
        return IntegerSupportReport(
            None,
            (NumericalFinding("integer-premise-invalid", str(error)),),
        )


def check_mvau_integer_support_from_operands(
    activation: SourceOperand,
    weight: SourceOperand,
    *,
    accumulator_datatype: QONNXDataType,
    output_datatype: QONNXDataType,
    invocation_scope: InvocationScope,
    runtime_writable: bool,
    runtime_promise: RuntimeWeightPromise | None,
    target_max_bits: int,
) -> IntegerSupportReport:
    from finn.dataflow.analysis.integer_dot import (  # noqa: PLC0415
        NumericalFinding,
        analyze_integer_dot_product,
    )

    try:
        premise = mvau_integer_premise_from_operands(
            activation,
            weight,
            accumulator_datatype=accumulator_datatype,
            output_datatype=output_datatype,
            invocation_scope=invocation_scope,
            runtime_writable=runtime_writable,
            runtime_promise=runtime_promise,
        )
        bounds = analyze_integer_dot_product(premise)
        output_carrier = (
            "INT32"
            if IntegerRange(-(1 << 31), (1 << 31) - 1).contains_range(bounds.result)
            else "INT64"
        )
        return check_integer_dot_product_support(
            premise=premise,
            selected_internal_bits=premise.accumulator_type.bit_width,
            selected_output_carrier=output_carrier,
            target_max_accumulator_bits=target_max_bits,
        )
    except (TypeError, ValueError) as error:
        return IntegerSupportReport(
            None,
            (NumericalFinding("integer-premise-invalid", str(error)),),
        )


def execute_mvau_integer(
    *,
    activation: Any,
    weights: Any,
    support_report: IntegerSupportReport,
    fixed_initializer: FrozenInitializer | None,
) -> Any:
    if not support_report.supported or support_report.support is None:
        reasons = ", ".join(item.code for item in support_report.findings) or "unsupported"
        raise ValueError(f"integer MVAU execution has no valid support witness: {reasons}")
    support = support_report.support
    premise = support.premise
    from qonnx.analysis.tensor_value_summary import (  # type: ignore[import-not-found]  # noqa: PLC0415
        summarize_tensor_values,
    )

    activation_tensor = RuntimeIntegerTensor(
        activation,
        array_carrier_name(activation),
        tuple(int(extent) for extent in activation.shape),
        ACTIVATION_IDENTITY,
        premise.invocation_scope,
    )
    weight_digest = (
        None if fixed_initializer is None else summarize_tensor_values(weights).content_digest
    )
    weight_tensor = RuntimeIntegerTensor(
        weights,
        array_carrier_name(weights),
        tuple(int(extent) for extent in weights.shape),
        WEIGHT_IDENTITY,
        premise.invocation_scope,
        weight_digest,
    )
    validated = validate_integer_dot_product_operands(
        activation_tensor,
        weight_tensor,
        support,
    )
    return execute_integer_dot_product(validated, support)


def expected_runtime_weight_count(source: SourceNode) -> int:
    return prod(source.operand("weight").shape)


__all__ = [
    "ACTIVATION_IDENTITY",
    "WEIGHT_IDENTITY",
    "array_carrier_name",
    "carrier_name",
    "check_mvau_integer_support",
    "check_mvau_integer_support_from_operands",
    "execute_mvau_integer",
    "expected_runtime_weight_count",
    "integer_type",
    "integer_graph_profile_fingerprint",
    "mvau_integer_premise",
    "mvau_integer_premise_from_operands",
    "target_accumulator_bits",
]
