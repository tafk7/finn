# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Conditional ordinary MatMul entry into the native DataflowOp graph.

The admitted pattern is an unfused, nonsparse matrix product with an authenticated
fixed weight matrix. Precision is derived from the canonical integer range model;
no folding, implementation alternative or hardware target is selected implicitly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from onnx import helper  # type: ignore[import-not-found]
from qonnx.transformation.base import Transformation  # type: ignore[import-not-found]

from finn.kernels._engine import (
    Absent,
    Answer,
    Decided,
    Finding,
    FindingKind,
    QualifiedPath,
    Unresolved,
)
from finn.parked.dataflow.analysis.integer_dot import IntegerRange, analyze_integer_dot_ranges
from finn.parked.dataflow.kernels.matmul.base import (
    AccumulationMode,
    MatmulInterface,
    accumulator_type_for_bounds,
    computation_profile,
)
from finn.dataflow.datatypes import QONNXDataType
from finn.parked.dataflow.ops.base import DATAFLOW_DOMAIN
from finn.parked.dataflow.ops.mvau.numerics import integer_type
from finn.parked.dataflow.ops.mvau.op import MvauDataflowOp, MvauSpace
from finn.parked.dataflow.ops.persistence import allocate_scope_id
from finn.parked.dataflow.ops.tensor_summary import FrozenInitializer
from finn.parked.dataflow.ops.type_context import producer_type


@dataclass(frozen=True)
class InferAdmission:
    node_name: str
    result_type: Answer[QONNXDataType]


def _answer(code: str, message: str, *, unresolved: bool = False) -> Answer[QONNXDataType]:
    finding = Finding(
        FindingKind.LIMITATION if unresolved else FindingKind.REJECTION,
        code,
        QualifiedPath("op.infer.matmul"),
        message,
    )
    return Unresolved((finding,)) if unresolved else Absent((finding,))


class InferDataflowMatMul(Transformation):  # type: ignore[misc]
    """Replace only semantically admitted fixed integer MatMuls.

    ``admissions`` describes this apply call, including rejected and unresolved
    matched cases. Unmatched nodes and unaccepted cases remain byte-for-byte
    unchanged. Accepted output type is a narrow family/source claim; concrete
    logical and codegen Views may still need choices or reject a hardware profile.
    """

    def __init__(self) -> None:
        self.admissions: tuple[InferAdmission, ...] = ()

    def apply(self, model: Any) -> tuple[Any, bool]:
        admissions: list[InferAdmission] = []
        changed = False
        for index in range(len(model.graph.node)):
            node = model.graph.node[index]
            if node.op_type != "MatMul" or node.domain not in ("", "ai.onnx"):
                continue
            if len(node.input) != 2 or len(node.output) != 1 or node.attribute:
                continue
            weights = tuple(item for item in model.graph.initializer if item.name == node.input[1])
            if len(weights) != 1 or any(item.name == node.input[1] for item in model.graph.input):
                continue
            sparsity = getattr(model, "get_tensor_sparsity", lambda name: None)(node.input[1])
            if sparsity is not None:
                continue
            candidate, answer = self._candidate(
                model, index, FrozenInitializer.from_tensor_proto(weights[0])
            )
            admissions.append(InferAdmission(node.name, answer))
            if candidate is not None and isinstance(answer, Decided):
                model.model.CopyFrom(candidate.model)
                changed = True
        self.admissions = tuple(admissions)
        return model, changed

    def _candidate(
        self, model: Any, index: int, weights: FrozenInitializer
    ) -> tuple[Any | None, Answer[QONNXDataType]]:
        node = model.graph.node[index]
        shape = model.get_tensor_shape(node.input[0])
        info = model.get_tensor_valueinfo(node.input[0])
        if (
            shape is None
            or info is None
            or not info.type.tensor_type.HasField("shape")
            or any(not dim.HasField("dim_value") for dim in info.type.tensor_type.shape.dim)
        ):
            return None, _answer(
                "infer-shape-unresolved", "activation shape is unavailable", unresolved=True
            )
        if len(shape) < 2 or len(weights.shape) != 2 or shape[-1] != weights.shape[0]:
            return None, _answer("infer-matrix-shape", "not an admitted matrix product shape")
        upstream = producer_type(model, node.input[0])
        if upstream is not None and not isinstance(upstream, Decided):
            return None, upstream
        activation_type = (
            upstream.value
            if isinstance(upstream, Decided)
            else model.get_tensor_datatype(node.input[0])
        )
        weight_type = model.get_tensor_datatype(node.input[1])
        profile = computation_profile(
            no_activation=True,
            binary_xnor=False,
            activation_type=activation_type,
            weight_type=weight_type,
        )
        if profile.accumulation is not AccumulationMode.INTEGER:
            return None, _answer(
                "infer-matmul-computation-profile",
                "these operand types select popcount semantics in the DataflowOp; "
                "ordinary MatMul requires product accumulation",
            )
        try:
            activation = integer_type(activation_type)
            integer_type(weight_type)
            raw = weights.array_copy().tolist()
            if any(
                type(value) not in (int, float) or int(value) != value
                for row in raw
                for value in row
            ):
                return None, _answer(
                    "infer-fixed-values", "fixed matrix values must be exact integers"
                )
            columns = tuple(
                tuple(
                    IntegerRange(int(raw[row][column]), int(raw[row][column]))
                    for row in range(weights.shape[0])
                )
                for column in range(weights.shape[1])
            )
            precision = accumulator_type_for_bounds(
                analyze_integer_dot_ranges(activation.value_range, columns)
            )
        except (TypeError, ValueError, OverflowError) as error:
            return None, _answer("infer-integer-semantics", str(error))
        candidate = deepcopy(model)
        replacement = helper.make_node(
            "MvauDataflowOp",
            list(node.input),
            list(node.output),
            domain=DATAFLOW_DOMAIN,
            name=node.name,
            accDataType=precision.name,
            outputDataType=precision.name,
            dataflow_scope_id=allocate_scope_id(),
            dataflow_source_nodes=node.name,
        )
        candidate.graph.node[index].CopyFrom(replacement)
        if not any(item.domain == DATAFLOW_DOMAIN for item in candidate.model.opset_import):
            candidate.model.opset_import.append(helper.make_opsetid(DATAFLOW_DOMAIN, 1))
        candidate.set_tensor_shape(node.output[0], (*shape[:-1], weights.shape[1]))
        operation = candidate.get_customop_wrapper(candidate.graph.node[index])
        if not isinstance(operation, MvauDataflowOp):
            raise TypeError("the registry did not construct the admitted DataflowOp")
        space = operation.space
        if not isinstance(space, MvauSpace):
            raise TypeError("the admitted DataflowOp did not construct its declared Space")
        accepted = cast("Answer[QONNXDataType]", space.operand_type("result"))
        if not isinstance(accepted, Decided):
            return None, accepted
        interface = space.interface_binding.resolve(space)
        if not isinstance(interface, Decided):
            return None, cast("Answer[QONNXDataType]", interface)
        eligible = interface.value.assess_view(MatmulInterface.integer_type_profile).accepted_answer
        if not isinstance(eligible, Decided):
            return None, eligible
        candidate.set_tensor_datatype(node.output[0], accepted.value)
        return candidate, accepted


__all__ = ["InferAdmission", "InferDataflowMatMul"]
