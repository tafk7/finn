# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Admission at conversion: ``ToKernelOps`` asks the kernels about each node a pattern
matches, on its inputs as the nodes before it state them. A node the kernels refuse
stays on the host with their codes; a fact the graph does not state is
``fact-unstated``; a contradiction stops the build."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import KernelOpError, write_target
from finn.transformation.kernels import (
    InferKernelTensors,
    Outcome,
    ToKernelOps,
    between_kernel_ops,
    kernel_ops_report,
    kernel_ops_summary,
)
from kernel_ops.models import DOMAIN, TARGET

GENERAL = "qonnx.custom_op.general"


def model_of(
    nodes: list[NodeProto],
    inputs: dict[str, tuple[list[int], str | None]],
    stored: dict[str, tuple[Any, str | None]],
) -> ModelWrapper:
    """The ``nodes``, ``inputs`` (shape, datatype) and ``stored`` initializers (values,
    datatype; None: unannotated); y the last node's output."""
    infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, s) for name, (s, _) in inputs.items()
    ]
    y = helper.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT, None)
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(nodes, "admission", infos, [y]),
            opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(GENERAL, 1)],
        )
    )
    for name, (values, _) in stored.items():
        model.set_initializer(name, np.asarray(values, dtype=np.float32))
    for name, (_, dtype) in {**inputs, **stored}.items():
        if dtype is not None:
            model.set_tensor_datatype(name, DataType[dtype])
    return model


def matmul(
    x: str | None = "INT4", w: str | None = "INT4", weights: Any = np.eye(4), name: str = "mm"
) -> ModelWrapper:
    """x (3, 4) -> MatMul ``name`` with the initializer w -> y."""
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name=name)
    return model_of([node], {"x": ([3, 4], x)}, {"w": (weights, w)})


def multithreshold_node(x: str, t: str, y: str, name: str = "mt") -> NodeProto:
    return helper.make_node(
        "MultiThreshold", [x, t], [y], name=name, domain=GENERAL, out_dtype="UINT2"
    )


SORTED = [[-1.5, 0.5, 2.5]] * 4
UNSORTED = [[2.5, -1.5, 0.5]] * 4  # count(x >= t) is any order's; the kernel's RTL needs one


def multithreshold(x: str = "INT4", t: str = "FLOAT32", table: Any = SORTED) -> ModelWrapper:
    """x (3, 4) -> MultiThreshold ``mt`` with the (4, 3) initializer t -> y (UINT2)."""
    return model_of([multithreshold_node("x", "t", "y")], {"x": ([3, 4], x)}, {"t": (table, t)})


def convert(model: ModelWrapper) -> tuple[ModelWrapper, ToKernelOps]:
    conversion = ToKernelOps(TARGET)
    return model.transform(conversion), conversion


def refused(model: ModelWrapper) -> list[tuple[str, str]]:
    """The (owner, code) of each finding of the one node, which stays on the host as it
    was."""
    (source,) = model.graph.node
    converted, conversion = convert(model)
    assert list(converted.graph.node) == [source]
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    return [(f.owner, f.code) for f in outcome.findings if f.kind.value == "rejection"]


def test_an_integer_matmul_and_multithreshold_convert() -> None:
    for model, op in ((matmul(), "MatMul"), (multithreshold(), "Thresholding")):
        assert convert(model)[1].outcomes == (Outcome((model.graph.node[0].name,), op),)


def test_a_float_matmul_stays_on_the_host_with_the_kernels_codes() -> None:
    # Each core refuses with the same finding, reported once; float weights that are
    # not integers are no contradiction of FLOAT32.
    for weights in (np.eye(4), np.full((4, 4), 0.5)):
        found = refused(matmul("FLOAT32", "FLOAT32", weights))
        assert found == [("matmul.result_range", "matmul-arithmetic")]


def test_an_int20_matmul_stays_on_the_host_with_each_cores_code() -> None:
    """INT20 activations on a DSP48E2: the packed core's DSP input is too narrow, and
    the INT8 core needs a DSP58. Each case of the ``compute`` Decision says why."""
    assert sorted(refused(matmul("INT20", "INT20"))) == [
        ("matmul.compute.int8_dsp58.core_supported", "dotp-target"),
        ("matmul.compute.packed.core_supported", "dotp-activation-width"),
    ]


def test_an_int32_matmul_in_float32_is_refused_by_its_domain_step_before_the_kernels() -> None:
    """Its partial sums can pass 2**24, where ONNX's float32 MatMul rounds and the
    reference is exact: the op's domain step refuses it."""
    assert refused(matmul("INT32", "INT32")) == [("MatMul", "matmul-container-exceeded")]


def test_a_float_multithreshold_stays_on_the_host_with_the_kernels_codes() -> None:
    assert refused(multithreshold("FLOAT32", "FLOAT32")) == [
        ("activate.table_supported", "threshold-type"),
        ("activate.types_supported", "dtype-family"),
    ]


def test_an_unannotated_input_is_an_unstated_fact() -> None:
    (source,) = matmul(x=None).graph.node
    _, conversion = convert(matmul(x=None))
    (outcome,) = conversion.outcomes
    assert outcome.op is None
    ((kind, owner, code, tensor),) = [
        (f.kind.value, f.owner, f.code, dict(f.details)["tensor"]) for f in outcome.findings
    ]
    assert (kind, owner, code, tensor) == ("limitation", "MatMul", "fact-unstated", "x")


def test_a_contradiction_still_stops_the_conversion() -> None:
    with pytest.raises(KernelOpError, match="mm: w is annotated INT2 and holds values over"):
        convert(matmul(w="INT2", weights=np.full((4, 4), 3.0)))


def test_a_refused_trial_leaves_the_node_as_inference_leaves_it() -> None:
    """Unsorted thresholds: the trial normalizes them (rounded up, annotated INT4) before
    the kernels refuse their order. The node stays, its thresholds as they were, and the
    model is what stating the target and inferring every node would make of it."""
    model = multithreshold(table=UNSORTED)
    expected = model.transform(_Stated()).transform(InferKernelTensors())
    converted, conversion = convert(model)
    (outcome,) = conversion.outcomes
    assert [f.code for f in outcome.findings] == ["threshold-order"]
    assert converted.get_tensor_datatype("t").name == "FLOAT32"
    table = converted.get_initializer("t")
    assert table is not None and np.array_equal(table, np.asarray(UNSORTED, dtype=np.float32))
    assert converted.model.SerializeToString() == expected.model.SerializeToString()


class _Stated(InferKernelTensors):
    """The target stated and the domain imported, as ``ToKernelOps`` does first."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        write_target(model, TARGET)
        model.set_opset_import(DOMAIN, 1)
        return model, False


def test_admission_reads_the_types_the_nodes_before_state() -> None:
    """``h`` carries a stale INT20 annotation, which the second MatMul's cores refuse.
    Conversion states ``h`` from the first MatMul's kernel (its weights' columns) before
    it asks the kernels about the second, so both convert."""
    assert ("matmul.compute.packed.core_supported", "dotp-activation-width") in refused(
        matmul("INT20", "INT4")
    )
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
        helper.make_node("MatMul", ["h", "w"], ["y"], name="b"),
    ]
    model = model_of(nodes, {"x": ([3, 4], "INT4")}, {"w": (np.eye(4), "INT4")})
    model.set_tensor_datatype("h", DataType["INT20"])
    converted, conversion = convert(model)
    assert [outcome.op for outcome in conversion.outcomes] == ["MatMul", "MatMul"]
    assert converted.get_tensor_datatype("h").name == "INT4"


def test_a_matmul_after_a_relu_reads_its_exact_type() -> None:
    """qonnx types Relu(INT8) by its range, UINT7, so the MatMul after it converts (its
    rule before typed the Relu FLOAT32, which the kernels refused: matmul-arithmetic)."""
    nodes = [
        helper.make_node("Relu", ["x"], ["r"], name="relu"),
        helper.make_node("MatMul", ["r", "w"], ["y"], name="mm"),
    ]
    model = model_of(nodes, {"x": ([3, 4], "INT8")}, {"w": (np.eye(4), "INT4")})
    converted, conversion = convert(model)
    assert [outcome.op for outcome in conversion.outcomes] == [None, "MatMul"]
    assert converted.get_tensor_datatype("r").name == "UINT7"


def test_a_refused_node_between_kernel_ops_is_a_host_node_there() -> None:
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"], name="a"),
        multithreshold_node("h", "t", "levels"),
        helper.make_node("MatMul", ["levels", "w"], ["y"], name="b"),
    ]
    model = model_of(
        nodes, {"x": ([3, 4], "INT4")}, {"w": (np.eye(4), "INT4"), "t": (UNSORTED, "FLOAT32")}
    )
    converted, conversion = convert(model)
    assert [outcome.op for outcome in conversion.outcomes] == ["MatMul", None, "MatMul"]
    assert between_kernel_ops(converted) == ("mt",)
    report = kernel_ops_report(converted, conversion.outcomes)
    assert kernel_ops_summary(report)[1:] == ["ToKernelOps:   threshold-order (rejection) 1: mt"]
