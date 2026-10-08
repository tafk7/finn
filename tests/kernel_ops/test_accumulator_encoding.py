# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The encodings that follow from values, through the kernel path: a MatMul's result is its
range's smallest encoding (``UINT`` when no result is negative), each core accumulates in
it, and the owner of known weights states them narrowed to their values.

Each case is a small ONNX graph (standard ``MatMul`` and ``MultiThreshold``) through
``ToKernelOps``, ``InferKernelTensors`` and every choice ranked by hand, at two lanes
(``kernels.helpers.Lanes``):

- ``unsigned``: UINT2 activations meet non-negative INT4-typed weights, so the first
  MatMul's results are ``UINT``; a Thresholding reads them, and a second MatMul its
  levels;
- ``ternary``: INT8-typed weights in {-1, 0, 1} over INT8 activations: the weights are
  ``INT2`` and the results ``INT10``, where one INT8 x INT8 product is 15 bits;
- ``below``: one 1 a column over INT8 activations: the results are ``INT8``, narrower
  than one product even of the narrowed operands (2 + 8 - 1 bits).

The fast tests read the encodings each node states; the ``xsim`` ones stream a frame
through the partition and compare it with ``execute_onnx`` of the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kernels.helpers import Lanes
from kernels.xsim import requires_xsim
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.kernels.base import kernel_op
from finn.custom_op.kernels.shell import shell_root
from finn.dataflow.tensor import ScalarEncoding
from finn.harness.rtl import pack, stream_through
from finn.kernels.configure import undecided
from finn.kernels.explore import Ranked
from finn.kernels.matmul import column_range
from finn.kernels.values.domains import range_dtype
from finn.platform import resolve_target
from finn.transformation.fpgadataflow.kernel_partitions import partition_facts
from finn.transformation.kernels import (
    ExploreKernelChoices,
    InferKernelTensors,
    ToKernelOps,
)
from finn.transformation.kernels.package import write_boundary_facts

ULTRA96 = resolve_target(part="xczu3eg-sbva484-1-e", period_ns=5.0)  # DSP48E2: the packed core
VCK190 = resolve_target(part="xcvc1902-vsva2197-2MP-e-S", period_ns=5.0)  # DSP58: the INT8 core too
ROWS = 3


@dataclass(frozen=True)
class Layer:
    """A MatMul of ``weights`` typed ``weights_dtype``, then, with ``thresholds``, a
    MultiThreshold to ``levels``."""

    weights: tuple[tuple[int, ...], ...]
    weights_dtype: str
    thresholds: tuple[tuple[int, ...], ...] | None = None
    levels: str = "UINT2"


@dataclass(frozen=True)
class Case:
    activations: str
    layers: tuple[Layer, ...]
    # Each MatMul's result type, its weight channel's element and its core's widths.
    results: tuple[str, ...]
    weights: tuple[str, ...]
    accumulators: tuple[int, ...]


def _unsigned() -> Case:
    w1 = tuple(tuple((n + 2 * k) % 4 for n in range(4)) for k in range(8))
    hidden = range_dtype(*column_range(DataType["UINT2"], w1))
    thresholds = tuple((3 + c, 20 + 2 * c, 40 - c) for c in range(4))
    w2 = tuple(tuple((3 * n + k) % 7 - 3 for n in range(2)) for k in range(4))
    assert hidden.name == "UINT6"  # column sums up to 16, over [0, 3]: [0, 48]
    return Case(
        "UINT2",
        (Layer(w1, "INT4", thresholds), Layer(w2, "INT3")),
        results=("UINT6", "INT6"),
        weights=("INT3 over [0, 3]", "INT3 over [-3, 3]"),
        accumulators=(6, 6),
    )


TERNARY = ((1, -1, 0, 1), (1, 1, -1, 0), (1, 0, 1, -1), (1, -1, 0, 0))
BELOW = ((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))

CASES = {
    "unsigned": _unsigned(),
    "ternary": Case(
        "INT8",
        (Layer(TERNARY, "INT8"),),
        results=("INT10",),  # a column of ones over INT8: [-512, 508]
        weights=("INT2 over [-1, 1]",),
        accumulators=(10,),
    ),
    "below": Case(
        "INT8",
        (Layer(BELOW, "INT8"),),
        results=("INT8",),
        weights=("INT2 over [0, 1]",),
        accumulators=(8,),
    ),
}


def source(case: Case) -> ModelWrapper:
    """The case as a source graph: x (ROWS, K) -> MatMul (-> MultiThreshold) ... -> y."""
    nodes, initializers, annotations = [], {}, {}
    current = "x"
    annotations["x"] = case.activations
    for index, layer in enumerate(case.layers):
        weights, out = f"w{index}", f"h{index}"
        nodes.append(helper.make_node("MatMul", [current, weights], [out], name=f"mm{index}"))
        initializers[weights] = np.array(layer.weights, dtype=np.float32)
        annotations[weights] = layer.weights_dtype
        current = out
        if layer.thresholds is not None:
            table, levels = f"t{index}", f"l{index}"
            nodes.append(
                helper.make_node(
                    "MultiThreshold",
                    [current, table],
                    [levels],
                    name=f"act{index}",
                    domain="qonnx.custom_op.general",
                    out_dtype=layer.levels,
                    out_bias=0.0,
                )
            )
            initializers[table] = np.array(layer.thresholds, dtype=np.float32)
            current = levels
    nodes[-1].output[0] = "y"
    k = len(case.layers[0].weights)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [ROWS, k])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, "encodings", [x], [y])
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="kernel-ops-test",
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("qonnx.custom_op.general", 1),
            ],
        )
    )
    for name, values in initializers.items():
        model.set_initializer(name, values)
    for name, dtype in annotations.items():
        model.set_tensor_datatype(name, DataType[dtype])
    for index, layer in enumerate(case.layers):
        if layer.thresholds is not None:
            # The thresholds in their MatMul's result type, as the ordered pass states it.
            model.set_tensor_datatype(f"t{index}", DataType[case.results[index]])
    return model.transform(InferShapes())


def kernel_ops(case: Case, target: Any = ULTRA96, compute: str | None = None) -> ModelWrapper:
    """The case as KernelOps, inferred, every choice committed at two lanes; ``compute``
    pins each MatMul's core."""
    model = source(case).transform(ToKernelOps(target)).transform(InferKernelTensors())
    if compute is not None:
        for node in model.graph.node:
            if node.op_type == "MatMul":
                kernel_op(model, node).save({"compute": compute})
    return model.transform(ExploreKernelChoices([Ranked(Lanes(2))]))


def _matmuls(model: ModelWrapper) -> list[Any]:
    return [kernel_op(model, node) for node in model.graph.node if node.op_type == "MatMul"]


@pytest.mark.parametrize("name", CASES)
def test_each_result_is_its_ranges_smallest_encoding_and_its_cores_accumulator(name: str) -> None:
    case = CASES[name]
    model = kernel_ops(case)
    for op, result, weights, accumulator in zip(
        _matmuls(model), case.results, case.weights, case.accumulators, strict=True
    ):
        root = op.base()
        # Inference writes the result type on the output; the weights' owner states them
        # narrowed to their values, and the core reads that encoding.
        assert model.get_tensor_datatype(op.onnx_node.output[0]).name == result
        assert root.matmul.result_type.name == result
        assert str(root.w.tensor.element) == weights
        rtl = dict(op.point().matmul.compute.parameters())
        assert rtl["ACCU_WIDTH"] == accumulator
        assert rtl["WEIGHT_WIDTH"] == root.w.tensor.element.bits


def test_a_thresholding_reads_an_unsigned_result() -> None:
    model = kernel_ops(CASES["unsigned"])
    (thresholding,) = [kernel_op(model, n) for n in model.graph.node if n.op_type != "MatMul"]
    point: Any = thresholding.point()
    assert point.x.tensor.element == ScalarEncoding(DataType["UINT6"])
    assert thresholding.verify_node() == []


def test_narrowed_weights_shrink_the_memory_image() -> None:
    """The source stores each weight at its stated width: INT8-typed ternary weights at two
    bits, so a beat of PE x SIMD = 2 x 2 weights is 8 bits, not 32."""
    (op,) = _matmuls(kernel_ops(CASES["ternary"]))
    point: Any = op.point()
    assert point.w.tensor.element.bits == 2
    assert point.w.source.output.axis.payload_bits == 2 * 2 * 2
    assert point.matmul.compute.w.axis.payload_bits == 2 * 2 * 2


def _frames(model: ModelWrapper, x: Any, y: Any) -> dict[str, Any]:
    """The input and the expected output as each boundary port's words, row-major in its
    lanes, from the partition's boundary facts."""
    write_boundary_facts(model)
    inputs, outputs = partition_facts(model)
    words = {}
    for facts, values in ((inputs[0], x), (outputs[0], y)):
        flat = [int(value) for value in values.reshape(-1)]
        lanes, bits = facts["lanes"], facts["element_bits"]
        words[facts["port"]] = (
            [pack(flat[i : i + lanes], bits) for i in range(0, len(flat), lanes)],
            lanes * bits,
        )
    return words


def _computes(case: Case, directory: Path, **options: Any) -> None:
    model = kernel_ops(case, **options)
    root = shell_root(model, model.graph.node)
    assert undecided(root.point, "*") == [] and not root.dropped
    low, high = int(DataType[case.activations].min()), int(DataType[case.activations].max())
    k = len(case.layers[0].weights)
    x = np.random.default_rng(5).integers(low, high + 1, size=(ROWS, k)).astype(np.float32)
    # Each MatMul's extreme rows too: the range's ends, where a narrow accumulator wraps.
    weights = np.array(case.layers[0].weights)
    x[0] = np.where(weights[:, 0] > 0, high, low)
    x[1] = np.where(weights[:, 0] > 0, low, high)
    y = execute_onnx(source(case), {"x": x})["y"]
    words = _frames(model, x, y)
    (port_in,) = [port for port in words if port.startswith("s_axis")]
    (port_out,) = [port for port in words if port.startswith("m_axis")]
    stream_through(
        root.point.module,
        directory,
        inputs={port_in: words[port_in]},
        outputs={port_out: words[port_out]},
    )


@requires_xsim
@pytest.mark.parametrize("name", CASES)
def test_the_packed_core_computes_at_the_ranges_encoding_in_xsim(name: str, tmp_path: Path) -> None:
    _computes(CASES[name], tmp_path)


@requires_xsim
@pytest.mark.parametrize("name", ("ternary", "below"))
def test_the_int8_core_computes_at_the_ranges_encoding_in_xsim(name: str, tmp_path: Path) -> None:
    _computes(CASES[name], tmp_path, target=VCK190, compute="int8_dsp58")
