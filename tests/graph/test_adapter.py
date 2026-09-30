# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The graph adapter: a small ONNX model of two MatMuls as a Design.

The model's weights are initializers; the adapter reads them as the kernels'
weights, stored ``(k, n)`` as ONNX stores them, and each MatMul's exact result
type flows to the next. The configured Design computes what ONNX computes, in
XSim, and the model rewritten for FINN carries each kernel's ``MVAU``
attributes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.kernels.configure import commit
from finn.graph import FINN_DOMAIN, GraphError, finn_model, graph_design
from finn.kernels.matmul import exact_result_dtype
from finn.kernels.target import DspBlock
from kernels.helpers import settled
from kernels.xsim import pack, requires_xsim, stream_through

ROWS, INPUTS, HIDDEN, OUTPUTS, PE, SIMD = 3, 4, 4, 2, 2, 2
A = W = DataType["INT3"]
W1 = np.array([[(3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)] for k in range(INPUTS)])
W2 = np.array([[(2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)] for k in range(HIDDEN)])
X = np.array([[(5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)] for r in range(ROWS)])


def model(*, second_weights: bool = True, hidden_type: str = "FLOAT32") -> ModelWrapper:
    """x (1, 3, 4) -> MatMul W1 -> h -> MatMul W2 -> y."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ROWS, INPUTS])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, ROWS, OUTPUTS])
    h = helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, ROWS, HIDDEN])
    w1 = helper.make_tensor_value_info("w1", TensorProto.FLOAT, [INPUTS, HIDDEN])
    w2 = helper.make_tensor_value_info("w2", TensorProto.FLOAT, [HIDDEN, OUTPUTS])
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["h"], name="first"),
        helper.make_node("MatMul", ["h", "w2"], ["y"], name="second"),
    ]
    inputs = [x] if second_weights else [x, w2]
    known = [h, w1, w2] if second_weights else [h, w1]
    graph = helper.make_graph(nodes, "two_matmuls", inputs, [y], value_info=known)
    wrapped = ModelWrapper(qonnx_make_model(graph, producer_name="test"))
    wrapped.set_initializer("w1", W1.astype(np.float32))
    if second_weights:
        wrapped.set_initializer("w2", W2.astype(np.float32))
    for name in ("x", "w1", "w2"):
        wrapped.set_tensor_datatype(name, DataType["INT3"])
    wrapped.set_tensor_datatype("h", DataType[hidden_type])
    return wrapped


def configured(wrapped: ModelWrapper, *, fused: bool = True) -> Any:
    design = graph_design(wrapped, target_dsp=DspBlock.DSP48E2, target_period_ns=5.0)
    choices: dict[str, object] = {}
    for _, kernel in design.kernels:
        choices |= {
            f"{kernel}.fused": fused,
            f"{kernel}.weight_stream.transport": "direct",
        }
        if f"{kernel}.memory" not in dict(design.pinned):
            choices[f"{kernel}.memory"] = "memstream"
    point = settled(commit(design.point, choices))
    # The Decisions inside the subspaces just selected, each keyed by its owner.
    nested: dict[str, object] = {}
    for _, kernel in design.kernels:
        nested |= {
            f"{kernel}.compute.packed.pe": PE,
            f"{kernel}.compute.packed.simd": SIMD,
            f"{kernel}.compute.packed.compute_pumping": False,
        }
        if getattr(point, kernel).supplied == "memstream":
            nested[f"{kernel}.memory.memstream.ram_style"] = "auto"
            nested[f"{kernel}.memory.memstream.pumped_memory"] = False
    return design, settled(commit(point, nested))


H = exact_result_dtype(INPUTS, A, W)
Y = exact_result_dtype(HIDDEN, H, W)


def test_each_matmul_is_a_kernel_on_the_streams_of_its_tensors():
    design, point = configured(model())
    assert design.kernels == (("first", "mm_first"), ("second", "mm_second"))
    # Leading axes are rows; the exact result type flows to the next MatMul.
    assert point.t_x.tensor.shape == (ROWS, INPUTS)
    assert point.t_h.tensor.element.datatype_name == H.name
    assert point.mm_second.activation_dtype == H
    assert point.mm_first.weights == tuple(tuple(int(v) for v in row) for row in W1)
    top = {port.name for port in point.structure.structure.top_abi.ports}
    assert top == {"ap_clk", "ap_rst_n", "in0_V", "out0_V"}


def test_weights_without_an_initializer_are_a_stream_and_need_no_memory():
    design, point = configured(model(second_weights=False))
    assert dict(design.pinned) == {"mm_second.memory": "none"}
    top = {port.name for port in point.structure.structure.top_abi.ports}
    assert "in1_V" in top


def test_what_the_graph_cannot_be_is_refused():
    with pytest.raises(GraphError, match="narrower than the exact result"):
        graph_design(model(hidden_type="INT4"), target_dsp=DspBlock.DSP48E2, target_period_ns=5.0)
    wrapped = model()
    wrapped.graph.node[1].op_type = "Add"
    with pytest.raises(GraphError, match="Add has no kernel yet"):
        graph_design(wrapped, target_dsp=DspBlock.DSP48E2, target_period_ns=5.0)


def test_the_finn_model_carries_each_kernels_mvau_attributes():
    design, point = configured(model())
    rewritten = finn_model(model(), point, design.kernels)
    first = rewritten.graph.node[0]
    assert (first.op_type, first.domain) == ("MVAU", FINN_DOMAIN)
    assert rewritten.get_tensor_datatype("h") == H
    # FINN's custom operators read FINN_ROOT when they load.
    os.environ.setdefault("FINN_ROOT", str(Path(__file__).resolve().parents[2]))
    op = getCustomOp(first)
    read = {name: op.get_nodeattr(name) for name in ("MW", "MH", "SIMD", "PE", "mem_mode")}
    assert read == {
        "MW": INPUTS,
        "MH": HIDDEN,
        "SIMD": SIMD,
        "PE": PE,
        "mem_mode": "internal_decoupled",
    }
    assert op.get_nodeattr("outputDataType") == H.name
    assert op.get_nodeattr("weightDataType") == W.name


@requires_xsim
@pytest.mark.parametrize("fused", (True, False))
def test_the_design_computes_what_onnx_computes(tmp_path, fused):
    _, point = configured(model(), fused=fused)
    y = execute_onnx(model(), {"x": X.reshape(1, ROWS, INPUTS).astype(np.float32)})["y"]
    y = y.reshape(ROWS, OUTPUTS).astype(int)
    stream_through(
        point.structure.requirements,
        tmp_path,
        inputs={
            "in0_V": (
                [
                    pack(X[r][f : f + SIMD].tolist(), A.bitwidth())
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD)
                ],
                SIMD * A.bitwidth(),
            )
        },
        outputs={
            "out0_V": (
                [
                    pack(y[r][f : f + PE].tolist(), Y.bitwidth())
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE)
                ],
                PE * Y.bitwidth(),
            )
        },
    )
