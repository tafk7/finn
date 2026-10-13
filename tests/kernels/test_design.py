# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain (``kernels.chain``): one flat netlist, its edges' adapters, its channels' tensors.

Each MatMul passes the root's tensors to the kernels that read them, and presents
their ends on the root's channels: each channel is one real edge, so each adapter
sits on the edge that needs it (a replay before each MatMul's core, none inside a
MatMul). The root's module is one netlist of every leaf; it computes
``thresholds(x @ W1) @ W2`` in XSim. A tensor the root states instead of reading
it from a MatMul must be the one MatMul derives (``carried``).
"""

from __future__ import annotations

from pathlib import Path

from qonnx.core.datatype import DataType

from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.space import Available, Rejected, design_space
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.matmul import MatMulKernel
from finn.toolchain.launcher import host_toolchain
from kernels.chain import (
    HIDDEN,
    INPUTS,
    OUTPUTS,
    PE,
    ROWS,
    SIMD,
    THRESHOLDS,
    W1,
    W2,
    A,
    H,
    W,
    X,
    Y,
    chain,
    matmul,
)
from kernels.helpers import FULL_DSP48E2, exact_result_dtype, labels
from kernels.xsim import requires_xsim


def test_each_edge_carries_its_own_adapter_and_the_netlist_is_flat():
    point = chain()
    # A replay before each MatMul's core, on the root's edge into it; none inside a MatMul.
    assert point.x.plan.steps == point.levels.plan.steps == (Step.REORDER, Step.MARKERS)
    assert point.hidden.plan.steps == ()
    # Each weight memory is its channel's source: a leaf below the channel, in member order.
    assert labels(point.module) == [
        "x.adapter.input_gen.input_gen",
        "w1.source.memstream",
        "levels.adapter.input_gen.input_gen",
        "w2.source.memstream",
        "first.compute.packed",
        "activate",
        "second.compute.packed",
    ]
    # The root's own ports are its boundary channels.
    assert {port.name for port in point.module.abi.pins} == {
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    }
    assert point.module.stem == "finn_chain"


def test_a_matmul_on_a_stream_of_another_tensor_is_refused():
    misplaced = matmul(
        INPUTS,
        HIDDEN,
        A,
        W1,
        Tensor((ROWS, INPUTS + 2), ScalarEncoding(A)),
        Tensor((ROWS, HIDDEN), ScalarEncoding(H)),
    )
    refused = design_space(misplaced).query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def test_a_matmul_on_a_stream_of_another_element_is_refused():
    """The result type MatMul states (its core's ``result_dtype``) meets its tensor's."""
    widened = matmul(
        INPUTS,
        HIDDEN,
        A,
        W1,
        Tensor((ROWS, INPUTS), ScalarEncoding(A)),
        Tensor((ROWS, HIDDEN), ScalarEncoding(DataType["INT32"])),
    )
    refused = design_space(widened).query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def carried(
    x: ScalarEncoding, w: ScalarEncoding, y: ScalarEncoding, *, known: bool = True
) -> object:
    """MatMul's ``carried`` on tensors of these elements; its weights W1 (known weights,
    whose tensor MatMul states) or a tensor of ``w``."""
    facts = dict(
        m=ROWS,
        n=HIDDEN,
        k=INPUTS,
        activation_dtype=A,
        weights_dtype=W,
        platform=FULL_DSP48E2,
    )
    weights = {"w_contents": W1} if known else {"w_tensor": Tensor((INPUTS, HIDDEN), w)}
    stated = MatMulKernel(
        **facts,
        x_tensor=Tensor((ROWS, INPUTS), x),
        y_tensor=Tensor((ROWS, HIDDEN), y),
        **weights,
    )
    return design_space(stated).query(MatMulKernel.carried)


def test_a_stream_s_values_fit_what_matmul_consumes_and_matmul_s_fit_what_it_produces():
    plain_a, plain_w, plain_h = ScalarEncoding(A), ScalarEncoding(W), ScalarEncoding(H)
    tight = ScalarEncoding(A, (-3, 3))  # INT3 without -4
    accepted = Available(True)
    # Consumed: an upstream with a tighter range feeds MatMul's activations.
    assert carried(tight, plain_w, plain_h) == accepted
    # Produced: MatMul's full result range fits a plainly stated channel, not a tighter one.
    assert carried(plain_a, plain_w, plain_h) == accepted
    narrow_y = ScalarEncoding(H, (0, 1))
    refused = carried(plain_a, plain_w, narrow_y)
    assert isinstance(refused, Rejected) and "y_tensor" in str(refused)
    # Weights are consumed: known, MatMul states their tensor over their range; unknown,
    # their tensor's values fit MatMul's datatype.
    assert carried(plain_a, ScalarEncoding(W, (-3, 3)), plain_h) == accepted
    # Unknown, the result is the datatypes' (K7), wider than W1's columns'.
    unknown_h = ScalarEncoding(exact_result_dtype(INPUTS, A, W))
    assert isinstance(carried(plain_a, tight, plain_h, known=False), Rejected)
    assert carried(plain_a, tight, unknown_h, known=False) == accepted
    refused = carried(plain_a, ScalarEncoding(DataType["INT4"]), unknown_h, known=False)
    assert isinstance(refused, Rejected) and "w_tensor" in str(refused)


@requires_xsim
def test_the_chain_computes_in_xsim(tmp_path: Path) -> None:
    hidden = [
        [sum(X[r][k] * W1[k][n] for k in range(INPUTS)) for n in range(HIDDEN)] for r in range(ROWS)
    ]
    levels = [
        [sum(t <= hidden[r][c] for t in THRESHOLDS[0][c]) for c in range(HIDDEN)]
        for r in range(ROWS)
    ]
    y = [
        [sum(levels[r][k] * W2[k][n] for k in range(HIDDEN)) for n in range(OUTPUTS)]
        for r in range(ROWS)
    ]
    a_bits, y_bits = A.bitwidth(), Y.bitwidth()
    design = chain()
    stream_through(
        design.module,
        tmp_path,
        inputs={
            "s_axis_0": (
                [
                    pack_lanes(X[r][f : f + SIMD], a_bits)
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD)
                ],
                SIMD * a_bits,
            )
        },
        outputs={
            "m_axis_0": (
                [
                    pack_lanes(y[r][f : f + PE], y_bits)
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE)
                ],
                PE * y_bits,
            )
        },
        cycles=design.cycles,
        toolchain=host_toolchain(),
    )
