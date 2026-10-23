# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain (``support.chain``): one flat netlist, its edges' adapters, its channels' tensors.

Each MatMul passes its tensors to the kernels that read them, and presents its ports'
contracts on the network's channels: each channel is one real edge, so each adapter sits
on the edge that needs it (a replay before each MatMul's core, none inside a MatMul).
The network's module is one netlist of every leaf; it computes
``thresholds(x @ W1) @ W2`` in XSim. An input tensor given to a MatMul must be the one
MatMul derives (``carried``); a MatMul states its results, and a channel holds its
tensor to them (``channel-tensor``).
"""

from __future__ import annotations

from pathlib import Path

from qonnx.core.datatype import DataType
from qonnx.core.ranges import TensorRange
from support.chain import (
    ACTIVATE,
    HIDDEN,
    INPUTS,
    OUTPUTS,
    PE,
    ROWS,
    SIMD,
    STEM,
    THRESHOLDS,
    W1,
    W2,
    A,
    H,
    W,
    X,
    Y,
    chain,
    configure_chain,
    matmul,
    network,
)
from support.helpers import FULL_DSP48E2, built, cycles_of, labels
from support.xsim import requires_xsim

from finn.core.rtlsim_bench import pack_lanes, stream_through
from finn.core.space import Available, Rejected, design_space
from finn.dataflow.steps import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel
from finn.kernels.matmul import MatMulKernel
from finn.kernels.values.semantics import IntegerTensorValue
from finn.toolchain.launcher import host_toolchain


def test_each_edge_carries_its_own_adapter_and_the_netlist_is_flat():
    point = chain()
    x, levels = point.member("x"), point.member("levels")
    # A replay before each MatMul's core, on the edge into it; none inside a MatMul.
    assert x.steps.kinds == levels.steps.kinds == (Step.REORDER, Step.MARKERS)
    assert point.member("hidden").steps.kinds == ()
    module = built(point, STEM)
    # Each weight memory is its channel's source: a leaf below the channel, in the order
    # the network declares its members.
    assert labels(module) == [
        "x.adapter.input_gen.input_gen",
        "w1.source.memstream",
        "levels.adapter.input_gen.input_gen",
        "w2.source.memstream",
        "first.compute.packed",
        "activate",
        "second.compute.packed",
    ]
    # The module's own ports are its boundary channels.
    assert {port.name for port in module.abi.pins} == {
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    }
    assert module.stem == "finn_chain"


def test_a_matmul_on_a_stream_of_another_tensor_is_refused():
    # Its K is the activations' last axis: two more than its known weights' rows.
    misplaced = matmul(W1, Tensor((ROWS, INPUTS + 2), ScalarEncoding(A)))
    refused = design_space(misplaced).query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def test_a_matmul_reads_its_extents_and_datatypes_from_its_tensors():
    """Its extents and its activations' datatype are its tensors', stated once; at any
    rank, its rows are the activations' leading axes, and its results keep them."""
    point = design_space(matmul(W1, Tensor((1, ROWS, INPUTS), ScalarEncoding(A))))
    assert (point.m, point.n, point.k) == (ROWS, HIDDEN, INPUTS)
    assert point.activation_dtype == A
    assert point.y_tensor.shape == (1, ROWS, HIDDEN)
    assert point.query(MatMulKernel.carried) == Available(True)


def hidden_on(element: ScalarEncoding) -> object:
    """The Chain's first MatMul's results channel on a tensor of ``element``: whether the
    channel takes what the MatMul states."""

    # The thresholding reads the channel's tensor, as a graph gives its consumer.
    stated = configure_chain(network(Tensor((ROWS, HIDDEN), element)), **ACTIVATE)
    return stated.member("hidden").query(Channel.well_formed)


def test_a_matmuls_results_fit_their_channels_tensor():
    """MatMul's results (``y_tensor``: H over its columns' range) fit a plainly stated
    channel of their type, not one of another type or a tighter range."""
    assert chain().member("first").y_tensor.element == ScalarEncoding(H, TensorRange(-28, 28))
    assert hidden_on(ScalarEncoding(H)) == Available(True)
    for element in (ScalarEncoding(DataType["INT32"]), ScalarEncoding(H, TensorRange(0, 1))):
        refused = hidden_on(element)
        assert isinstance(refused, Rejected)
        assert {finding.code for finding in refused.findings} == {"channel-tensor"}


def carried(
    x: ScalarEncoding, w: ScalarEncoding, *, known: bool = True, declared: bool | None = None
) -> object:
    """MatMul's ``carried`` on tensors of these elements; its weights W1 (known weights,
    whose tensor MatMul states over the datatype they are declared in) or a tensor of
    ``w``. ``declared``: whether ``weights_dtype`` is given (as known weights state it)."""
    declares = known if declared is None else declared
    weights = (
        {"w_contents": IntegerTensorValue.of(W1)}
        if known
        else {"w_tensor": Tensor((INPUTS, HIDDEN), w)}
    )
    stated = MatMulKernel(
        platform=FULL_DSP48E2,
        x_tensor=Tensor((ROWS, INPUTS), x),
        **({"weights_dtype": W} if declares else {}),
        **weights,
    )
    return design_space(stated).query(MatMulKernel.carried)


def test_a_stream_s_values_fit_what_matmul_consumes_and_matmul_s_fit_what_it_produces():
    plain_a, plain_w = ScalarEncoding(A), ScalarEncoding(W)
    tight = ScalarEncoding(A, TensorRange(-3, 3))  # INT3 without -4
    accepted = Available(True)
    # Consumed: the activations' datatype is their tensor's, at any range.
    assert carried(tight, plain_w) == accepted
    assert carried(plain_a, plain_w) == accepted
    # Weights are consumed: known, MatMul states their tensor over their range, in the
    # datatype they are declared in; streamed, their datatype is their tensor's.
    assert carried(plain_a, ScalarEncoding(W, TensorRange(-3, 3))) == accepted
    assert carried(plain_a, tight, known=False) == accepted
    assert carried(plain_a, ScalarEncoding(DataType["INT4"]), known=False) == accepted
    # A datatype stated twice is refused: streamed weights' is their tensor's.
    restated = carried(plain_a, plain_w, known=False, declared=True)
    assert isinstance(restated, Rejected) and "weights_dtype" in str(restated)


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
        built(design, STEM),
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
        cycles=cycles_of(design).value,
        toolchain=host_toolchain(),
    )
