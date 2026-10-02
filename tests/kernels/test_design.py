# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A root of two MatMuls and a thresholding on its streams: one flat netlist.

Each MatMul sits on the root's streams through its reference inputs, which it
passes to the kernels that use them: each stream is one real edge, so each
adapter sits on the edge that needs it (a replay before each MatMul's core,
none inside a MatMul). The root's module is one netlist of every leaf; it
computes ``thresholds(x @ W1) @ W2`` in XSim. Each MatMul's control bus is
presented below its node (``first_s_axilite``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import Rejected, design_space
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Bus
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from finn.kernels.streams import BufferedStream, Stream
from finn.kernels.target import DspBlock
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import Root, labels, settled
from kernels.xsim import pack, requires_xsim, stream_through

ROOT = Path(__file__).resolve().parents[2]
ROWS, INPUTS, HIDDEN, OUTPUTS, PE, SIMD = 3, 4, 4, 4, 2, 2
A = W = DataType["INT3"]
H = exact_result_dtype(INPUTS, A, W)
T = DataType["UINT2"]
Y = exact_result_dtype(HIDDEN, T, W)
THRESHOLDS = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(HIDDEN)),)
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))


def matmul(k: int, n: int, dtype: Any, weights: Any, **streams: Stream) -> MatMulKernel:
    return MatMulKernel(
        m=ROWS,
        n=n,
        k=k,
        activation_dtype=dtype,
        weights_dtype=W,
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
        weights=weights,
        **streams,
    )


def weights(k: int, n: int) -> BufferedStream:
    """A MatMul's weight stream: from its memory to its core."""
    return BufferedStream(tensor=Tensor((k, n), ScalarEncoding(W)))


class Chain(Root):
    x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
    w1 = weights(INPUTS, HIDDEN)
    hidden = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)))
    levels = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(T)))
    w2 = weights(HIDDEN, OUTPUTS)
    y = Stream(tensor=Tensor((ROWS, OUTPUTS), ScalarEncoding(Y)), port="out0_V")
    first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, w_stream=w1, y_stream=hidden)
    activate = ThresholdingAxiKernel(
        input_dtype=H,
        threshold_dtype=H,
        thresholds=THRESHOLDS,
        bias=0,
        pe=PE,
        depth_trigger_bram=0,
        depth_trigger_uram=0,
        input_stream=hidden,
        output_stream=levels,
    )
    second = matmul(HIDDEN, OUTPUTS, T, W2, x_stream=levels, w_stream=w2, y_stream=y)


def configured(root: Root, layers: tuple[str, ...] = ("first", "second"), **extra: object) -> Any:
    choices: dict[str, object] = dict(extra)
    for layer, stream in zip(layers, ("w1", "w2")):
        choices |= {f"{layer}.memory": "memstream", f"{stream}.transport": "direct"}
    point = settled(commit(design_space(root), choices))
    # The Decisions inside the subspaces just selected, each keyed by its owner.
    nested: dict[str, object] = {}
    for layer in layers:
        nested |= {
            f"{layer}.compute.packed.pe": PE,
            f"{layer}.compute.packed.simd": SIMD,
            f"{layer}.compute.packed.compute_pumping": False,
            f"{layer}.memory.memstream.ram_style": "auto",
            f"{layer}.memory.memstream.pumped_memory": False,
        }
    return settled(commit(point, nested))


def chain() -> Any:
    return configured(Chain(), **{"activate.use_axilite": False, "activate.deep_pipeline": False})


def test_each_edge_carries_its_own_adapter_and_the_netlist_is_flat():
    point = chain()
    # A replay before each MatMul's core, on the root's edge into it; none inside a MatMul.
    assert point.x.plan.steps == point.levels.plan.steps == (Step.REORDER, Step.MARKERS)
    assert point.hidden.plan.steps == ()
    assert labels(point.module) == [
        "x.adapter.input_gen.input_gen",
        "levels.adapter.input_gen.input_gen",
        "first.compute.packed",
        "first.memory.memstream",
        "activate",
        "second.compute.packed",
        "second.memory.memstream",
    ]
    # The root's own ports are its boundary streams.
    assert {port.name for port in point.module.pins.ports} == {
        "ap_clk",
        "ap_rst_n",
        "in0_V",
        "out0_V",
    }
    assert point.module.stem == "finn_chain"


def test_a_matmul_on_a_stream_of_another_tensor_is_refused():
    class Misplaced(Root):
        x = Stream(tensor=Tensor((ROWS, INPUTS + 2), ScalarEncoding(A)), port="in0_V")
        w = weights(INPUTS, HIDDEN)
        y = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), port="out0_V")
        first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, w_stream=w, y_stream=y)

    refused = design_space(Misplaced()).first.query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def test_a_matmul_on_a_stream_of_another_element_is_refused():
    """The result type MatMul states (its core's ``result_dtype``) meets the stream's."""

    class Widened(Root):
        x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
        w = weights(INPUTS, HIDDEN)
        y = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(DataType["INT32"])), port="out0_V")
        first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, w_stream=w, y_stream=y)

    refused = design_space(Widened()).first.query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


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
    stream_through(
        chain().module,
        tmp_path,
        inputs={
            "in0_V": (
                [
                    pack(X[r][f : f + SIMD], a_bits)
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD)
                ],
                SIMD * a_bits,
            )
        },
        outputs={
            "out0_V": (
                [
                    pack(y[r][f : f + PE], y_bits)
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE)
                ],
                PE * y_bits,
            )
        },
    )


def test_two_matmuls_present_their_control_buses_below_their_nodes():
    class Rewritable(Root):
        x = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V")
        w1 = weights(INPUTS, HIDDEN)
        hidden = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), port="out0_V")
        first = MatMulKernel(
            m=ROWS,
            n=HIDDEN,
            k=INPUTS,
            activation_dtype=A,
            weights_dtype=W,
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
            weights=W1,
            writable_weights=True,
            x_stream=x,
            w_stream=w1,
            y_stream=hidden,
        )
        z = Stream(tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in1_V")
        w2 = weights(INPUTS, HIDDEN)
        out = Stream(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), port="out1_V")
        second = MatMulKernel(
            m=ROWS,
            n=HIDDEN,
            k=INPUTS,
            activation_dtype=A,
            weights_dtype=W,
            target_dsp=DspBlock.DSP48E2,
            target_period_ns=5.0,
            weights=W1,
            writable_weights=True,
            x_stream=z,
            w_stream=w2,
            y_stream=out,
        )

    point = configured(Rewritable())
    buses = [port.name for port in point.module.pins.ports if isinstance(port, Bus)]
    assert buses[-2:] == ["first_s_axilite", "second_s_axilite"]
    exported = [(item.instance, item.port) for item in point.module.fragment.exports]
    assert exported == [
        ("first.memory.memstream", "first_s_axilite"),
        ("second.memory.memstream", "second_s_axilite"),
    ]
