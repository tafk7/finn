# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A root of two MatMuls and a thresholding on its streams: one flat netlist.

Each MatMul sits on the root's streams through its reference inputs, which it
passes to the kernels that use them: each stream is one real edge, so each
adapter sits on the edge that needs it (a replay before each MatMul's core,
none inside a MatMul). The root's module is one netlist of every leaf; it
computes ``thresholds(x @ W1) @ W2`` in XSim.

The root declares every stream: the tensor of a MatMul's weights and results is
read from the MatMul's views (``weight_tensor``, ``result_tensor``); a tensor
the root states instead must be the one MatMul derives (``carried``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import Available, Rejected, derived, design_space
from finn.dataflow.plan import Step
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import (
    FULL_DSP48E2,
    Root,
    labels,
    with_adapter_memories,
    with_direct_transports,
)
from kernels.xsim import pack, requires_xsim, stream_through

ROOT = Path(__file__).resolve().parents[2]
ROWS, INPUTS, HIDDEN, OUTPUTS, PE, SIMD = 3, 4, 4, 4, 2, 2
A = W = DataType["INT3"]
H = exact_result_dtype(INPUTS, A, W)
T = DataType["UINT2"]
Y = exact_result_dtype(HIDDEN, T, W)
THRESHOLDS = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(HIDDEN)),)
# The smallest type of H's signedness holding THRESHOLDS: what the ordered pass annotates.
THRESHOLD_DTYPE = DataType["INT5"]
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))


def matmul(k: int, n: int, dtype: Any, weights: Any, **streams: Channel) -> MatMulKernel:
    return MatMulKernel(
        m=ROWS,
        n=n,
        k=k,
        activation_dtype=dtype,
        weights_dtype=W,
        platform=FULL_DSP48E2,
        weights=weights,
        **streams,
    )


def weights(k: int, n: int) -> Channel:
    """A MatMul's weight stream: from its memory to its core."""
    return Channel(tensor=Tensor((k, n), ScalarEncoding(W)), platform=FULL_DSP48E2)


class Chain(Root):
    # The input is the root's to state, and the thresholding's output too: a leaf
    # binds its extents from its own ports, so its output tensor is not read from it.
    # Each MatMul's weights and results are its views, which read only its facts.
    @derived
    def w1_tensor(self) -> Tensor:
        return self.first.weight_tensor

    @derived
    def hidden_tensor(self) -> Tensor:
        return self.first.result_tensor

    @derived
    def w2_tensor(self) -> Tensor:
        return self.second.weight_tensor

    @derived
    def y_tensor(self) -> Tensor:
        return self.second.result_tensor

    x = Channel(
        platform=FULL_DSP48E2, tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="s_axis_0"
    )
    w1 = Channel(tensor=w1_tensor, platform=FULL_DSP48E2)
    hidden = Channel(tensor=hidden_tensor, platform=FULL_DSP48E2)
    levels = Channel(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(T)), platform=FULL_DSP48E2)
    w2 = Channel(tensor=w2_tensor, platform=FULL_DSP48E2)
    y = Channel(tensor=y_tensor, port="m_axis_0", platform=FULL_DSP48E2)
    first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, w_stream=w1, y_stream=hidden)
    activate = ThresholdingAxiKernel(
        input_dtype=H,
        threshold_dtype=THRESHOLD_DTYPE,
        thresholds=THRESHOLDS,
        bias=0,
        pe=PE,
        input_stream=hidden,
        output_stream=levels,
        platform=FULL_DSP48E2,
    )
    second = matmul(HIDDEN, OUTPUTS, T, W2, x_stream=levels, w_stream=w2, y_stream=y)
    # Each weight stream carries its MatMul's weights, which the stream's source stores.
    w1.contents = first.weight_values
    w2.contents = second.weight_values


def configured(root: Root, layers: tuple[str, ...] = ("first", "second"), **extra: object) -> Any:
    choices: dict[str, object] = dict(extra)
    for _, stream in zip(layers, ("w1", "w2")):
        choices[f"{stream}.transport"] = "direct"
    point = with_adapter_memories(commit(with_direct_transports(design_space(root)), choices))
    # The Decisions inside the subspaces just selected, each keyed by its owner.
    nested: dict[str, object] = {}
    for layer, stream in zip(layers, ("w1", "w2")):
        nested |= {
            f"{layer}.compute.packed.pe": PE,
            f"{layer}.compute.packed.simd": SIMD,
            f"{layer}.compute.packed.compute_pumping": False,
            f"{layer}.compute.packed.reducer": "tree",
            f"{stream}.source.memstream.ram_style": "auto",
            f"{stream}.source.memstream.pumped_memory": False,
        }
    return with_adapter_memories(commit(point, nested))


# The thresholding's choices: no AXI-Lite, no deep pipeline, memories Vivado's.
ACTIVATE = {
    "activate.use_axilite": False,
    "activate.deep_pipeline": False,
    "activate.ram_style": "auto",
    "activate.ultra_stages": 0,
}


def chain() -> Any:
    return configured(Chain(), **ACTIVATE)


def test_each_edge_carries_its_own_adapter_and_the_netlist_is_flat():
    point = chain()
    # A replay before each MatMul's core, on the root's edge into it; none inside a MatMul.
    assert point.x.plan.steps == point.levels.plan.steps == (Step.REORDER, Step.MARKERS)
    assert point.hidden.plan.steps == ()
    # Each weight memory is its stream's source: a leaf below the stream, in member order.
    assert labels(point.module) == [
        "x.adapter.input_gen.input_gen",
        "w1.source.memstream",
        "levels.adapter.input_gen.input_gen",
        "w2.source.memstream",
        "first.compute.packed",
        "activate",
        "second.compute.packed",
    ]
    # The root's own ports are its boundary streams.
    assert {port.name for port in point.module.pins.ports} == {
        "ap_clk",
        "ap_rst_n",
        "s_axis_0",
        "m_axis_0",
    }
    assert point.module.stem == "finn_chain"


def test_a_matmul_on_a_stream_of_another_tensor_is_refused():
    class Misplaced(Root):
        x = Channel(
            tensor=Tensor((ROWS, INPUTS + 2), ScalarEncoding(A)),
            port="in0_V",
            platform=FULL_DSP48E2,
        )
        w = weights(INPUTS, HIDDEN)
        y = Channel(
            platform=FULL_DSP48E2, tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(H)), port="out0_V"
        )
        first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, w_stream=w, y_stream=y)

    refused = design_space(Misplaced()).first.query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def test_a_matmul_on_a_stream_of_another_element_is_refused():
    """The result type MatMul states (its core's ``result_dtype``) meets the stream's."""

    class Widened(Root):
        x = Channel(
            platform=FULL_DSP48E2, tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="in0_V"
        )
        w = weights(INPUTS, HIDDEN)
        y = Channel(
            tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(DataType["INT32"])),
            port="out0_V",
            platform=FULL_DSP48E2,
        )
        first = matmul(INPUTS, HIDDEN, A, W1, x_stream=x, w_stream=w, y_stream=y)

    refused = design_space(Widened()).first.query(MatMulKernel.carried)
    assert isinstance(refused, Rejected)
    assert {finding.code for finding in refused.findings} == {"matmul-tensor"}


def carried(
    x: ScalarEncoding, w: ScalarEncoding, y: ScalarEncoding, *, known: bool = True
) -> object:
    """MatMul's ``carried`` on streams stating these elements; weights known or not."""
    facts = dict(
        m=ROWS,
        n=HIDDEN,
        k=INPUTS,
        activation_dtype=A,
        weights_dtype=W,
        platform=FULL_DSP48E2,
    )

    class Stated(Root):
        x_ = Channel(tensor=Tensor((ROWS, INPUTS), x), port="in0_V", platform=FULL_DSP48E2)
        w_ = Channel(tensor=Tensor((INPUTS, HIDDEN), w), port="in1_V", platform=FULL_DSP48E2)
        y_ = Channel(tensor=Tensor((ROWS, HIDDEN), y), port="out0_V", platform=FULL_DSP48E2)
        first = MatMulKernel(
            **facts, **({"weights": W1} if known else {}), x_stream=x_, w_stream=w_, y_stream=y_
        )

    return design_space(Stated()).first.query(MatMulKernel.carried)


def test_a_stream_s_values_fit_what_matmul_consumes_and_matmul_s_fit_what_it_produces():
    plain_a, plain_w, plain_h = ScalarEncoding(A), ScalarEncoding(W), ScalarEncoding(H)
    tight = ScalarEncoding(A, (-3, 3))  # INT3 without -4
    accepted = Available(True)
    # Consumed: an upstream with a tighter range feeds MatMul's activations.
    assert carried(tight, plain_w, plain_h) == accepted
    # Produced: MatMul's full result range fits a plainly stated stream, not a tighter one.
    assert carried(plain_a, plain_w, plain_h) == accepted
    narrow_y = ScalarEncoding(H, (0, 1))
    refused = carried(plain_a, plain_w, narrow_y)
    assert isinstance(refused, Rejected) and "y_stream" in str(refused)
    # Known weights (W1 holds -3 to 3) are produced by MatMul's memory: they fit a plain
    # stream (above), not a tighter one.
    refused = carried(plain_a, ScalarEncoding(W, (-2, 2)), plain_h)
    assert isinstance(refused, Rejected) and "w_stream" in str(refused)
    # Unknown weights come from outside: a tighter stream fits MatMul's full range.
    assert carried(plain_a, tight, plain_h, known=False) == accepted
    refused = carried(plain_a, ScalarEncoding(DataType["INT4"]), plain_h, known=False)
    assert isinstance(refused, Rejected) and "w_stream" in str(refused)


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
            "s_axis_0": (
                [
                    pack(X[r][f : f + SIMD], a_bits)
                    for r in range(ROWS)
                    for f in range(0, INPUTS, SIMD)
                ],
                SIMD * a_bits,
            )
        },
        outputs={
            "m_axis_0": (
                [
                    pack(y[r][f : f + PE], y_bits)
                    for r in range(ROWS)
                    for f in range(0, OUTPUTS, PE)
                ],
                PE * y_bits,
            )
        },
    )
