# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain: a root of two MatMuls and a thresholding on its streams.

It computes ``thresholds(x @ W1) @ W2``. Each MatMul sits on the root's streams
through its reference inputs; the root declares every stream, reading the
tensor of a MatMul's weights and results from the MatMul's views
(``weight_tensor``, ``result_tensor``). ``chain()`` is the Chain configured as
the kernel tests, the KernelOp partition tests and the XSim harness use it.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import derived, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, exact_result_dtype
from finn.kernels.thresholding import ThresholdingAxiKernel
from kernels.helpers import FULL_DSP48E2, Root, with_adapter_memories, with_direct_transports

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


def matmul(
    k: int,
    n: int,
    dtype: Any,
    weights: Any,
    *,
    x_channel: Channel,
    w_channel: Channel,
    y_channel: Channel,
) -> MatMulKernel:
    return MatMulKernel(
        m=ROWS,
        n=n,
        k=k,
        activation_dtype=dtype,
        weights_dtype=W,
        platform=FULL_DSP48E2,
        weights=weights,
        x_channel=x_channel,
        w_channel=w_channel,
        y_channel=y_channel,
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
    first = matmul(INPUTS, HIDDEN, A, W1, x_channel=x, w_channel=w1, y_channel=hidden)
    activate = ThresholdingAxiKernel(
        input_dtype=H,
        threshold_dtype=THRESHOLD_DTYPE,
        thresholds=THRESHOLDS,
        bias=0,
        pe=PE,
        input_channel=hidden,
        output_channel=levels,
        platform=FULL_DSP48E2,
    )
    second = matmul(HIDDEN, OUTPUTS, T, W2, x_channel=levels, w_channel=w2, y_channel=y)
    # Each weight stream carries its MatMul's weights, which the stream's source stores.
    w1.contents = first.weight_values
    w2.contents = second.weight_values


LAYERS = (("first", "w1"), ("second", "w2"))
"""Each MatMul of the Chain and its weight stream."""


def configure_chain(root: Root, **choices: object) -> Any:
    """``root`` (a Chain) with ``choices`` and each MatMul's packed core configured."""
    choices = choices | {f"{stream}.transport": "direct" for _, stream in LAYERS}
    point = with_adapter_memories(commit(with_direct_transports(design_space(root)), choices))
    # The Decisions inside the subspaces just selected, each keyed by its owner.
    nested: dict[str, object] = {}
    for layer, stream in LAYERS:
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
    """The Chain, configured."""
    return configure_chain(Chain(), **ACTIVATE)
