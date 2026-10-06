# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Chain: a root of two MatMuls and a thresholding on its channels.

It computes ``thresholds(x @ W1) @ W2``. Each MatMul sits on the root's channels
through its reference inputs; the root declares every channel, reading the
tensor of a MatMul's results from the MatMul's view (``result_tensor``). The
weights are the root's: each weight channel carries them (``contents``), its
tensor the MatMul's ``weight_tensor`` over their range. ``chain()`` is the
Chain configured as the kernel tests, the KernelOp partition tests and the XSim
harness use it.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from finn.core.space import derived, design_space
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.channels import Channel
from finn.kernels.configure import commit
from finn.kernels.matmul import MatMulKernel, column_range
from finn.kernels.thresholding import ThresholdingAxiKernel
from finn.kernels.values.domains import range_dtype, stored_element
from kernels.helpers import FULL_DSP48E2, Root, with_adapter_memories, with_direct_transports

ROWS, INPUTS, HIDDEN, OUTPUTS, PE, SIMD = 3, 4, 4, 4, 2, 2
A = W = DataType["INT3"]
T = DataType["UINT2"]
THRESHOLDS = (tuple((-9 + c, 1 - c, 8 + 2 * c) for c in range(HIDDEN)),)
# The smallest type of H's signedness holding THRESHOLDS: what the ordered pass annotates.
THRESHOLD_DTYPE = DataType["INT5"]
W1 = tuple(tuple((3 * n + 2 * k) % 7 - 3 for n in range(HIDDEN)) for k in range(INPUTS))
W2 = tuple(tuple((2 * n + 5 * k) % 7 - 3 for n in range(OUTPUTS)) for k in range(HIDDEN))
X = tuple(tuple((5 * r + 3 * k) % 8 - 4 for k in range(INPUTS)) for r in range(ROWS))
# Each MatMul's result: its weights' columns over its activations' datatype (K7), in
# the range's smallest encoding: INT6 over [-28, 28] and INT5 over [-15, 12].
H = range_dtype(*column_range(A, W1))
Y = range_dtype(*column_range(T, W2))


def matmul(
    k: int,
    n: int,
    dtype: Any,
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
        x_channel=x_channel,
        w_channel=w_channel,
        y_channel=y_channel,
    )


def weights(values: Any) -> Channel:
    """A MatMul's weight channel, carrying ``values``: from its memory to its core, its
    tensor over their range, in the encoding they need (``stored_element``)."""
    k, n = len(values), len(values[0])
    least, greatest = min(map(min, values)), max(map(max, values))
    tensor = Tensor((k, n), stored_element(W, (least, greatest)))
    return Channel(tensor=tensor, contents=values, platform=FULL_DSP48E2)


class Chain(Root):
    # The input is the root's to state, and the thresholding's output too: a leaf
    # binds its extents from its own ports, so its output tensor is not read from it.
    # Each MatMul's results are its view, which reads only its facts and its weights.
    @derived
    def hidden_tensor(self) -> Tensor:
        return self.first.result_tensor

    @derived
    def y_tensor(self) -> Tensor:
        return self.second.result_tensor

    x = Channel(
        platform=FULL_DSP48E2, tensor=Tensor((ROWS, INPUTS), ScalarEncoding(A)), port="s_axis_0"
    )
    w1 = weights(W1)
    hidden = Channel(tensor=hidden_tensor, platform=FULL_DSP48E2)
    levels = Channel(tensor=Tensor((ROWS, HIDDEN), ScalarEncoding(T)), platform=FULL_DSP48E2)
    w2 = weights(W2)
    y = Channel(tensor=y_tensor, port="m_axis_0", platform=FULL_DSP48E2)
    first = matmul(INPUTS, HIDDEN, A, x_channel=x, w_channel=w1, y_channel=hidden)
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
    second = matmul(HIDDEN, OUTPUTS, T, x_channel=levels, w_channel=w2, y_channel=y)


LAYERS = (("first", "w1"), ("second", "w2"))
"""Each MatMul of the Chain and its weight channel."""


def configure_chain(root: Root, **choices: object) -> Any:
    """``root`` (a Chain) with ``choices`` and each MatMul's packed core configured."""
    choices = choices | {f"{stream}.transport": "direct" for _, stream in LAYERS}
    point = with_adapter_memories(commit(with_direct_transports(design_space(root)), choices))
    # The Decisions inside the candidates just selected, each keyed by its owner.
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
